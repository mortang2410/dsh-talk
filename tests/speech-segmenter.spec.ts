/**
 * The pure speech segmenter: per-block voice-activity detection, a pre-roll that keeps the
 * audio before a detected onset, and the resampling to the 16 kHz the STT route takes.
 */

import { describe, expect, it } from 'vitest'
import { createSegmenter, feedBlock, flushSegment, PRE_ROLL_MS, toSttRate, type SegmenterEvent } from '../src/client/speech-segmenter.ts'

const RATE = 48000
/** One 20 ms block at 48 kHz. */
const BLOCK_SAMPLES = 960

/** Blocks that together last `ms`, each holding a constant level (so its RMS is that level). */
function blocks(ms: number, level: number): Float32Array[] {
  return Array.from({ length: Math.round(ms / 20) }, () => new Float32Array(BLOCK_SAMPLES).fill(level))
}

function feedAll(seg: ReturnType<typeof createSegmenter>, parts: Float32Array[][]): SegmenterEvent[] {
  const events: SegmenterEvent[] = []
  for (const part of parts) for (const block of part) events.push(...feedBlock(seg, block))
  return events
}

function segmentsOf(events: SegmenterEvent[]): Float32Array[] {
  return events.flatMap(event => (event.kind === 'segment' ? [event.samples] : []))
}

function newSegmenter(silenceMs = 1500): ReturnType<typeof createSegmenter> {
  return createSegmenter({ sampleRate: RATE, threshold: 0.01, silenceMs })
}

/**
 * The one-shot conversion from before streaming (commit cdffc2a): all the audio at once, with the
 * same windowed-sinc low-pass. It shares no code with the segmenter, so it independently checks the
 * streamed output. Input above 16 kHz only.
 */
function oneShotToSttRate(samples: Float32Array, fromRate: number): Float32Array {
  const ratio = fromRate / 16000
  let count = Math.round((5.5 * fromRate) / 1000)
  if (count % 2 === 0) count += 1
  const half = (count - 1) / 2
  const reach = half + 1
  const cutoff = 7500 / fromRate
  const phases = 256
  const sinc = (x: number): number => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))
  const taps = new Float64Array(phases * count)
  for (let phase = 0; phase < phases; phase += 1) {
    const fraction = phase / phases
    let sum = 0
    for (let k = 0; k < count; k += 1) {
      const distance = fraction + half - k
      const window = 0.42 + 0.5 * Math.cos((Math.PI * distance) / reach) + 0.08 * Math.cos((2 * Math.PI * distance) / reach)
      const value = 2 * cutoff * sinc(2 * cutoff * distance) * window
      taps[phase * count + k] = value
      sum += value
    }
    for (let k = 0; k < count; k += 1) taps[phase * count + k] = taps[phase * count + k]! / sum
  }
  const length = Math.floor(samples.length / ratio)
  const out = new Float32Array(length)
  for (let index = 0; index < length; index += 1) {
    const position = index * ratio
    let base = Math.floor(position)
    let phase = Math.round((position - base) * phases)
    if (phase === phases) {
      base += 1
      phase = 0
    }
    let sum = 0
    for (let k = 0; k < count; k += 1) {
      const at = Math.min(Math.max(base - half + k, 0), samples.length - 1)
      sum += taps[phase * count + k]! * samples[at]!
    }
    out[index] = sum
  }
  return out
}

/**
 * Exact equality. On a mismatch the message names the first differing sample: a plain toEqual on
 * arrays this long builds a full diff, which takes minutes.
 */
function expectSameSamples(actual: Float32Array | null, expected: Float32Array): void {
  expect(actual).not.toBeNull()
  expect(actual!.length).toBe(expected.length)
  let index = 0
  while (index < expected.length && actual![index] === expected[index]) index += 1
  expect(index, `first differing sample at ${index}`).toBe(expected.length)
}

describe('pre-roll', () => {
  it('starts a segment PRE_ROLL_MS before the first speech block, so audio under the energy floor just before the onset is kept', () => {
    expect(PRE_ROLL_MS).toBe(600)
    // 600 ms at 16 kHz is 9600 samples: literal, so a changed constant cannot pass by changing the expectation too.
    const seg = newSegmenter()
    // 1 s of digital silence, then 400 ms of a soft onset (0.002, below the 0.01 floor), then speech.
    const events = feedAll(seg, [blocks(1000, 0), blocks(400, 0.002), blocks(500, 0.2), blocks(1600, 0)])
    const [out] = segmentsOf(events)
    expect(out).toBeDefined()
    // The 600 ms before the speech block: 200 ms of silence, then 400 ms of the soft onset.
    // The anti-alias filter (toSttRate) smooths each level change over about 44 samples at 16 kHz
    // (the sample at 3200 reads 0.0013), and it rings about 2 ms ahead of the loud onset (the sample
    // at 9599 reads -0.0114). So the level checks sit 100 samples inside each run, out of that reach.
    expect(out![0]).toBeCloseTo(0, 6)
    expect(out![3300]).toBeCloseTo(0.002, 6)
    expect(out![9500]).toBeCloseTo(0.002, 6)
    // The first speech sample sits exactly PRE_ROLL_MS into the 16 kHz segment.
    expect(out!.findIndex(value => value > 0.1)).toBe(9600)
  })

  it('never keeps more than PRE_ROLL_MS of silence before the onset, however long the silence was', () => {
    const seg = newSegmenter()
    const events = feedAll(seg, [blocks(30000, 0), blocks(200, 0.2), blocks(1600, 0)])
    const [out] = segmentsOf(events)
    expect(out!.findIndex(value => value > 0.1)).toBe(9600)
  })
})

describe('pre-roll after a segment closes', () => {
  it('a soft onset in the silence that closes one segment is in the pre-roll of the next', () => {
    // Review round 1 F1. 16 kHz, 1600-sample blocks (100 ms each), silenceMs 1000, so no resampling.
    const seg = createSegmenter({ sampleRate: 16000, threshold: 0.01, silenceMs: 1000 })
    const block = (level: number): Float32Array => new Float32Array(1600).fill(level)
    const run = [
      block(0.2), // speech: opens segment 1
      ...Array.from({ length: 9 }, () => block(0)), // 900 ms of silence
      block(0.002), // soft, under the floor: the 1000 ms pause closes segment 1 on this block
      block(0.002), // soft, after segment 1 closed
      block(0.002), // soft, after segment 1 closed
      block(0.2), // speech: opens segment 2
      ...Array.from({ length: 12 }, () => block(0)), // 1200 ms of silence closes segment 2
    ]
    const [, second] = segmentsOf(feedAll(seg, [run]))
    expect(second).toBeDefined()
    // 600 ms of pre-roll (9600 samples): blocks 7 to 9 are silence (samples 0 to 4799), blocks 10 to 12 are the soft onset (4800 to 9599).
    expect(second![0]).toBe(0)
    expect(second![4799]).toBe(0)
    expect(second![4800]).toBeCloseTo(0.002, 6)
    expect(second![9599]).toBeCloseTo(0.002, 6)
    expect(second!.findIndex(value => value > 0.1)).toBe(9600)
  })

  it('keeps exactly PRE_ROLL_MS of history when the block size does not divide it', () => {
    // Review round 1 F2. 1024-sample blocks at 48 kHz do not divide 28800 samples (600 ms), so the
    // history must be trimmed inside a block, not only between blocks.
    const seg = createSegmenter({ sampleRate: 48000, threshold: 0.01, silenceMs: 1500 })
    feedAll(seg, [Array.from({ length: 30 }, () => new Float32Array(1024))])
    expect(seg.historySamples).toBe(28800)
    // 28800 pre-roll + 1024 speech + 71 silent blocks (1500 ms) = 102528 samples at 48 kHz, which is 34176 at 16 kHz.
    const events = feedAll(seg, [[new Float32Array(1024).fill(0.2)], Array.from({ length: 71 }, () => new Float32Array(1024))])
    const [out] = segmentsOf(events)
    expect(out!.length).toBe(34176)
  })
})

describe('per-block detection', () => {
  it('finds a 100 ms word wherever it falls, not only at a fixed sample point', () => {
    const seg = newSegmenter()
    // 1234 ms is not a multiple of 500 ms, so the burst does not line up with any old detector tick.
    const events = feedAll(seg, [blocks(1234, 0), blocks(100, 0.1), blocks(1600, 0)])
    expect(segmentsOf(events)).toHaveLength(1)
  })

  it('a pause shorter than silenceMs does not split the segment', () => {
    const seg = newSegmenter()
    const events = feedAll(seg, [blocks(500, 0.2), blocks(1000, 0), blocks(500, 0.2), blocks(1600, 0)])
    expect(segmentsOf(events)).toHaveLength(1)
  })

  it('a pause of silenceMs ends the segment, and speech after it starts a new one', () => {
    const seg = newSegmenter()
    const first = feedAll(seg, [blocks(500, 0.2), blocks(1600, 0)])
    expect(segmentsOf(first)).toHaveLength(1)
    const second = feedAll(seg, [blocks(300, 0.2), blocks(1600, 0)])
    expect(segmentsOf(second)).toHaveLength(1)
    expect(seg.segments).toBe(2)
  })
})

describe('flushSegment', () => {
  it('returns the words of an open segment when the capture stops mid-speech, and leaves nothing open', () => {
    const seg = newSegmenter()
    const events = feedAll(seg, [blocks(1000, 0), blocks(500, 0.2)])
    expect(events).toEqual([])
    const tail = flushSegment(seg)
    expect(tail).not.toBeNull()
    expect(tail!.length).toBe(17600)
    expect(flushSegment(seg)).toBeNull()
  })

  it('returns null when no segment is open', () => {
    expect(flushSegment(newSegmenter())).toBeNull()
  })
})

describe('120-second timeout', () => {
  it('does not time out while no segment has started', () => {
    const seg = newSegmenter()
    const events = feedAll(seg, [blocks(130000, 0)])
    expect(events.some(event => event.kind === 'timeout')).toBe(false)
  })

  it('times out only after 120 s of silence since the last speech, once a segment exists', () => {
    const seg = newSegmenter()
    expect(feedAll(seg, [blocks(200, 0.2), blocks(119000, 0)]).some(event => event.kind === 'timeout')).toBe(false)
    expect(feedAll(seg, [blocks(2000, 0)]).some(event => event.kind === 'timeout')).toBe(true)
  })
})

describe('toSttRate', () => {
  it('keeps the level and divides the length by the integer ratio for 48 kHz input', () => {
    const out = toSttRate(new Float32Array(48).fill(0.5), 48000)
    expect(out.length).toBe(16)
    expect(out.every(value => Math.abs(value - 0.5) < 1e-6)).toBe(true)
  })

  it('resamples a non-integer ratio such as 44.1 kHz to exactly one second at 16 kHz', () => {
    const out = toSttRate(new Float32Array(44100).fill(0.25), 44100)
    expect(out.length).toBe(16000)
    expect(out.every(value => Math.abs(value - 0.25) < 1e-6)).toBe(true)
  })

  it('returns 16 kHz input unchanged', () => {
    const samples = new Float32Array([0.1, 0.2, 0.3])
    expect(toSttRate(samples, 16000)).toBe(samples)
  })

  // Review round 1 F3. Measured on the middle of each output, away from the edges where the filter settles.
  function toneOf(hz: number, rate: number, seconds: number): Float32Array {
    return Float32Array.from({ length: Math.round(rate * seconds) }, (_, n) => Math.sin((2 * Math.PI * hz * n) / rate))
  }
  function levelOf(samples: Float32Array): number {
    const middle = samples.subarray(1600, samples.length - 1600)
    let sum = 0
    for (const value of middle) sum += value * value
    return Math.sqrt(sum / middle.length)
  }

  it.each([44100, 48000])('keeps a 10 kHz tone out of the 16 kHz band at %i Hz, so it does not fold back to 6 kHz', rate => {
    // 10 kHz is above the 8 kHz limit of 16 kHz audio and must be filtered before the rate drops.
    expect(levelOf(toSttRate(toneOf(10000, rate, 1), rate))).toBeLessThan(0.01)
  })

  it.each([44100, 48000])('keeps a 1 kHz tone at its full level at %i Hz', rate => {
    // A full-scale sine has RMS 0.7071.
    const level = levelOf(toSttRate(toneOf(1000, rate, 1), rate))
    expect(level).toBeGreaterThan(0.69)
    expect(level).toBeLessThan(0.72)
  })
})

describe('conversion while a segment is open', () => {
  it('converts the open segment as its blocks arrive, so closing it does not convert the whole segment', () => {
    // Review round 2 R2-1. The conversion used to run only at close: about 0.9 s on the page's main
    // thread for 120 s of speech at 48 kHz. 10 s of speech in 20 ms blocks is 480000 samples at 48 kHz,
    // which gives 160000 samples at 16 kHz; 264 + 1 taps reach 132 input samples ahead, so 159956 are final.
    const seg = newSegmenter()
    feedAll(seg, [blocks(10000, 0.2)])
    expect(seg.convertedSamples).toBeGreaterThan(159900)
    expect(flushSegment(seg)!.length).toBe(160000)
  })

  it.each([44100, 48000, 96000])('gives the same samples as the one-shot conversion of all the audio, for any block sizes, past the first output chunk, at %i Hz', rate => {
    // Deterministic pseudo-random audio and block sizes, so a failure reproduces. The threshold is 0,
    // so every block is speech and the one open segment holds all of the audio.
    let state = 12345
    const random = (): number => {
      state = (state * 1103515245 + 12345) >>> 0
      return state / 4294967296
    }
    // 5 s at 16 kHz is 80000 output samples, more than the 65536 of one output chunk.
    const total = rate * 5
    const audio = Float32Array.from({ length: total }, () => random() * 2 - 1)
    const seg = createSegmenter({ sampleRate: rate, threshold: 0, silenceMs: 1500 })
    let at = 0
    while (at < total) {
      const size = Math.min(total - at, 1 + Math.floor(random() * 2000))
      feedBlock(seg, audio.slice(at, at + size))
      at += size
    }
    expectSameSamples(flushSegment(seg), oneShotToSttRate(audio, rate))
  })

  it.each([44100, 48000])('a segment that starts with a pre-roll gives the same samples as the one-shot conversion of the pre-roll and the speech, at %i Hz', rate => {
    // Quiet audio under the energy floor, then speech, in random block sizes. The segment starts with the
    // newest PRE_ROLL_MS of the quiet audio, so its expected samples are the one-shot conversion of exactly that.
    let state = 777
    const random = (): number => {
      state = (state * 1103515245 + 12345) >>> 0
      return state / 4294967296
    }
    const quiet = Float32Array.from({ length: 960 * 100 }, () => (random() * 2 - 1) * 0.005)
    const speech = Float32Array.from({ length: rate * 2 }, () => random() * 2 - 1)
    const seg = createSegmenter({ sampleRate: rate, threshold: 0.01, silenceMs: 1500 })
    for (let at = 0; at < quiet.length; at += 960) feedBlock(seg, quiet.slice(at, at + 960))
    let at = 0
    while (at < speech.length) {
      const size = Math.min(speech.length - at, 256 + Math.floor(random() * 1792))
      feedBlock(seg, speech.slice(at, at + size))
      at += size
    }
    const preRoll = quiet.subarray(quiet.length - Math.round((PRE_ROLL_MS / 1000) * rate))
    const joined = new Float32Array(preRoll.length + speech.length)
    joined.set(preRoll)
    joined.set(speech, preRoll.length)
    expectSameSamples(flushSegment(seg), oneShotToSttRate(joined, rate))
  })

  it('converts a pre-roll over several feed calls, not all in the onset call', () => {
    // Review round 3, R3-2. At 96 kHz the pre-roll is 60 blocks of 960 samples. Converting all of it in
    // the onset call took about 9 ms in Node (about 18 ms with the filter table built in that call).
    const seg = createSegmenter({ sampleRate: 96000, threshold: 0.01, silenceMs: 1500 })
    feedAll(seg, [Array.from({ length: 60 }, () => new Float32Array(960))])
    feedBlock(seg, new Float32Array(960).fill(0.2))
    // The whole pre-roll is 9600 samples at 16 kHz. The onset call must convert only a small part of it.
    expect(seg.convertedSamples).toBeLessThan(4800)
  })

  it('keeps the whole pre-roll when the segment is flushed right after its onset, at 96 kHz', () => {
    // Review round 4, R4-3. The onset queues the pre-roll blocks, and each feed call converts only a few of
    // them. Stopping right after the onset must still convert the queued blocks: without that, most of the
    // pre-roll is lost. Quiet audio fills the 600 ms pre-roll exactly (60 blocks of 960 at 96 kHz).
    let state = 4242
    const random = (): number => {
      state = (state * 1103515245 + 12345) >>> 0
      return state / 4294967296
    }
    const quiet = Float32Array.from({ length: 960 * 60 }, () => (random() * 2 - 1) * 0.005)
    const onset = Float32Array.from({ length: 960 }, () => random() * 2 - 1)
    const seg = createSegmenter({ sampleRate: 96000, threshold: 0.01, silenceMs: 1500 })
    for (let at = 0; at < quiet.length; at += 960) feedBlock(seg, quiet.slice(at, at + 960))
    feedBlock(seg, onset)
    const joined = new Float32Array(quiet.length + onset.length)
    joined.set(quiet)
    joined.set(onset, quiet.length)
    expectSameSamples(flushSegment(seg), oneShotToSttRate(joined, 96000))
  })
})
