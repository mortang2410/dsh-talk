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
    expect(out![0]).toBeCloseTo(0, 6)
    expect(out![3200]).toBeCloseTo(0.002, 6)
    expect(out![9599]).toBeCloseTo(0.002, 6)
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
})
