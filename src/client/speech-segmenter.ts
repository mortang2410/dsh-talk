/**
 * Pure speech segmenter for a segmented capture. It sees every audio block in order, decides
 * when a speech run starts and ends, and keeps PRE_ROLL_MS of the audio that came before the
 * onset, so a soft word start under the energy floor still reaches the transcript. No browser
 * APIs: the microphone button feeds it the blocks an AudioWorklet posts.
 *
 * @module dsh-talk/client/speech-segmenter
 */

import { STT_SAMPLE_RATE } from './wav.ts'
import { advanceSilence, CAPTURE_SILENCE_TIMEOUT_MS, isSpeechFrame, rmsOf, silenceTrips } from './vad.ts'

/**
 * Audio kept before the block that opens a segment. The energy floor opens the segment on
 * the first block above it, so an unvoiced start (such as "s" or "f") below the floor sits
 * before that block. 600 ms covers that lead-in with margin.
 */
export const PRE_ROLL_MS = 600

export interface SegmenterOptions {
  /** Sample rate of the blocks fed in, in hertz (the AudioContext rate). */
  sampleRate: number
  /** RMS energy at or above which a block counts as speech (`record.vad.energyThreshold`). */
  threshold: number
  /** Silence, in milliseconds, that ends a segment (`record.vad.silenceMs`). */
  silenceMs: number
}

export interface Segmenter {
  readonly sampleRate: number
  readonly threshold: number
  readonly silenceMs: number
  /**
   * The newest non-speech blocks since the last speech block, kept to PRE_ROLL_MS. The next
   * onset starts with them, so the closing silence of the previous segment can be in it.
   */
  history: Float32Array[]
  historySamples: number
  /** Blocks of the open segment, or null between segments. */
  open: Float32Array[] | null
  /** Silence, in milliseconds, inside the open segment. */
  pauseMs: number
  /** Silence, in milliseconds, since the last speech block (the 120-second timeout). */
  idleMs: number
  /** Segments started so far. */
  segments: number
}

/**
 * What one block produced. `segment` carries the finished segment as 16 kHz mono samples.
 * `timeout` means 120 seconds passed with no speech after at least one segment.
 */
export type SegmenterEvent = { kind: 'segment'; samples: Float32Array } | { kind: 'timeout' }

export function createSegmenter(options: SegmenterOptions): Segmenter {
  return {
    sampleRate: options.sampleRate,
    threshold: options.threshold,
    silenceMs: options.silenceMs,
    history: [],
    historySamples: 0,
    open: null,
    pauseMs: 0,
    idleMs: 0,
    segments: 0,
  }
}

function preRollSamples(seg: Segmenter): number {
  return Math.round((PRE_ROLL_MS / 1000) * seg.sampleRate)
}

function concatBlocks(blocks: Float32Array[]): Float32Array {
  const total = blocks.reduce((sum, block) => sum + block.length, 0)
  const out = new Float32Array(total)
  let offset = 0
  for (const block of blocks) {
    out.set(block, offset)
    offset += block.length
  }
  return out
}

/**
 * Remember a non-speech block. The history never holds more than PRE_ROLL_MS: whole blocks are
 * dropped from the front, and the oldest block that is still partly needed is cut to the exact count.
 */
function pushHistory(seg: Segmenter, block: Float32Array): void {
  seg.history.push(block)
  seg.historySamples += block.length
  let excess = seg.historySamples - preRollSamples(seg)
  while (excess > 0) {
    const oldest = seg.history[0]!
    if (oldest.length <= excess) {
      seg.history.shift()
      seg.historySamples -= oldest.length
      excess -= oldest.length
    } else {
      seg.history[0] = oldest.subarray(excess)
      seg.historySamples -= excess
      excess = 0
    }
  }
}

/** Close the open segment and return its audio at the STT rate. */
function closeOpen(seg: Segmenter): Float32Array {
  const blocks = seg.open ?? []
  seg.open = null
  seg.pauseMs = 0
  return toSttRate(concatBlocks(blocks), seg.sampleRate)
}

/**
 * Feed one audio block, in order. A block above the energy floor while no segment is open opens
 * one, and the segment starts with the history: the newest PRE_ROLL_MS of non-speech audio. That
 * history can include the closing silence of the previous segment, so a soft onset there reaches
 * the next segment too. A pause of `silenceMs` after speech ends the segment. Every block is
 * judged, so no audio between two samples is skipped.
 */
export function feedBlock(seg: Segmenter, block: Float32Array): SegmenterEvent[] {
  const blockMs = (block.length / seg.sampleRate) * 1000
  const speech = isSpeechFrame(rmsOf(block), seg.threshold)
  seg.idleMs = advanceSilence(seg.idleMs, blockMs, speech)
  const events: SegmenterEvent[] = []
  if (speech) {
    if (seg.open === null) {
      seg.open = [...seg.history, block]
      seg.segments += 1
    } else {
      seg.open.push(block)
    }
    seg.history = []
    seg.historySamples = 0
    seg.pauseMs = 0
  } else {
    pushHistory(seg, block)
    if (seg.open !== null) {
      seg.open.push(block)
      seg.pauseMs = advanceSilence(seg.pauseMs, blockMs, false)
      if (silenceTrips(seg.pauseMs, seg.silenceMs)) events.push({ kind: 'segment', samples: closeOpen(seg) })
    }
  }
  if (seg.segments > 0 && silenceTrips(seg.idleMs, CAPTURE_SILENCE_TIMEOUT_MS)) events.push({ kind: 'timeout' })
  return events
}

/** Close the open segment, if any, and return its audio at the STT rate. Used when the capture stops. */
export function flushSegment(seg: Segmenter): Float32Array | null {
  return seg.open === null ? null : closeOpen(seg)
}

/** The anti-alias low-pass cuts above this frequency, in hertz: below the 8 kHz limit of 16 kHz audio. */
const ANTI_ALIAS_CUTOFF_HZ = 7500
/** A fractional input position is rounded to one of this many filter phases. */
const FILTER_PHASES = 256

/**
 * A low-pass for one input rate. Phase `p` weighs input sample `base - half + k` by
 * `taps[p * count + k]`, for `k` from 0 to `count - 1`.
 */
interface FilterTable {
  readonly count: number
  readonly half: number
  readonly taps: Float64Array
}

const filterTables = new Map<number, FilterTable>()

function sinc(x: number): number {
  return x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)
}

/**
 * The windowed-sinc filter for decimation from `fromRate`. The sinc is cut at ANTI_ALIAS_CUTOFF_HZ
 * and tapered by a Blackman window whose main lobe is about 1 kHz wide, which puts the stopband
 * above 8 kHz. Each phase is normalised so a constant input stays constant.
 */
function filterTableFor(fromRate: number): FilterTable {
  const cached = filterTables.get(fromRate)
  if (cached !== undefined) return cached
  let count = Math.round((5.5 * fromRate) / 1000)
  if (count % 2 === 0) count += 1
  const half = (count - 1) / 2
  const reach = half + 1
  const cutoff = ANTI_ALIAS_CUTOFF_HZ / fromRate
  const taps = new Float64Array(FILTER_PHASES * count)
  for (let phase = 0; phase < FILTER_PHASES; phase += 1) {
    const fraction = phase / FILTER_PHASES
    let sum = 0
    for (let k = 0; k < count; k += 1) {
      // Distance from the output position to this tap. It stays inside the window's reach.
      const distance = fraction + half - k
      const window = 0.42 + 0.5 * Math.cos((Math.PI * distance) / reach) + 0.08 * Math.cos((2 * Math.PI * distance) / reach)
      const value = 2 * cutoff * sinc(2 * cutoff * distance) * window
      taps[phase * count + k] = value
      sum += value
    }
    for (let k = 0; k < count; k += 1) taps[phase * count + k] = taps[phase * count + k]! / sum
  }
  const table: FilterTable = { count, half, taps }
  filterTables.set(fromRate, table)
  return table
}

/**
 * Convert mono samples to the STT rate. Above 16 kHz the samples are low-passed first, so content
 * above the new 8 kHz limit cannot fold back into the speech band, and then read at the new rate.
 * Edge samples are repeated past each end. Input at or below 16 kHz is linearly interpolated,
 * which creates no alias.
 */
export function toSttRate(samples: Float32Array, fromRate: number): Float32Array {
  if (fromRate === STT_SAMPLE_RATE) return samples
  const ratio = fromRate / STT_SAMPLE_RATE
  const length = Math.floor(samples.length / ratio)
  const out = new Float32Array(length)
  if (ratio < 1) {
    for (let index = 0; index < length; index += 1) {
      const position = index * ratio
      const low = Math.floor(position)
      const fraction = position - low
      const first = samples[low] ?? 0
      const second = samples[low + 1] ?? first
      out[index] = first + (second - first) * fraction
    }
    return out
  }
  const { count, half, taps } = filterTableFor(fromRate)
  for (let index = 0; index < length; index += 1) {
    const position = index * ratio
    let base = Math.floor(position)
    let phase = Math.round((position - base) * FILTER_PHASES)
    if (phase === FILTER_PHASES) {
      base += 1
      phase = 0
    }
    const row = phase * count
    let sum = 0
    for (let k = 0; k < count; k += 1) {
      const at = Math.min(Math.max(base - half + k, 0), samples.length - 1)
      sum += taps[row + k]! * samples[at]!
    }
    out[index] = sum
  }
  return out
}
