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
  /** The newest blocks heard while no segment is open, kept to PRE_ROLL_MS. */
  idle: Float32Array[]
  idleSamples: number
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
    idle: [],
    idleSamples: 0,
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

/** Keep only the last `count` samples. */
function tailOf(samples: Float32Array, count: number): Float32Array {
  return samples.length <= count ? samples : samples.subarray(samples.length - count)
}

/** Remember an idle block, dropping the oldest blocks once the rest still cover the pre-roll. */
function pushIdle(seg: Segmenter, block: Float32Array): void {
  seg.idle.push(block)
  seg.idleSamples += block.length
  const keep = preRollSamples(seg)
  while (seg.idle.length > 0 && seg.idleSamples - seg.idle[0]!.length >= keep) {
    seg.idleSamples -= seg.idle.shift()!.length
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
 * Feed one audio block, in order. A block above the energy floor while no segment is open
 * opens one, starting PRE_ROLL_MS before it. A pause of `silenceMs` after speech ends it.
 * Every block is judged, so no audio between two samples is skipped.
 */
export function feedBlock(seg: Segmenter, block: Float32Array): SegmenterEvent[] {
  const blockMs = (block.length / seg.sampleRate) * 1000
  const speech = isSpeechFrame(rmsOf(block), seg.threshold)
  seg.idleMs = advanceSilence(seg.idleMs, blockMs, speech)
  const events: SegmenterEvent[] = []
  if (seg.open === null) {
    if (speech) {
      seg.open = [tailOf(concatBlocks(seg.idle), preRollSamples(seg)), block]
      seg.idle = []
      seg.idleSamples = 0
      seg.pauseMs = 0
      seg.segments += 1
    } else {
      pushIdle(seg, block)
    }
  } else {
    seg.open.push(block)
    if (speech) {
      seg.pauseMs = 0
    } else {
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

/**
 * Convert mono samples to the STT rate. An integer ratio (48 kHz to 16 kHz) averages each
 * group of samples, which removes most of the content above the new Nyquist limit. Any other
 * ratio uses linear interpolation.
 */
export function toSttRate(samples: Float32Array, fromRate: number): Float32Array {
  if (fromRate === STT_SAMPLE_RATE) return samples
  const ratio = fromRate / STT_SAMPLE_RATE
  if (Number.isInteger(ratio)) {
    const out = new Float32Array(Math.floor(samples.length / ratio))
    for (let index = 0; index < out.length; index += 1) {
      let sum = 0
      for (let step = 0; step < ratio; step += 1) sum += samples[index * ratio + step]!
      out[index] = sum / ratio
    }
    return out
  }
  const length = Math.floor(samples.length / ratio)
  const out = new Float32Array(length)
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
