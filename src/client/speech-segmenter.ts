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
 * Audio kept before the block that opens a segment: at most 600 ms. The energy floor opens the
 * segment on the first block above it, so an unvoiced start (such as "s" or "f") below the floor
 * sits before that block. The bound is a choice: quieter audio more than PRE_ROLL_MS before the
 * onset is not kept, so a soft lead-in longer than PRE_ROLL_MS loses its earliest part.
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
  /** The open segment's audio, converted to the STT rate as its blocks arrive (above 16 kHz), or null between segments. */
  open: SttStream | null
  /** 16 kHz samples already converted for the open segment (input above 16 kHz). Closing converts only the rest. */
  convertedSamples: number
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
  // Build the filter when the capture starts, not in the first onset callback. Measured cold in Node 22.23.4 (2026-10-25, one machine): 4.8 to 5.1 ms at 48 kHz, 6.6 to 10.6 ms at 96 kHz. The times depend on the machine and the Node version.
  if (options.sampleRate > STT_SAMPLE_RATE) filterTableFor(options.sampleRate)
  return {
    sampleRate: options.sampleRate,
    threshold: options.threshold,
    silenceMs: options.silenceMs,
    history: [],
    historySamples: 0,
    open: null,
    convertedSamples: 0,
    pauseMs: 0,
    idleMs: 0,
    segments: 0,
  }
}

function preRollSamples(seg: Segmenter): number {
  return Math.round((PRE_ROLL_MS / 1000) * seg.sampleRate)
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

/** Close the open segment and return its audio at the STT rate. Only the samples not yet converted are converted here. */
function closeOpen(seg: Segmenter, stream: SttStream): Float32Array {
  seg.open = null
  seg.pauseMs = 0
  seg.convertedSamples = 0
  return finishStt(stream)
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
    let stream: SttStream | null = seg.open
    if (stream === null) {
      // The onset starts with the history: the newest PRE_ROLL_MS of non-speech audio. Its blocks are
      // queued, and queueStt pushes a few per call, so the onset call does not convert all of them.
      stream = createSttStream(seg.sampleRate)
      for (const earlier of seg.history) stream.queued.push(earlier)
      seg.open = stream
      seg.segments += 1
    }
    queueStt(stream, block)
    seg.convertedSamples = stream.outLength
    seg.history = []
    seg.historySamples = 0
    seg.pauseMs = 0
  } else {
    pushHistory(seg, block)
    const stream = seg.open
    if (stream !== null) {
      queueStt(stream, block)
      seg.convertedSamples = stream.outLength
      seg.pauseMs = advanceSilence(seg.pauseMs, blockMs, false)
      if (silenceTrips(seg.pauseMs, seg.silenceMs)) events.push({ kind: 'segment', samples: closeOpen(seg, stream) })
    }
  }
  if (seg.segments > 0 && silenceTrips(seg.idleMs, CAPTURE_SILENCE_TIMEOUT_MS)) events.push({ kind: 'timeout' })
  return events
}

/** Close the open segment, if any, and return its audio at the STT rate. Used when the capture stops. */
export function flushSegment(seg: Segmenter): Float32Array | null {
  const stream = seg.open
  return stream === null ? null : closeOpen(seg, stream)
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
 * The open segment's audio at the input rate, with its 16 kHz output built as the blocks arrive.
 * `raw` holds the input from absolute index `rawStart` on (not used at 16 kHz); input that no later
 * output reads is dropped. `chunks` hold the 16 kHz samples that are final so far, `outLength` of them. `queued`
 * holds blocks that are not in `raw` yet (see queueStt).
 */
interface SttStream {
  readonly fromRate: number
  raw: Float32Array
  rawStart: number
  rawLength: number
  chunks: Float32Array[]
  outLength: number
  queued: Float32Array[]
}

function createSttStream(fromRate: number): SttStream {
  return { fromRate, raw: new Float32Array(0), rawStart: 0, rawLength: 0, chunks: [], outLength: 0, queued: [] }
}

function appendRaw(stream: SttStream, block: Float32Array): void {
  const needed = stream.rawLength + block.length
  if (needed > stream.raw.length) {
    const grown = new Float32Array(Math.max(needed, 2 * stream.raw.length, 4096))
    grown.set(stream.raw.subarray(0, stream.rawLength))
    stream.raw = grown
  }
  stream.raw.set(block, stream.rawLength)
  stream.rawLength += block.length
}

/** 16 kHz samples per output chunk. Output grows a chunk at a time, so no growth step copies the whole segment. */
const OUT_CHUNK = 65536

function appendOut(stream: SttStream, value: number): void {
  const at = stream.outLength % OUT_CHUNK
  if (at === 0) stream.chunks.push(new Float32Array(OUT_CHUNK))
  stream.chunks[stream.chunks.length - 1]![at] = value
  stream.outLength += 1
}

/** The 16 kHz samples so far as one array: each chunk is copied once, in order. */
function joinOut(stream: SttStream): Float32Array {
  const out = new Float32Array(stream.outLength)
  for (let i = 0; i < stream.chunks.length; i += 1) {
    const start = i * OUT_CHUNK
    out.set(stream.chunks[i]!.subarray(0, Math.min(OUT_CHUNK, out.length - start)), start)
  }
  return out
}

/**
 * Add the 16 kHz outputs that are final. Without `final`, an output waits until the input it reads
 * has arrived. With `final`, the input ends at its last sample, and the last outputs repeat that
 * sample, as the one-shot conversion always did. Only input above 16 kHz is filtered here.
 */
function emitStt(stream: SttStream, final: boolean): void {
  const ratio = stream.fromRate / STT_SAMPLE_RATE
  const { count, half, taps } = filterTableFor(stream.fromRate)
  const raw = stream.raw
  const origin = stream.rawStart
  const end = origin + stream.rawLength
  const limit = Math.floor(end / ratio)
  for (let index = stream.outLength; index < limit; index += 1) {
    const position = index * ratio
    let base = Math.floor(position)
    let phase = Math.round((position - base) * FILTER_PHASES)
    if (phase === FILTER_PHASES) {
      base += 1
      phase = 0
    }
    if (!final && base + half >= end) break
    const row = phase * count
    let sum = 0
    for (let k = 0; k < count; k += 1) {
      const at = Math.min(Math.max(base - half + k, 0), end - 1)
      sum += taps[row + k]! * raw[at - origin]!
    }
    appendOut(stream, sum)
  }
  // The next output reads input from floor(outLength * ratio) - half on, so anything before that can go.
  const drop = Math.floor(stream.outLength * ratio) - half - origin
  if (drop > 0) {
    stream.raw.copyWithin(0, drop, stream.rawLength)
    stream.rawLength -= drop
    stream.rawStart += drop
  }
}

/**
 * Add the 16 kHz outputs that are final, for input below 16 kHz. Each output interpolates two input
 * samples in the same order as the one-shot conversion, so the samples match it exactly. Without
 * `final`, an output waits until its second sample has arrived. With `final`, the input ends at its
 * last sample, and that sample repeats.
 */
function emitLinear(stream: SttStream, final: boolean): void {
  const ratio = stream.fromRate / STT_SAMPLE_RATE
  const raw = stream.raw
  const origin = stream.rawStart
  const end = origin + stream.rawLength
  const limit = Math.floor(end / ratio)
  for (let index = stream.outLength; index < limit; index += 1) {
    const position = index * ratio
    const low = Math.floor(position)
    if (!final && low + 1 >= end) break
    const fraction = position - low
    const first = low < end ? raw[low - origin]! : 0
    const second = low + 1 < end ? raw[low + 1 - origin]! : first
    appendOut(stream, first + (second - first) * fraction)
  }
  // The next output reads input from floor(outLength * ratio) on, so anything before that can go.
  const drop = Math.floor(stream.outLength * ratio) - origin
  if (drop > 0) {
    raw.copyWithin(0, drop, stream.rawLength)
    stream.rawLength -= drop
    stream.rawStart += drop
  }
}

/**
 * Feed one block of input at the segment's rate. The outputs that are ready are built now. At 16 kHz
 * the input already is the output, so its samples go straight to the output chunks.
 */
function pushSttBlock(stream: SttStream, block: Float32Array): void {
  if (stream.fromRate === STT_SAMPLE_RATE) {
    for (let index = 0; index < block.length; index += 1) appendOut(stream, block[index]!)
    return
  }
  appendRaw(stream, block)
  if (stream.fromRate > STT_SAMPLE_RATE) emitStt(stream, false)
  else emitLinear(stream, false)
}

/** Blocks pushed into an open stream per feed call. Above 16 kHz, one block costs about 0.16 ms in Node at 96 kHz, so this keeps each call short. */
const STT_BLOCKS_PER_FEED = 4

/**
 * Queue one block of the open segment behind the blocks already waiting, then push up to
 * STT_BLOCKS_PER_FEED blocks from the front. The stream sees every block in order, so its output is
 * the same as if each block were pushed when it arrived. Only the timing of the work changes.
 */
function queueStt(stream: SttStream, block: Float32Array): void {
  stream.queued.push(block)
  for (let pushed = 0; pushed < STT_BLOCKS_PER_FEED && stream.queued.length > 0; pushed += 1) {
    pushSttBlock(stream, stream.queued.shift()!)
  }
}

/** Finish a conversion: the 16 kHz samples for all the input, with the last outputs repeating the final sample. */
function finishStt(stream: SttStream): Float32Array {
  while (stream.queued.length > 0) pushSttBlock(stream, stream.queued.shift()!)
  if (stream.fromRate > STT_SAMPLE_RATE) emitStt(stream, true)
  else if (stream.fromRate < STT_SAMPLE_RATE) emitLinear(stream, true)
  return joinOut(stream)
}

/**
 * Convert mono samples to the STT rate. Above 16 kHz the samples are low-passed first, so content
 * above the new 8 kHz limit cannot fold back into the speech band, and then read at the new rate.
 * Edge samples are repeated past each end. Input below 16 kHz is linearly interpolated, which
 * creates no alias. Converting a segment block by block gives the same samples as this.
 */
export function toSttRate(samples: Float32Array, fromRate: number): Float32Array {
  if (fromRate === STT_SAMPLE_RATE) return samples
  const stream = createSttStream(fromRate)
  pushSttBlock(stream, samples)
  return finishStt(stream)
}
