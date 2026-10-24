/**
 * WAV conversion for the composer microphone. MediaRecorder records in the
 * browser's own container (WebM/Opus in Chrome and Firefox, MP4 in Safari),
 * and the transcription route refuses those containers. The recording is
 * therefore decoded by the browser at 16 kHz, mixed to mono, and written as
 * 16-bit PCM WAV. Only the decode step touches a browser API; the rest is pure.
 *
 * @module dsh-talk/client/wav
 */

/** Sample rate of every WAV this client sends. The transcription route takes 16 kHz. */
export const STT_SAMPLE_RATE = 16000

/** The decoded buffer fields this module reads (an AudioBuffer subset). */
export interface DecodedAudioLike {
  readonly numberOfChannels: number
  getChannelData(channel: number): Float32Array
}

/** The OfflineAudioContext face used to decode (its decodeAudioData resamples to the context rate). */
export interface OfflineDecoderLike {
  decodeAudioData(data: ArrayBuffer): Promise<DecodedAudioLike>
}

/**
 * Average every channel into one mono channel. A single channel is returned as is.
 *
 * @param channels - one Float32Array per channel, all the same length.
 * @returns the mono samples, each in -1..1.
 */
export function mixToMono(channels: ReadonlyArray<Float32Array>): Float32Array {
  const first = channels[0]
  if (first === undefined) return new Float32Array(0)
  if (channels.length === 1) return first
  const out = new Float32Array(first.length)
  for (const channel of channels) {
    for (let index = 0; index < out.length; index += 1) {
      out[index] = (out[index] ?? 0) + (channel[index] ?? 0) / channels.length
    }
  }
  return out
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index))
  }
}

/**
 * Encode mono samples as a 16-bit PCM WAV at 16 kHz. The header declares
 * format 1 (PCM), 1 channel, 16000 Hz, 16 bits. Samples are clamped to -1..1.
 * The caller must pass samples already at 16 kHz; {@link recordingToWav16kMono}
 * guarantees that by decoding through a 16 kHz context.
 *
 * @param samples - mono samples at 16 kHz, each nominally in -1..1.
 * @returns the complete WAV file bytes (44-byte header plus data).
 */
export function encodeWav16kMono(samples: Float32Array): Uint8Array {
  const dataBytes = samples.length * 2
  const out = new Uint8Array(44 + dataBytes)
  const view = new DataView(out.buffer)
  writeAscii(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  writeAscii(view, 8, 'WAVE')
  writeAscii(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, STT_SAMPLE_RATE, true)
  view.setUint32(28, STT_SAMPLE_RATE * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeAscii(view, 36, 'data')
  view.setUint32(40, dataBytes, true)
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index] ?? 0))
    view.setInt16(44 + index * 2, value < 0 ? Math.round(value * 0x8000) : Math.round(value * 0x7fff), true)
  }
  return out
}

/**
 * The browser's decoder: an OfflineAudioContext with one channel at 16 kHz.
 * Its decodeAudioData resamples the decoded audio to the context's rate.
 *
 * @returns the decoder, or undefined when the browser has no OfflineAudioContext.
 */
export function createDecoder(): OfflineDecoderLike | undefined {
  const Ctor = (globalThis as unknown as {
    OfflineAudioContext?: new (channels: number, length: number, sampleRate: number) => OfflineDecoderLike
  }).OfflineAudioContext
  if (Ctor === undefined) return undefined
  return new Ctor(1, 1, STT_SAMPLE_RATE)
}

/**
 * Turn one recording (any container the browser decodes) into 16 kHz mono WAV bytes.
 *
 * @param recording - the MediaRecorder output bytes.
 * @param decoder - the 16 kHz decoder; defaults to the browser's.
 * @returns the WAV file bytes.
 * @throws when no decoder exists or the browser cannot decode the bytes.
 */
export async function recordingToWav16kMono(
  recording: ArrayBuffer,
  decoder: OfflineDecoderLike | undefined = createDecoder(),
): Promise<Uint8Array> {
  if (decoder === undefined) throw new Error('OfflineAudioContext is unavailable in this browser')
  const decoded = await decoder.decodeAudioData(recording)
  const channels: Float32Array[] = []
  for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
    channels.push(decoded.getChannelData(channel))
  }
  return encodeWav16kMono(mixToMono(channels))
}
