/**
 * WAV conversion for the composer microphone. The capture (capture-worklet.ts)
 * hands over mono samples; the segmenter resamples each segment to 16 kHz, and
 * encodeWav16kMono writes it as 16-bit PCM WAV. No browser container or codec is
 * involved, because the transcription route refuses browser containers (WebM/Opus
 * in Chrome and Firefox, MP4 in Safari). The module is pure: it touches no browser API.
 *
 * @module dsh-talk/client/wav
 */

/** Sample rate of every WAV this client sends. The transcription route takes 16 kHz. */
export const STT_SAMPLE_RATE = 16000

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
 * The caller must pass samples already at 16 kHz; the segmenter's toSttRate does
 * that for the microphone.
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
