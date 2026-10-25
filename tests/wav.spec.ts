/**
 * The 16 kHz mono WAV encoder the microphone sends to the transcription
 * engine: header fields, sample round trip and clamping, and channel mixing.
 *
 * @module dsh-talk/test/wav.spec
 */

import { describe, expect, it } from 'vitest'
import { encodeWav16kMono, mixToMono, STT_SAMPLE_RATE } from '../src/client/wav.ts'

function header(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
}

function ascii(bytes: Uint8Array, offset: number, length: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + length))
}

describe('encodeWav16kMono', () => {
  it('writes a 16 kHz mono 16-bit PCM WAV header', () => {
    const samples = new Float32Array(16000)
    const bytes = encodeWav16kMono(samples)
    const view = header(bytes)
    expect(bytes.byteLength).toBe(44 + 32000)
    expect(ascii(bytes, 0, 4)).toBe('RIFF')
    expect(view.getUint32(4, true)).toBe(36 + 32000)
    expect(ascii(bytes, 8, 4)).toBe('WAVE')
    expect(ascii(bytes, 12, 4)).toBe('fmt ')
    expect(view.getUint32(16, true)).toBe(16)
    expect(view.getUint16(20, true)).toBe(1)
    expect(view.getUint16(22, true)).toBe(1)
    expect(view.getUint32(24, true)).toBe(16000)
    expect(view.getUint32(28, true)).toBe(32000)
    expect(view.getUint16(32, true)).toBe(2)
    expect(view.getUint16(34, true)).toBe(16)
    expect(ascii(bytes, 36, 4)).toBe('data')
    expect(view.getUint32(40, true)).toBe(32000)
    expect(STT_SAMPLE_RATE).toBe(16000)
  })

  it('round-trips sample values and clamps out-of-range input', () => {
    const bytes = encodeWav16kMono(new Float32Array([0, 0.5, -0.5, 1, -1, 2, -2]))
    const view = header(bytes)
    const read = Array.from({ length: 7 }, (_, index) => view.getInt16(44 + index * 2, true))
    expect(read).toEqual([0, 16384, -16384, 32767, -32768, 32767, -32768])
  })

  it('encodes a one-second 440 Hz synthetic buffer as a 32044-byte file', () => {
    const samples = new Float32Array(16000)
    for (let index = 0; index < samples.length; index += 1) {
      samples[index] = Math.sin((2 * Math.PI * 440 * index) / 16000) * 0.5
    }
    const bytes = encodeWav16kMono(samples)
    expect(bytes.byteLength).toBe(32044)
    const view = header(bytes)
    const peak = Math.max(...Array.from({ length: 16000 }, (_, index) => Math.abs(view.getInt16(44 + index * 2, true))))
    expect(peak).toBeGreaterThan(16000)
    expect(peak).toBeLessThanOrEqual(16384)
  })
})

describe('mixToMono', () => {
  it('averages channels into one', () => {
    expect(Array.from(mixToMono([new Float32Array([1, 1]), new Float32Array([0, 0.5])]))).toEqual([0.5, 0.75])
  })

  it('returns a mono input unchanged and an empty list as empty', () => {
    const mono = new Float32Array([0.25, -0.25])
    expect(mixToMono([mono])).toBe(mono)
    expect(mixToMono([]).length).toBe(0)
  })
})
