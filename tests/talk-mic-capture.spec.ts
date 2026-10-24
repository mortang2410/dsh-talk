// @vitest-environment jsdom
/**
 * The record button's capture path, rendered under jsdom with the browser
 * recorder stubbed: a FunASR-resolved profile never opens Web Speech, the
 * recording reaches the host as a 16 kHz mono WAV, silence leaves no text and no
 * error, and a failing stage shows its message in the composer (the element
 * next to the button, not only the hover title).
 *
 * @module dsh-talk/test/talk-mic-capture.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { TalkMicButton } from '../src/client/TalkMicButton.tsx'
import type { TalkStatus, TalkTranscript } from '../src/wire.ts'

const SETTINGS: TalkStatus = {
  stt: { engine: 'funasr', resolved: 'funasr', language: 'en-US', silenceFinaliseMs: 1500, funasrUrl: 'http://127.0.0.1:8002', whisperModel: null },
  tts: {
    engine: 'edge-tts',
    resolved: 'edge-tts',
    voice: null,
    fallbackToBrowser: false,
    piperModel: null,
    edgeTtsVoice: 'en-US-AriaNeural',
    browser: { voiceName: null, rate: 1, pitch: 1 },
  },
  announce: { enabled: false, onTurnEnd: false, onApproval: false, onError: false },
  record: { enabled: true, hotkey: null, maxSeconds: 60, autoSubmit: false, vad: { enabled: false, silenceMs: 1000, energyThreshold: 0.01 } },
  interrupt: false,
  patchFile: null,
}

/** Recorder instances the button created, one per press. */
const recorders: FakeRecorder[] = []
/** The base64 payloads the button sent to `transcribe`. */
const transcribed: string[] = []
/** Text the button wrote into the composer draft. */
const drafts: string[] = []
/** How many times the Web Speech constructor was reached. */
let speechConstructed = 0
/** What the stubbed `transcribe` does. */
let transcribeImpl: () => Promise<TalkTranscript> = async () => ({ text: '', engine: 'funasr', language: 'en-US' })
/** Whether the stubbed decoder accepts the recording bytes. */
let decodeMode: 'ok' | 'fail' = 'ok'

class FakeBlob {
  readonly parts: Uint8Array[] = []
  constructor(parts: ReadonlyArray<Uint8Array | FakeBlob> = []) {
    for (const part of parts) this.parts.push(part instanceof FakeBlob ? concat(part.parts) : part)
  }
  arrayBuffer(): Promise<ArrayBuffer> {
    const joined = concat(this.parts)
    return Promise.resolve(joined.buffer.slice(joined.byteOffset, joined.byteOffset + joined.byteLength) as ArrayBuffer)
  }
}

function concat(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((sum, part) => sum + part.byteLength, 0)
  const out = new Uint8Array(total)
  let offset = 0
  for (const part of parts) {
    out.set(part, offset)
    offset += part.byteLength
  }
  return out
}

class FakeRecorder {
  static isTypeSupported(type: string): boolean {
    return type === 'audio/webm;codecs=opus'
  }
  state: 'inactive' | 'recording' = 'inactive'
  ondataavailable: ((event: { data: FakeBlob }) => void) | null = null
  onstop: (() => void) | null = null
  start(): void {
    this.state = 'recording'
    recorders.push(this)
  }
  stop(): void {
    this.state = 'inactive'
    this.ondataavailable?.({ data: new FakeBlob([new Uint8Array([1, 2, 3, 4])]) })
    this.onstop?.()
  }
}

class FakeSpeechRecognition {
  constructor() {
    speechConstructed += 1
  }
}

class FakeOfflineAudioContext {
  readonly channels: number
  readonly length: number
  readonly sampleRate: number
  constructor(channels: number, length: number, sampleRate: number) {
    this.channels = channels
    this.length = length
    this.sampleRate = sampleRate
    decoderArgs.push([channels, length, sampleRate])
  }
  decodeAudioData(_data: ArrayBuffer): Promise<{ numberOfChannels: number; getChannelData: (channel: number) => Float32Array }> {
    if (decodeMode === 'fail') return Promise.reject(new Error('EncodingError: Unable to decode audio data'))
    return Promise.resolve({ numberOfChannels: 1, getChannelData: () => new Float32Array(1600) })
  }
}

const decoderArgs: number[][] = []

let container: HTMLElement
let root: Root | undefined

function mount(): void {
  const props = {
    status: async () => SETTINGS,
    interrupt: async () => ({ stopped: false }),
    transcribe: (audioData: string) => {
      transcribed.push(audioData)
      return transcribeImpl()
    },
    audio: async () => null,
    latest: async () => null,
    setDraft: (text: string) => {
      drafts.push(text)
    },
    send: async () => {},
  }
  root = createRoot(container)
  act(() => {
    root?.render(createElement(TalkMicButton as never, props as never))
  })
}

function unmount(): void {
  if (root !== undefined) act(() => root?.unmount())
  root = undefined
}

function button(): HTMLButtonElement {
  const found = container.querySelector<HTMLButtonElement>('[data-dsh-talk-mic]')
  if (found === null) throw new Error('mic button not rendered')
  return found
}

function errorText(): string | null {
  return container.querySelector('[data-dsh-talk-mic-error]')?.textContent ?? null
}

/** Let pending promises (settings load, recording conversion, transcribe) settle. */
async function flush(): Promise<void> {
  await act(async () => {
    for (let index = 0; index < 6; index += 1) await new Promise<void>(resolve => setTimeout(resolve, 0))
  })
}

/** Press the button: first press records, second press stops and transcribes. */
async function press(): Promise<void> {
  await act(async () => {
    button().click()
    for (let index = 0; index < 6; index += 1) await new Promise<void>(resolve => setTimeout(resolve, 0))
  })
}

beforeEach(() => {
  ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
  recorders.length = 0
  transcribed.length = 0
  drafts.length = 0
  decoderArgs.length = 0
  speechConstructed = 0
  decodeMode = 'ok'
  transcribeImpl = async () => ({ text: '', engine: 'funasr', language: 'en-US' })
  vi.stubGlobal('MediaRecorder', FakeRecorder)
  vi.stubGlobal('Blob', FakeBlob)
  vi.stubGlobal('SpeechRecognition', FakeSpeechRecognition)
  vi.stubGlobal('OfflineAudioContext', FakeOfflineAudioContext)
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => ({ getTracks: () => [{ stop: () => {} }] }) },
  })
  container = document.createElement('div')
  document.body.append(container)
})

/** Mount and let the settings snapshot load, inside one act scope. */
async function mountAndSettle(): Promise<void> {
  mount()
  await flush()
}

afterEach(() => {
  unmount()
  container.remove()
  vi.unstubAllGlobals()
  Reflect.deleteProperty(navigator, 'mediaDevices')
})

describe('TalkMicButton capture (FunASR-resolved profile, stubbed browser)', () => {
  it('sends a 16 kHz mono WAV to the host and shows no text and no error for silence', async () => {
    await mountAndSettle()
    await press()
    await press()
    expect(recorders).toHaveLength(1)
    expect(speechConstructed).toBe(0)
    expect(decoderArgs).toEqual([[1, 1, 16000]])
    expect(transcribed).toHaveLength(1)
    const wav = Uint8Array.from(atob(transcribed[0] ?? ''), character => character.charCodeAt(0))
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength)
    expect(String.fromCharCode(...wav.subarray(0, 4))).toBe('RIFF')
    expect(String.fromCharCode(...wav.subarray(8, 12))).toBe('WAVE')
    expect(view.getUint16(22, true)).toBe(1)
    expect(view.getUint32(24, true)).toBe(16000)
    expect(view.getUint16(34, true)).toBe(16)
    expect(errorText()).toBeNull()
    expect(drafts).toEqual([])
  })

  it('shows a host failure in the composer and clears it on the next press', async () => {
    transcribeImpl = async () => {
      throw new Error('talk.transcribe failed: INTERNAL: funasr responded 500: boom')
    }
    await mountAndSettle()
    await press()
    await press()
    expect(errorText()).toBe('transcription failed: talk.transcribe failed: INTERNAL: funasr responded 500: boom')
    expect(button().title).toBe(errorText())
    expect(drafts).toEqual([])
    await press()
    expect(errorText()).toBeNull()
    await press()
  })

  it('shows a conversion failure in the composer and does not call the host', async () => {
    decodeMode = 'fail'
    await mountAndSettle()
    await press()
    await press()
    expect(errorText()).toBe('could not convert the recording to 16 kHz WAV: EncodingError: Unable to decode audio data')
    expect(transcribed).toEqual([])
  })
})
