// @vitest-environment jsdom
/**
 * The record button in segmented (voice-activity) capture: the microphone stays open
 * until the heal call's trigger, the 120-second silence timeout, or a stop. The browser
 * recorder, the analyser, the transcriber and the heal endpoint are stubbed, and the
 * clock is virtual (setInterval and Date only), so 120 seconds of silence runs at once.
 *
 * @module dsh-talk/test/talk-mic-segments.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { TalkMicButton } from '../src/client/TalkMicButton.tsx'
import type { TalkStatus, TalkTranscript } from '../src/wire.ts'

const TICK_MS = 500
const SPEECH_LEVEL = 0.2

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
  record: { enabled: true, hotkey: null, maxSeconds: 60, autoSubmit: false, vad: { enabled: true, silenceMs: 1000, energyThreshold: 0.01 } },
  interrupt: false,
  patchFile: null,
}

/** Recorder instances, one per speech segment. */
const recorders: FakeRecorder[] = []
/** Base64 payloads sent to `transcribe`, in call order. */
const transcribed: string[] = []
/** Ordered event trace: segment raw logs and heal calls. */
const events: string[] = []
/** Transcripts the stubbed transcriber returns, one per call. */
let transcriptQueue: string[] = []
/** The microphone level the stubbed analyser reports. */
let micLevel = 0
/** What the stubbed heal endpoint does: answer like the model, or fail. */
let healMode: 'model' | 'error' = 'model'
/** Text the button wrote into the composer draft. */
const drafts: string[] = []
/** Texts the button submitted. */
const sent: string[] = []

const FINAL_PHRASE = /\bthat(?: is |'s )my prompt\b[.!?]?\s*$/iu

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

class FakeAudioContext {
  createMediaStreamSource(): { connect: () => void } {
    return { connect: () => {} }
  }
  createAnalyser(): { fftSize: number; getFloatTimeDomainData: (buffer: Float32Array) => void } {
    return {
      fftSize: 2048,
      getFloatTimeDomainData: (buffer: Float32Array) => {
        buffer.fill(micLevel)
      },
    }
  }
  close(): Promise<void> {
    return Promise.resolve()
  }
}

class FakeOfflineAudioContext {
  decodeAudioData(_data: ArrayBuffer): Promise<{ numberOfChannels: number; getChannelData: (channel: number) => Float32Array }> {
    return Promise.resolve({ numberOfChannels: 1, getChannelData: () => new Float32Array(1600) })
  }
}

/** The stubbed heal endpoint: answers like the model (phrase must end the transcript) or errors. */
async function fakeFetch(_url: string, init?: { body?: string }): Promise<unknown> {
  const body = JSON.parse(init?.body ?? '{}') as { messages: Array<{ content: string }> }
  const raw = (body.messages[1]?.content ?? '').replace(/^raw transcript: /u, '')
  events.push(`heal:${raw}`)
  if (healMode === 'error') return { ok: false, status: 500, text: async () => '' }
  const trigger = FINAL_PHRASE.test(raw)
  const prompt = trigger ? raw.replace(FINAL_PHRASE, '').trim() : raw
  const content = JSON.stringify({ trigger, prompt })
  return { ok: true, status: 200, text: async () => JSON.stringify({ choices: [{ message: { content } }] }) }
}

let container: HTMLElement
let root: Root | undefined

function mount(): void {
  const props = {
    status: async () => SETTINGS,
    interrupt: async () => ({ stopped: false }),
    transcribe: async (audioData: string): Promise<TalkTranscript> => {
      transcribed.push(audioData)
      return { text: transcriptQueue.shift() ?? '', engine: 'funasr', language: 'en-US' }
    },
    audio: async () => null,
    latest: async () => null,
    setDraft: (text: string) => {
      drafts.push(text)
    },
    send: async (text: string) => {
      sent.push(text)
    },
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

function recording(): boolean {
  return button().dataset['recording'] === 'true'
}

/** Let pending promises settle (real timers: setTimeout is not faked). */
async function flush(): Promise<void> {
  await act(async () => {
    for (let index = 0; index < 8; index += 1) await new Promise<void>(resolve => setTimeout(resolve, 0))
  })
}

/** Advance the virtual clock by whole detector ticks, flushing promises after each. */
async function advance(ms: number): Promise<void> {
  for (let elapsed = 0; elapsed < ms; elapsed += TICK_MS) {
    await act(async () => {
      vi.advanceTimersByTime(TICK_MS)
    })
    await flush()
  }
}

/** Speak for `ms` milliseconds, then go silent. */
async function speak(ms: number): Promise<void> {
  micLevel = SPEECH_LEVEL
  await advance(ms)
  micLevel = 0
}

/** Press the button to start (the capture then waits for speech). */
async function startCapture(): Promise<void> {
  await act(async () => {
    button().click()
  })
  await flush()
}

beforeEach(() => {
  ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  recorders.length = 0
  transcribed.length = 0
  events.length = 0
  drafts.length = 0
  sent.length = 0
  transcriptQueue = []
  micLevel = 0
  healMode = 'model'
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    if (args[0] === '[dsh-talk] segment raw transcript:') events.push(`segment:${String(args[1])}`)
  })
  vi.spyOn(console, 'warn').mockImplementation(() => {})
  vi.stubGlobal('MediaRecorder', FakeRecorder)
  vi.stubGlobal('Blob', FakeBlob)
  vi.stubGlobal('AudioContext', FakeAudioContext)
  vi.stubGlobal('OfflineAudioContext', FakeOfflineAudioContext)
  vi.stubGlobal('fetch', fakeFetch)
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => ({ getTracks: () => [{ stop: () => {} }] }) },
  })
  container = document.createElement('div')
  document.body.append(container)
})

async function mountAndSettle(): Promise<void> {
  mount()
  await flush()
}

afterEach(() => {
  unmount()
  container.remove()
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  Reflect.deleteProperty(navigator, 'mediaDevices')
})

describe('TalkMicButton segmented capture (voice-activity detection on)', () => {
  it('does not end the capture before anyone speaks', async () => {
    await mountAndSettle()
    await startCapture()
    await advance(3000)
    expect(recording()).toBe(true)
    expect(transcribed).toEqual([])
  })

  it('keeps one capture open across pauses and transcribes each segment alone', async () => {
    transcriptQueue = ['first sentence', 'second sentence', 'third sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await advance(10000)
    await speak(1000)
    await advance(1000)
    await advance(10000)
    await speak(1000)
    await advance(1000)
    expect(recorders).toHaveLength(3)
    expect(transcribed).toHaveLength(3)
    expect(recording()).toBe(true)
    expect(sent).toEqual([])
    expect(drafts).toEqual([])
  })

  it('sends the accumulated prompt at once on the trigger, without the phrase', async () => {
    transcriptQueue = ['first sentence', 'second sentence', 'third sentence that is my prompt']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence second sentence third sentence'])
    expect(sent.join(' ')).not.toMatch(/my prompt/iu)
    expect(recording()).toBe(false)
    expect(drafts).toEqual([])
  })

  it('does not send when the phrase is mid-sentence', async () => {
    transcriptQueue = ['that is my prompt was a phrase I once said']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual([])
    expect(recording()).toBe(true)
  })

  it('stops after 120 seconds of silence and leaves the transcript in the composer unsent', async () => {
    transcriptQueue = ['first sentence', 'second sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await advance(119_000)
    expect(recording()).toBe(true)
    await advance(2000)
    expect(recording()).toBe(false)
    expect(drafts).toEqual(['first sentence second sentence'])
    expect(sent).toEqual([])
  })

  it('keeps the capture open across pauses shorter than 120 seconds, however long the capture runs', async () => {
    transcriptQueue = ['first sentence', 'second sentence', 'third sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(100_000)
    await speak(1000)
    await advance(100_000)
    await speak(1000)
    await advance(1000)
    expect(recording()).toBe(true)
    expect(transcribed).toHaveLength(3)
    expect(drafts).toEqual([])
  })

  it('keeps an unbroken utterance open past 120 seconds, because speech resets the timeout', async () => {
    await mountAndSettle()
    await startCapture()
    await speak(130_000)
    expect(recording()).toBe(true)
    expect(transcribed).toEqual([])
  })

  it('does not trip the 120-second timeout while no segment has been captured', async () => {
    await mountAndSettle()
    await startCapture()
    await advance(130_000)
    expect(recording()).toBe(true)
    expect(transcribed).toEqual([])
    expect(drafts).toEqual([])
  })

  it('sends the text before the phrase when the heal endpoint fails and the transcript ends with the phrase', async () => {
    healMode = 'error'
    transcriptQueue = ['first sentence that is my prompt']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence'])
    expect(recording()).toBe(false)
  })

  it('keeps the capture open and shows the error when the heal endpoint fails and there is no phrase', async () => {
    healMode = 'error'
    transcriptQueue = ['first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(recording()).toBe(true)
    expect(sent).toEqual([])
    expect(errorText()).toContain('does not end with the trigger phrase')
  })

  it('logs each segment raw transcript before the heal call that judges it', async () => {
    transcriptQueue = ['first sentence', 'second sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    expect(events).toEqual([
      'segment:first sentence',
      'heal:first sentence',
      'segment:second sentence',
      'heal:first sentence second sentence',
    ])
  })
})
