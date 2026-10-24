// @vitest-environment jsdom
/**
 * The record button in segmented (voice-activity) capture: the microphone stays open
 * until a stop or the hard ceiling. A heal trigger sends its prompt and the mic stays open. The browser
 * recorder, the analyser, the transcriber and the heal endpoint are stubbed, and the
 * clock is virtual (setInterval and Date only), so 120 seconds of silence runs at once.
 *
 * @module dsh-talk/test/talk-mic-segments.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { TalkMicButton } from '../src/client/TalkMicButton.tsx'
import { healTranscript } from '../src/heal.ts'
import type { TalkHeal, TalkStatus, TalkTranscript } from '../src/wire.ts'

const TICK_MS = 500
const SPEECH_LEVEL = 0.2
/**
 * The model endpoint as the host sees it. The host's `talk/heal` runs `healTranscript`
 * against it; the browser must never fetch it, and nothing here points at port 8090.
 */
const HOST_MODEL_URL = 'http://host-model.test/v1/chat/completions'

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
  record: { enabled: true, hotkey: null, maxSeconds: 60, ceilingSeconds: 600, autoSubmit: false, vad: { enabled: true, silenceMs: 1000, energyThreshold: 0.01 } },
  interrupt: false,
  patchFile: null,
}

/** Recorder instances, one per speech segment. */
const recorders: FakeRecorder[] = []
/** Base64 payloads sent to `transcribe`, in call order. */
const transcribed: string[] = []
/** Ordered event trace: segment raw logs and heal calls. */
const events: string[] = []
/** Every `talk/heal` call the browser made, in order: the accumulated transcript it sent. */
const healCalls: string[] = []
/** Every URL the browser fetched. A correct browser fetches only the host's model call, once per heal. */
const fetched: string[] = []
/** Transcripts the stubbed transcriber returns, one per call. */
let transcriptQueue: string[] = []
/** How many of the next transcriber calls throw. */
let transcribeFailures = 0
/** When true, a page-requested stop fires its onstop later, as the W3C stop steps do. */
let asyncStops = false
/** Microphone tracks stopped so far (every track the button releases). */
let trackStops = 0
/** When set, every host heal call waits on it before answering (the call is in flight). */
let healGate: Promise<void> | null = null
let openHealGate: () => void = () => {}
/** The microphone level the stubbed analyser reports. */
let micLevel = 0
/** What the host's model call does: answer like the model, or fail (the host fallback then decides). */
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
  /** The page asked the recorder to stop: state turns inactive now, and the onstop follows (queued when asyncStops). */
  stop(): void {
    if (this.state === 'inactive') return
    this.state = 'inactive'
    this.deliver(asyncStops)
  }
  /** The browser ended the recorder by itself. The W3C stop steps queue its onstop, so it is always async here. */
  selfStop(): void {
    if (this.state === 'inactive') return
    this.state = 'inactive'
    this.deliver(true)
  }
  private deliver(later: boolean): void {
    const fire = (): void => {
      this.ondataavailable?.({ data: new FakeBlob([new Uint8Array([1, 2, 3, 4])]) })
      this.onstop?.()
    }
    if (later) setTimeout(fire, 0)
    else fire()
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

/**
 * The stubbed global `fetch`. Only the host's model URL answers; any other URL, such as the
 * loopback `127.0.0.1:8090`, is recorded and fails, so a browser-side heal call shows up in
 * `fetched` and breaks the test that checks it.
 */
async function fakeFetch(url: string, init?: { body?: string }): Promise<unknown> {
  fetched.push(url)
  if (url !== HOST_MODEL_URL) return { ok: false, status: 599, text: async () => '' }
  const body = JSON.parse(init?.body ?? '{}') as { messages: Array<{ content: string }> }
  const raw = (body.messages[1]?.content ?? '').replace(/^raw transcript: /u, '')
  events.push(`heal:${raw}`)
  if (healMode === 'error') return { ok: false, status: 500, text: async () => '' }
  const trigger = FINAL_PHRASE.test(raw)
  const prompt = trigger ? raw.replace(FINAL_PHRASE, '').trim() : raw
  const content = JSON.stringify({ trigger, prompt })
  return { ok: true, status: 200, text: async () => JSON.stringify({ choices: [{ message: { content } }] }) }
}

/**
 * The host's `talk/heal`, as the wire delivers it: the real `healTranscript` against the
 * host model URL, with a failure reaching the browser as a failed call (its message is
 * the one `src/service.ts` produces through the wire).
 */
async function hostHeal(rawTranscript: string): Promise<TalkHeal> {
  healCalls.push(rawTranscript)
  if (healGate !== null) await healGate
  try {
    const decision = await healTranscript(rawTranscript, { endpoint: HOST_MODEL_URL, logRawTranscript: () => {} })
    return { trigger: decision.trigger, prompt: decision.prompt, source: decision.source }
  } catch (error) {
    throw new Error(`talk.heal failed: INTERNAL: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/** Hold every host heal call in flight until openHealGate runs. */
function holdHeals(): void {
  healGate = new Promise<void>((resolve) => {
    openHealGate = resolve
  })
}

/** On every path, the browser made only the host's heal calls, and never fetched the heal endpoint on port 8090. */
function expectHostOnlyFetches(): void {
  expect(fetched).toEqual(healCalls.map(() => HOST_MODEL_URL))
  expect(fetched.filter(url => url.includes(':8090'))).toEqual([])
}

let container: HTMLElement
let root: Root | undefined

/** The legacy single recording: voice-activity detection off, so no AnalyserNode and no segments. */
const LEGACY_SETTINGS: TalkStatus = { ...SETTINGS, record: { ...SETTINGS.record, vad: { ...SETTINGS.record.vad, enabled: false } } }

function mount(settings: TalkStatus = SETTINGS): void {
  const props = {
    status: async () => settings,
    heal: hostHeal,
    interrupt: async () => ({ stopped: false }),
    transcribe: async (audioData: string): Promise<TalkTranscript> => {
      transcribed.push(audioData)
      if (transcribeFailures > 0) {
        transcribeFailures -= 1
        throw new Error('engine down')
      }
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
  healCalls.length = 0
  fetched.length = 0
  drafts.length = 0
  sent.length = 0
  transcriptQueue = []
  micLevel = 0
  healMode = 'model'
  transcribeFailures = 0
  asyncStops = false
  trackStops = 0
  healGate = null
  openHealGate = () => {}
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
    value: { getUserMedia: async () => ({ getTracks: () => [{ stop: () => { trackStops += 1 } }] }) },
  })
  container = document.createElement('div')
  document.body.append(container)
})

async function mountAndSettle(settings: TalkStatus = SETTINGS): Promise<void> {
  mount(settings)
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

  it('sends the accumulated prompt at once on the trigger, without the phrase, and keeps the mic open', async () => {
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
    expect(recording()).toBe(true)
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

  it('stays open through 120 seconds of silence, with the transcript kept and nothing sent', async () => {
    transcriptQueue = ['first sentence', 'second sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await advance(130_000)
    expect(recording()).toBe(true)
    expect(drafts).toEqual([])
    expect(sent).toEqual([])
  })

  it('keeps the capture open across long pauses, however long the capture runs', async () => {
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

  it('keeps an unbroken utterance open past 120 seconds', async () => {
    await mountAndSettle()
    await startCapture()
    await speak(130_000)
    expect(recording()).toBe(true)
    expect(transcribed).toEqual([])
  })

  it('keeps the capture open through silence before any segment', async () => {
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
    expect(recording()).toBe(true)
    expectHostOnlyFetches()
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
    expectHostOnlyFetches()
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

  it('a segmented capture outlives record.maxSeconds and keeps running after the trigger', async () => {
    transcriptQueue = ['first sentence', 'second sentence that is my prompt']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await advance(62_000)
    expect(recording()).toBe(true)
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence second sentence'])
    expect(recording()).toBe(true)
    expect(drafts).toEqual([])
  })

  it('the legacy single recording (voice-activity detection off) still stops at record.maxSeconds', async () => {
    transcriptQueue = ['legacy recording']
    await mountAndSettle(LEGACY_SETTINGS)
    await startCapture()
    await advance(59_000)
    expect(recording()).toBe(true)
    await advance(2000)
    expect(recording()).toBe(false)
    expect(drafts).toEqual(['legacy recording'])
    expect(healCalls).toEqual([])
  })

  it('sends the accumulated transcript to the host heal route, and the browser never fetches the heal endpoint', async () => {
    transcriptQueue = ['first sentence', 'second sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    expect(healCalls).toEqual(['first sentence', 'first sentence second sentence'])
    expect(fetched).toEqual(healCalls.map(() => HOST_MODEL_URL))
    expect(fetched.filter(url => url.includes(':8090'))).toEqual([])
  })

  it('a trigger from the host heal route submits the cleaned prompt and keeps the capture open', async () => {
    transcriptQueue = ['first sentence that is my prompt']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence'])
    expect(recording()).toBe(true)
    expect(healCalls).toEqual(['first sentence that is my prompt'])
  })

  it('after a trigger the next prompt starts from empty, because the capture stays open', async () => {
    transcriptQueue = ['first sentence that is my prompt', 'second sentence that is my prompt']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence', 'second sentence'])
    expect(recording()).toBe(true)
    expect(drafts).toEqual([])
  })

  it('a HealFailure from the host heal route shows its error and keeps the capture open', async () => {
    healMode = 'error'
    transcriptQueue = ['first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(recording()).toBe(true)
    expect(sent).toEqual([])
    expect(errorText()).toBe(
      'heal failed: talk.heal failed: INTERNAL: heal call failed (endpoint responded 500) and the transcript does not end with the trigger phrase',
    )
    expectHostOnlyFetches()
  })

  it('unmounting mid-capture releases the microphone and the recorder, and a heal answer that arrives later submits nothing', async () => {
    // Blocker 1 (P-F6): the stale closure must not submit, and nothing may keep recording.
    transcriptQueue = ['first sentence', 'second sentence that is my prompt']
    holdHeals()
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    const open = recorders[recorders.length - 1]
    expect(open?.state).toBe('recording')
    unmount()
    expect(trackStops).toBeGreaterThan(0)
    expect(open?.state).toBe('inactive')
    const started = recorders.length
    openHealGate()
    await flush()
    micLevel = SPEECH_LEVEL
    await advance(5000)
    micLevel = 0
    expect(recorders).toHaveLength(started)
    expect(sent).toEqual([])
    expect(drafts).toEqual([])
  })

  it('a trigger answered after the ceiling is dropped, and the transcript stays in the composer unsent', async () => {
    // Blocker 2 (P-F2): the answer lands after the ceiling ends the capture; it must not submit.
    transcriptQueue = ['hello that is my prompt']
    holdHeals()
    await mountAndSettle({ ...SETTINGS, record: { ...SETTINGS.record, ceilingSeconds: 30 } })
    await startCapture()
    await speak(1000)
    await advance(1000)
    await advance(31_000)
    expect(recording()).toBe(false)
    expect(drafts).toEqual([])
    openHealGate()
    await flush()
    expect(sent).toEqual([])
    expect(drafts).toEqual(['hello that is my prompt'])
  })

  it('a segmented capture ends at record.ceilingSeconds with its transcript in the composer unsent', async () => {
    // Item 4: sustained sound ends the capture at the ceiling, the same way the silence trip does.
    transcriptQueue = ['long speech']
    await mountAndSettle({ ...SETTINGS, record: { ...SETTINGS.record, ceilingSeconds: 30 } })
    await startCapture()
    await speak(29_000)
    expect(recording()).toBe(true)
    await speak(3000)
    expect(recording()).toBe(false)
    expect(transcribed).toHaveLength(1)
    expect(healCalls).toEqual([])
    expect(sent).toEqual([])
    expect(drafts).toEqual(['long speech'])
  })

  it('the last segment reaches the draft when the browser fires its onstop after the user stops', async () => {
    // Item 5 regression: the W3C stop steps are asynchronous; the draft waits for the last onstop.
    asyncStops = true
    transcriptQueue = ['first part', 'last words']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(drafts).toEqual(['first part last words'])
    expect(sent).toEqual([])
  })

  it('a recorder that stops by itself ends its segment, and speech after it starts a new segment at once', async () => {
    // Item 5 (P-F1): the dead window must not drop speech, and no segment may be lost from the draft.
    transcriptQueue = ['first part', 'second part']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    recorders[0]?.selfStop()
    await speak(1000)
    await advance(1000)
    await flush()
    expect(recorders).toHaveLength(2)
    expect(transcribed).toHaveLength(2)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(drafts).toEqual(['first part second part'])
  })

  it('a transcription failure keeps the capture open, shows the error, and later words still reach the draft', async () => {
    // Item 6: the failed segment's words are lost and named; the capture stays open.
    transcribeFailures = 1
    transcriptQueue = ['second sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(recording()).toBe(true)
    expect(errorText()).toBe("transcription failed, so this segment's words are lost: engine down")
    expect(sent).toEqual([])
    await speak(1000)
    await advance(1000)
    expect(recording()).toBe(true)
    await act(async () => {
      button().click()
    })
    await flush()
    expect(drafts).toEqual(['second sentence'])
  })
})
