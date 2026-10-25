// @vitest-environment jsdom
/**
 * The record button in segmented (voice-activity) capture: the microphone stays open
 * until a stop or the 120-second silence timeout. A heal trigger sends its prompt and the mic stays open.
 * The capture worklet is replaced by a fake port that the test feeds with audio blocks, so every
 * block the segmenter judges is one the test chose. The transcriber and the heal endpoint are
 * stubbed, and the clock is virtual (setInterval and Date only), so 120 seconds of silence runs at once.
 *
 * @module dsh-talk/test/talk-mic-segments.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { CAPTURE_FLUSH_MESSAGE } from '../src/client/capture-worklet.ts'
import { TalkMicButton } from '../src/client/TalkMicButton.tsx'
import { healWindow } from '../src/heal.ts'
import type { TalkHeal, TalkStatus, TalkTranscript, TalkTranscriptLogEntry, TalkVoiceTurn } from '../src/wire.ts'

const TICK_MS = 500
/** The capture rate. At 16 kHz a 1000-sample block is 62.5 ms, and no resampling happens. */
const SAMPLE_RATE = 16000
const BLOCK_SAMPLES = 1000
const SPEECH_LEVEL = 0.2
/** Under the energy floor (record.vad.energyThreshold is 0.01), so not speech. */
const SOFT_LEVEL = 0.005
/**
 * The model endpoint as the host sees it. The host's `talk/heal` runs `healWindow`
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
  record: { enabled: true, hotkey: null, autoSubmit: false, vad: { silenceMs: 1000, energyThreshold: 0.01 } },
  interrupt: false,
  patchFile: null,
}

/** The same settings with autoSubmit on, for the cases where a stop or a trigger must submit at once. */
const AUTO_SUBMIT: TalkStatus = { ...SETTINGS, record: { ...SETTINGS.record, autoSubmit: true } }

/**
 * Ten distinct words for segment `n` (for example "s2w1 ... s2w10"), so a window check can tell
 * which segment each word came from.
 */
function segmentWords(n: number): string {
  return Array.from({ length: 10 }, (_, index) => `s${String(n)}w${String(index + 1)}`).join(' ')
}

/** Base64 payloads sent to `transcribe`, in call order. */
const transcribed: string[] = []
/** Ordered event trace: segment raw logs and heal calls. */
const events: string[] = []
/**
 * Every `talk/heal` call the browser made, in order: the window of earlier raw words (`context`)
 * and the newest segment (`text`). The window is what the model may read as CONTEXT.
 */
const healCalls: Array<{ context: string; text: string }> = []
/** Every URL the browser fetched. A correct browser fetches only the host's model call, once per heal. */
const fetched: string[] = []
/** Transcripts the stubbed transcriber returns, one per call. */
let transcriptQueue: string[] = []
/** How many of the next transcriber calls throw. */
let transcribeFailures = 0
/** Microphone tracks stopped so far (every track the button releases). */
let trackStops = 0
/** When set, every host heal call waits on it before answering (the call is in flight). */
let healGate: Promise<void> | null = null
let openHealGate: () => void = () => {}
/** The microphone level the test feeds into the capture worklet. */
let micLevel = 0
/**
 * What the host's model call does: answer like the model, fail with a 500, or answer with
 * finish_reason `length` (a cut answer, which the host must treat as a failure).
 */
let healMode: 'model' | 'error' | 'truncated' = 'model'
/** Answers the model gives, first in first out, before the stand-in answers. */
let modelAnswers: string[] = []
/** Text the button wrote into the composer draft. */
const drafts: string[] = []
/** Texts the button submitted. */
const sent: string[] = []
/** Every transcript-trail entry the button sent to the host, in order. */
const transcriptLog: TalkTranscriptLogEntry[] = []
/** When true, the host's trail write fails, and the stop must not care. */
let logFails = false
/** Every audio context the button opened, in order. */
const contexts: FakeAudioContext[] = []
/** Every capture worklet node the button created, in order. The test feeds blocks to their ports. */
const workletNodes: FakeWorkletNode[] = []

class FakeAudioContext {
  readonly sampleRate = SAMPLE_RATE
  readonly destination = {}
  readonly audioWorklet = { addModule: async (_url: string): Promise<void> => {} }
  closed = false
  constructor() {
    contexts.push(this)
  }
  createMediaStreamSource(): { connect: () => void; disconnect: () => void } {
    return { connect: () => {}, disconnect: () => {} }
  }
  createGain(): { gain: { value: number }; connect: () => void; disconnect: () => void } {
    return { gain: { value: 1 }, connect: () => {}, disconnect: () => {} }
  }
  state = 'running'
  resume(): Promise<void> {
    return Promise.resolve()
  }
  close(): Promise<void> {
    this.closed = true
    return Promise.resolve()
  }
}

/**
 * The worklet node: its port is the only part the button uses, and the test calls it to deliver blocks.
 * Like the real worklet, a flush message is answered after a macrotask with the partial block the node
 * holds: `held` is the audio the test kept back from the full blocks (empty unless a test sets it).
 */
class FakeWorkletNode {
  /** Audio held back from the full blocks, returned as the partial block on a flush. */
  held: Float32Array = new Float32Array(0)
  readonly port: { onmessage: ((event: { data: unknown }) => void) | null; postMessage: (data: unknown) => void } = {
    onmessage: null,
    postMessage: (data: unknown) => {
      if (data !== CAPTURE_FLUSH_MESSAGE) return
      const samples = this.held
      this.held = new Float32Array(0)
      setTimeout(() => this.port.onmessage?.({ data: { kind: 'flushed', samples } }), 0)
    },
  }
  constructor(_context: unknown, _name: string) {
    workletNodes.push(this)
  }
  connect(): void {}
  disconnect(): void {}
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
  // Only the TEXT TO CLEAN block is cleaned; the CONTEXT block is never the text the model is asked about.
  const toClean = /TEXT TO CLEAN \(return the cleaned version of exactly this\): ([^\n]*)$/u.exec(body.messages[1]?.content ?? '')
  const text = toClean?.[1] ?? ''
  events.push(`heal:${text}`)
  if (healMode === 'error') return { ok: false, status: 500, text: async () => '' }
  // The stand-in for the model: the next queued answer, else the text with "um" dropped. It never
  // decides the trigger: the trigger phrase stays in the prompt as ordinary words, as the model sees it.
  const answer = modelAnswers.shift() ?? text.replace(/\bum\s+/giu, '').trim()
  const content = JSON.stringify({ prompt: answer })
  const finishReason = healMode === 'truncated' ? 'length' : 'stop'
  return {
    ok: true,
    status: 200,
    text: async () => JSON.stringify({ choices: [{ message: { content }, finish_reason: finishReason }] }),
  }
}

/**
 * The host's `talk/heal`, as the wire delivers it: the real `healWindow` against the host model URL.
 * `healWindow` never throws, so a failed call reaches the browser as a raw answer, not as an error:
 * the browser keeps the raw newest text for that segment and shows nothing.
 */
async function hostHeal(context: string, text: string): Promise<TalkHeal> {
  healCalls.push({ context, text })
  if (healGate !== null) await healGate
  const result = await healWindow(context, text, { endpoint: HOST_MODEL_URL })
  return { prompt: result.prompt, source: result.source }
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

/**
 * Decode a base64 WAV the button sent: its sample rate, its byte length, and one 16-bit PCM
 * sample by index (the data starts after the 44-byte header).
 */
function wavOf(base64: string): { bytes: number; rate: number; sample: (index: number) => number } {
  const binary = atob(base64)
  const view = new DataView(new ArrayBuffer(binary.length))
  for (let index = 0; index < binary.length; index += 1) view.setUint8(index, binary.charCodeAt(index))
  return {
    bytes: binary.length,
    rate: view.getUint32(24, true),
    sample: (index: number) => view.getInt16(44 + index * 2, true),
  }
}

/**
 * Feed `ms` of audio at `level` to every capture worklet, in blocks of BLOCK_SAMPLES. A graph
 * torn down by the button has no port handler, so blocks sent after that go nowhere.
 */
function feed(ms: number, level: number): void {
  const blocks = Math.round((ms / 1000) * SAMPLE_RATE / BLOCK_SAMPLES)
  for (let index = 0; index < blocks; index += 1) {
    const block = new Float32Array(BLOCK_SAMPLES).fill(level)
    for (const node of workletNodes) node.port.onmessage?.({ data: block })
  }
}

let container: HTMLElement
let root: Root | undefined

function mount(settings: TalkStatus = SETTINGS, extra: Record<string, unknown> = {}): void {
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
    logTranscript: async (entry: TalkTranscriptLogEntry) => {
      if (logFails) throw new Error('trail down')
      transcriptLog.push(entry)
      return { path: '/tmp/dsh-talk-transcripts-test.jsonl' }
    },
    setDraft: (text: string) => {
      drafts.push(text)
    },
    send: async (text: string) => {
      sent.push(text)
    },
    ...extra,
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

function noticeText(): string | null {
  return container.querySelector('[data-dsh-talk-mic-notice]')?.textContent ?? null
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

/** Advance the virtual clock by whole ticks, feeding the microphone level in between, and flush after each. */
async function advance(ms: number): Promise<void> {
  for (let elapsed = 0; elapsed < ms; elapsed += TICK_MS) {
    await act(async () => {
      feed(TICK_MS, micLevel)
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

/** Wait real time: the transcription retry wait uses setTimeout, which this file does not fake. */
async function waitRealMs(ms: number): Promise<void> {
  await act(async () => {
    await new Promise<void>(resolve => setTimeout(resolve, ms))
  })
  await flush()
}

beforeEach(() => {
  ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  contexts.length = 0
  workletNodes.length = 0
  transcribed.length = 0
  events.length = 0
  healCalls.length = 0
  fetched.length = 0
  drafts.length = 0
  sent.length = 0
  transcriptLog.length = 0
  logFails = false
  transcriptQueue = []
  micLevel = 0
  healMode = 'model'
  modelAnswers = []
  transcribeFailures = 0
  trackStops = 0
  healGate = null
  openHealGate = () => {}
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    if (args[0] === '[dsh-talk] segment raw transcript:') events.push(`segment:${String(args[1])}`)
  })
  vi.spyOn(console, 'warn').mockImplementation(() => {})
  vi.stubGlobal('AudioContext', FakeAudioContext)
  vi.stubGlobal('AudioWorkletNode', FakeWorkletNode)
  vi.stubGlobal('fetch', fakeFetch)
  // jsdom has no object URLs; the worklet source is loaded through one.
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, writable: true, value: () => 'blob:dsh-talk-test' })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, writable: true, value: () => {} })
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
  Reflect.deleteProperty(URL, 'createObjectURL')
  Reflect.deleteProperty(URL, 'revokeObjectURL')
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
    expect(transcribed).toHaveLength(3)
    expect(recording()).toBe(true)
    expect(sent).toEqual([])
    expect(drafts).toEqual([])
  })

  it('sends the accumulated prompt at once on the trigger, without the phrase, and keeps the mic open', async () => {
    // Breaks if: the trigger waits for the stop, the phrase reaches the sent prompt, or the mic closes on a trigger.
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

  it('a phrase that opens a segment sends nothing, and the words after it are dropped', async () => {
    // Breaks if: an empty prompt is submitted, or the words after the phrase reach a later prompt or the draft.
    transcriptQueue = ['that is my prompt was a phrase I once said']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual([])
    expect(recording()).toBe(true)
    await act(async () => {
      button().click()
    })
    await flush()
    expect(drafts).toEqual([])
  })

  it('a phrase that opens a segment writes its raw words to the trail with action none and an empty cleaned text', async () => {
    // Breaks if: a phrase-only trigger leaves no trail line, or its line drops the phrase or the words after it, or names an action other than none.
    transcriptQueue = ['that is my prompt was a phrase I once said']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual([])
    expect(transcriptLog).toEqual([
      { raw: 'that is my prompt was a phrase I once said', cleaned: '', source: 'model', action: 'none' },
    ])
  })

  it('stops after 120 seconds of silence and leaves the transcript in the composer unsent', async () => {
    transcriptQueue = ['first sentence', 'second sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await advance(118_000)
    expect(recording()).toBe(true)
    await advance(3000)
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

  it('a stop before any speech writes nothing to the composer', async () => {
    await mountAndSettle()
    await startCapture()
    await act(async () => {
      button().click()
    })
    await flush()
    expect(recording()).toBe(false)
    expect(transcribed).toEqual([])
    expect(drafts).toEqual([])
  })

  it('a failed heal falls back to the raw text, so a trigger in that text still submits the words before it', async () => {
    // Breaks if: a failed call drops the trigger check, or the words before the phrase are lost when the model is down.
    healMode = 'error'
    transcriptQueue = ['first sentence that is my prompt']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence'])
    expect(recording()).toBe(true)
    expect(transcriptLog).toEqual([
      { raw: 'first sentence that is my prompt', cleaned: 'first sentence', source: 'raw', action: 'submit' },
    ])
    expectHostOnlyFetches()
  })

  it('a failed heal keeps the capture open with no error shown, and the stop writes the raw words', async () => {
    // Breaks if: a failed call shows an error or sends anything, or the stop loses the raw words of the failed segment.
    healMode = 'error'
    transcriptQueue = ['first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(recording()).toBe(true)
    expect(sent).toEqual([])
    expect(drafts).toEqual([])
    expect(errorText()).toBeNull()
    await act(async () => {
      button().click()
    })
    await flush()
    expect(drafts).toEqual(['first sentence'])
    expectHostOnlyFetches()
  })

  it('logs each segment raw transcript before the heal call that judges it', async () => {
    // Breaks if: a call's text is the whole transcript again (the window is not the text), or the raw log follows its call.
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
      'heal:second sentence',
    ])
  })

  it('a segmented capture is not cut off at a fixed length and keeps running after the trigger', async () => {
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

  it('sends each segment with its window of earlier words to the host heal route, and the browser never fetches the heal endpoint', async () => {
    // Breaks if: a call sends the accumulated transcript instead of its window and newest segment, or the browser fetches the heal endpoint itself.
    transcriptQueue = ['first sentence', 'second sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    expect(healCalls).toEqual([
      { context: '', text: 'first sentence' },
      { context: 'first sentence', text: 'second sentence' },
    ])
    expect(fetched).toEqual(healCalls.map(() => HOST_MODEL_URL))
    expect(fetched.filter(url => url.includes(':8090'))).toEqual([])
  })

  it('a trigger found in the cleaned text submits the prompt and keeps the capture open', async () => {
    // Breaks if: the trigger is read from the model's answer instead of the glued text, or the capture closes on a trigger.
    transcriptQueue = ['first sentence that is my prompt']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence'])
    expect(recording()).toBe(true)
    expect(healCalls).toEqual([{ context: '', text: 'first sentence that is my prompt' }])
  })

  it('after a trigger the next prompt starts from empty, with an empty window, because the capture stays open', async () => {
    // Breaks if: the window or the cleaned text survives a trigger, so the next prompt carries the last one's words.
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
    expect(healCalls).toEqual([
      { context: '', text: 'first sentence that is my prompt' },
      { context: '', text: 'second sentence that is my prompt' },
    ])
  })

  it('a failed heal keeps its raw words in the window and the prompt, and the next segment submits them with its trigger', async () => {
    // Breaks if: the raw fallback leaves the window or the glued prompt, or a failed call shows an error.
    healMode = 'error'
    transcriptQueue = ['first sentence', 'second sentence that is my prompt']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(errorText()).toBeNull()
    expect(sent).toEqual([])
    healMode = 'model'
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence second sentence'])
    expect(errorText()).toBeNull()
    expect(healCalls).toEqual([
      { context: '', text: 'first sentence' },
      { context: 'first sentence', text: 'second sentence that is my prompt' },
    ])
    expectHostOnlyFetches()
  })

  it('unmounting mid-capture releases the microphone and the capture graph, and a heal answer that arrives later submits nothing', async () => {
    // Blocker 1 (P-F6): the stale closure must not submit, and nothing may keep capturing.
    // Breaks if: a heal answer that lands after unmount submits or drafts, or the graph stays open.
    transcriptQueue = ['first sentence', 'second sentence that is my prompt']
    holdHeals()
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    const graph = contexts[contexts.length - 1]
    expect(graph?.closed).toBe(false)
    unmount()
    expect(trackStops).toBeGreaterThan(0)
    expect(graph?.closed).toBe(true)
    const heard = transcribed.length
    openHealGate()
    await flush()
    micLevel = SPEECH_LEVEL
    await advance(5000)
    micLevel = 0
    expect(transcribed).toHaveLength(heard)
    expect(sent).toEqual([])
    expect(drafts).toEqual([])
  })

  it('a trigger answered after the 120-second trip is dropped, and the text before the phrase reaches the composer unsent', async () => {
    // Blocker 2 (P-F2): the answer lands after the trip; it must not submit. The segment was already
    // judged once, so the stop does not ask again: the answer's cut is the stop's text, so the composer
    // gets the text before the phrase, and the phrase itself is not written.
    // Breaks if: the late answer submits, the stop calls the model a second time, or the phrase reaches the draft.
    transcriptQueue = ['hello that is my prompt']
    holdHeals()
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await advance(121_000)
    expect(recording()).toBe(false)
    expect(drafts).toEqual([])
    openHealGate()
    await flush()
    expect(sent).toEqual([])
    expect(drafts).toEqual(['hello'])
    expect(healCalls).toEqual([{ context: '', text: 'hello that is my prompt' }])
  })

  it('a trigger whose voice-turn arm answers after unmount submits nothing', async () => {
    // Review round 1 F4: the arm is awaited after the trigger decision, so an unmount during the
    // await must stop the submit that follows it.
    // Breaks if: the submit runs after an unmount that happens during the arm's await.
    transcriptQueue = ['hello that is my prompt']
    let releaseArm: () => void = () => {}
    const voiceTurn = vi.fn((_sessionId: string): Promise<TalkVoiceTurn> => new Promise(resolve => {
      releaseArm = () => resolve({ armed: true })
    }))
    mount(SETTINGS, { voiceTurn, sessionId: 'review-session' })
    await flush()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(voiceTurn).toHaveBeenCalledTimes(1)
    expect(sent).toEqual([])
    unmount()
    await act(async () => {
      releaseArm()
    })
    await flush()
    expect(sent).toEqual([])
  })

  it('the open segment is transcribed when the user stops mid-speech, so its words reach the draft', async () => {
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
    expect(transcribed).toHaveLength(2)
    expect(drafts).toEqual(['first part last words'])
    expect(sent).toEqual([])
  })

  it('the audio the worklet still holds when the user stops reaches the transcription', async () => {
    // Review round 4, R4-1: the worklet posts only full blocks, so the partial block it held at a stop was
    // dropped. The fake holds 100 ms back (1600 samples at 16 kHz); the stop must return it to the segment.
    transcriptQueue = ['held words']
    await mountAndSettle()
    await startCapture()
    micLevel = SPEECH_LEVEL
    await advance(1000)
    for (const node of workletNodes) node.held = new Float32Array(1600).fill(SPEECH_LEVEL)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(transcribed).toHaveLength(1)
    // 1000 ms of full blocks (16000 samples) and the 100 ms held back (1600): 17600 samples of 16-bit PCM after a 44-byte header.
    expect(wavOf(transcribed[0] ?? '').bytes).toBe(44 + 2 * 17600)
  })

  it('keeps the audio before a soft onset in the transcribed WAV, so a word start under the energy floor is not lost', async () => {
    transcriptQueue = ['soft onset']
    await mountAndSettle()
    await startCapture()
    micLevel = SOFT_LEVEL
    await advance(1000)
    micLevel = SPEECH_LEVEL
    await advance(500)
    micLevel = 0
    await advance(1000)
    expect(transcribed).toHaveLength(1)
    const wav = wavOf(transcribed[0] ?? '')
    // 600 ms of pre-roll (9600 samples), 500 ms of speech (8000), 1000 ms of trailing silence (16000).
    expect(wav.rate).toBe(16000)
    expect(wav.bytes).toBe(44 + 2 * 33600)
    expect(wav.sample(0)).toBe(164) // 0.005 x 32767 = 163.8, rounded
    expect(wav.sample(9599)).toBe(164) // the last pre-roll sample is still the soft onset
    expect(wav.sample(9600)).toBe(6553) // 0.2 x 32767 = 6553.4: the first loud sample
    expect(wav.sample(33599)).toBe(0) // the trailing silence
  })

  it('the open segment of a capture stopped by the user keeps its pre-roll too', async () => {
    transcriptQueue = ['late word']
    await mountAndSettle()
    await startCapture()
    micLevel = SOFT_LEVEL
    await advance(1000)
    micLevel = SPEECH_LEVEL
    await advance(500)
    micLevel = 0
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(transcribed).toHaveLength(1)
    const wav = wavOf(transcribed[0] ?? '')
    // 600 ms of pre-roll and 500 ms of speech; the stop closes the segment with no trailing silence.
    expect(wav.bytes).toBe(44 + 2 * (9600 + 8000))
    expect(wav.sample(0)).toBe(164)
    expect(wav.sample(9600)).toBe(6553)
    expect(drafts).toEqual(['late word'])
  })

  it('refuses to record and shows an error when the browser has no AudioContext', async () => {
    vi.stubGlobal('AudioContext', undefined)
    await mountAndSettle()
    await startCapture()
    expect(errorText()).toBe('voice-activity detection is unavailable in this browser')
    expect(recording()).toBe(false)
    expect(workletNodes).toEqual([])
    expect(trackStops).toBe(1)
  })

  it('refuses to record and shows an error when the capture worklet cannot load', async () => {
    vi.stubGlobal('AudioContext', class extends FakeAudioContext {
      override readonly audioWorklet = {
        addModule: async (): Promise<void> => {
          throw new Error('worklet blocked')
        },
      }
    })
    await mountAndSettle()
    await startCapture()
    expect(errorText()).toBe('voice-activity detection is unavailable in this browser')
    expect(recording()).toBe(false)
    expect(workletNodes).toEqual([])
    expect(contexts.every(context => context.closed)).toBe(true)
    expect(trackStops).toBe(1)
  })

  it('refuses to record and shows an error when the AudioContext stays suspended after resume', async () => {
    // B12: a suspended context captures nothing, so the button refuses to record instead of recording silence.
    // The worklet node is created before resume, so only the teardown and the refusal are checked here.
    vi.stubGlobal('AudioContext', class extends FakeAudioContext {
      override state = 'suspended'
    })
    await mountAndSettle()
    await startCapture()
    expect(errorText()).toBe('voice-activity detection is unavailable in this browser')
    expect(recording()).toBe(false)
    expect(contexts.every(context => context.closed)).toBe(true)
    expect(trackStops).toBe(1)
  })

  it('refuses to record and shows an error when the AudioContext cannot attach the microphone', async () => {
    vi.stubGlobal('AudioContext', class extends FakeAudioContext {
      override createMediaStreamSource(): never {
        throw new Error('no source node')
      }
    })
    await mountAndSettle()
    await startCapture()
    expect(errorText()).toBe('voice-activity detection is unavailable in this browser')
    expect(workletNodes).toEqual([])
    expect(trackStops).toBe(1)
  })

  it('a transcription failure keeps the capture open, shows the error, and later words still reach the draft', async () => {
    // Item 6: the failed segment's words are lost and named; the capture stays open. All three
    // attempts fail here; the two retries each wait 1 s in real time.
    transcribeFailures = 3
    transcriptQueue = ['second sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await waitRealMs(2500)
    expect(transcribed).toHaveLength(3)
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

  it('an empty segment shows a short note instead of nothing, and the next words clear the note', async () => {
    // B2: a segment with no words is not silent on the page. The note is not an error, and the capture stays open.
    transcriptQueue = ['', 'first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(noticeText()).toBe('no words recognised in that segment')
    expect(errorText()).toBeNull()
    expect(recording()).toBe(true)
    await speak(1000)
    await advance(1000)
    expect(noticeText()).toBeNull()
    expect(recording()).toBe(true)
  })

  it('a transcription that fails once and then succeeds shows no error, and its words reach the draft', async () => {
    // B3: the first attempt fails, the retry (1 s later, real time) succeeds, so the segment is not lost.
    transcribeFailures = 1
    transcriptQueue = ['first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await waitRealMs(1500)
    expect(transcribed).toHaveLength(2)
    expect(errorText()).toBeNull()
    await act(async () => {
      button().click()
    })
    await flush()
    expect(drafts).toEqual(['first sentence'])
    expect(sent).toEqual([])
  })

  it('a stop writes the cleaned text, not the raw transcript, and asks the model no second time', async () => {
    // B4: the heal answer for the segment is kept, and the stop writes its cleaned prompt to the composer.
    // Breaks if: the stop re-judges a segment already answered, or it writes the raw words over the cleaned ones.
    transcriptQueue = ['um first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    // The stop reuses the answer; it asks heal no second time.
    expect(healCalls).toEqual([{ context: '', text: 'um first sentence' }])
    expect(drafts).toEqual(['first sentence'])
    expect(sent).toEqual([])
  })

  it('a stop heals the open segment once, with the window of the segment before it, and writes its cleaned form', async () => {
    // B4: the open segment's words arrive after the stop began, so no heal call judged them. The stop asks heal once.
    // Breaks if: the stop's call has no window, or it is given the whole transcript instead of the newest segment.
    transcriptQueue = ['first part', 'um last words']
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
    expect(healCalls).toEqual([
      { context: '', text: 'first part' },
      { context: 'first part', text: 'um last words' },
    ])
    expect(drafts).toEqual(['first part last words'])
    expect(sent).toEqual([])
  })

  it('a stop keeps the raw text when the heal call fails and the text has no trigger phrase', async () => {
    // B4: the segment's heal call failed, so its raw words stand in and the stop writes them as they were transcribed.
    // Breaks if: the failed segment's raw words are dropped, or the stop writes an answer that was never given.
    healMode = 'error'
    transcriptQueue = ['um first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(drafts).toEqual(['um first sentence'])
    expect(sent).toEqual([])
  })

  it('a failed heal still fires on the misheard "that is my problem" mid-capture, and autoSubmit sends the words before it', async () => {
    // B5 during an outage: the matcher accepts the mishearing on the raw text, so the phrase itself is
    // not submitted as part of the prompt.
    // Breaks if: a failed call disables the trigger check, or the misheard phrase reaches the prompt.
    healMode = 'error'
    transcriptQueue = ['one thought that is my problem']
    await mountAndSettle({ ...SETTINGS, record: { ...SETTINGS.record, autoSubmit: true } })
    await startCapture()
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(sent).toEqual(['one thought'])
    expect(drafts).toEqual([])
  })

  it('a segment whose heal call failed is not healed again at the stop: its raw words are written as transcribed', async () => {
    // SPEC: a failed call falls back to the raw text for that segment only, and the stop heals only the
    // segments still waiting for cleanup. The raw fallback is logged as source raw.
    // Breaks if: the stop retries a segment already answered by a failed call, or the trail calls raw words model.
    healMode = 'error'
    transcriptQueue = ['um first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    healMode = 'model'
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(healCalls).toEqual([{ context: '', text: 'um first sentence' }])
    expect(drafts).toEqual(['um first sentence'])
    expect(transcriptLog).toEqual([{ raw: 'um first sentence', cleaned: 'um first sentence', source: 'raw', action: 'draft' }])
  })

  it('a stop after the trigger phrase writes the text before the phrase to the composer, and sends nothing', async () => {
    // B4: a trigger heard at the stop is not submitted (the capture is ending), and the phrase is not written.
    // Breaks if: the trigger at the stop submits, or the phrase or the words after it reach the draft.
    transcriptQueue = ['first sentence', 'second sentence that is my prompt']
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
    expect(healCalls).toEqual([
      { context: '', text: 'first sentence' },
      { context: 'first sentence', text: 'second sentence that is my prompt' },
    ])
    expect(drafts).toEqual(['first sentence second sentence'])
    expect(sent).toEqual([])
  })

  it('a trigger heal still in flight at the stop is reused, not asked again, and sends nothing', async () => {
    // Review round 5, R5-1: the trigger answer arrives after the stop began. Its text before the phrase is the
    // stop's cleaned text, so the stop makes no second heal call and submits nothing.
    // Breaks if: the stop asks the model for the same segment again, or the cut prompt is submitted instead of drafted.
    transcriptQueue = ['one thought that is my prompt']
    holdHeals()
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(healCalls).toEqual([{ context: '', text: 'one thought that is my prompt' }])
    await act(async () => {
      button().click()
    })
    await flush()
    openHealGate()
    await flush()
    await flush()
    expect(healCalls).toEqual([{ context: '', text: 'one thought that is my prompt' }])
    expect(drafts).toEqual(['one thought'])
    expect(sent).toEqual([])
  })

  it('a stop whose only words are the phrase, with its heal still in flight, writes action none with an empty cleaned text', async () => {
    // Breaks if: the stop drafts or submits the phrase, the trail drops the phrase from its raw text, or the action is not none.
    transcriptQueue = ['that is my prompt']
    holdHeals()
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(healCalls).toEqual([{ context: '', text: 'that is my prompt' }])
    await act(async () => {
      button().click()
    })
    await flush()
    openHealGate()
    await flush()
    await flush()
    expect(drafts).toEqual([])
    expect(sent).toEqual([])
    expect(transcriptLog).toEqual([
      { raw: 'that is my prompt', cleaned: '', source: 'model', action: 'none' },
    ])
  })

  it('a trigger heal still in flight at the stop, with autoSubmit on, submits the text before the phrase once', async () => {
    // Review round 6, R6-1: with autoSubmit on, the stop submits the cleaned text before the phrase. The trigger
    // prompt itself is not sent, and no second heal call is made.
    // Breaks if: the stop asks the model for the same segment again, or the cut prompt is submitted twice or not at all.
    transcriptQueue = ['one thought that is my prompt']
    holdHeals()
    await mountAndSettle(AUTO_SUBMIT)
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(healCalls).toEqual([{ context: '', text: 'one thought that is my prompt' }])
    await act(async () => {
      button().click()
    })
    await flush()
    openHealGate()
    await flush()
    await flush()
    expect(healCalls).toEqual([{ context: '', text: 'one thought that is my prompt' }])
    expect(sent).toEqual(['one thought'])
    expect(drafts).toEqual([])
  })

  it('a stop submits the words said so far when the autoSubmit setting is on', async () => {
    // B14: the host path honours record.autoSubmit at the stop, as the Web Speech path does.
    // Breaks if: a stop with autoSubmit on drafts instead of submitting, or a stop with autoSubmit off submits.
    transcriptQueue = ['first sentence']
    await mountAndSettle({ ...SETTINGS, record: { ...SETTINGS.record, autoSubmit: true } })
    await startCapture()
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(sent).toEqual(['first sentence'])
    expect(drafts).toEqual([])
  })

  it('a stop appends the raw text, the cleaned text, and the draft action to the transcript trail', async () => {
    // The trail: the raw words stay beside the cleaned text, so the model's cleanup can be checked.
    // Breaks if: the stop's trail line drops the raw words, writes the cleaned text as raw, or reports the wrong action.
    transcriptQueue = ['um first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(transcriptLog).toEqual([
      { raw: 'um first sentence', cleaned: 'first sentence', source: 'model', action: 'draft' },
    ])
  })

  it('the trail records a submitted stop as submit, and a trigger prompt as submit too', async () => {
    // Breaks if: the trigger's trail line loses its raw words, or a submitted line says draft.
    transcriptQueue = ['um first sentence']
    await mountAndSettle({ ...SETTINGS, record: { ...SETTINGS.record, autoSubmit: true } })
    await startCapture()
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(transcriptLog).toEqual([
      { raw: 'um first sentence', cleaned: 'first sentence', source: 'model', action: 'submit' },
    ])
    transcriptLog.length = 0
    transcriptQueue = ['write a haiku that is my prompt']
    await startCapture()
    await speak(1000)
    await advance(1000)
    await flush()
    expect(sent).toContain('write a haiku')
    expect(transcriptLog).toEqual([
      { raw: 'write a haiku that is my prompt', cleaned: 'write a haiku', source: 'model', action: 'submit' },
    ])
  })

  it('a stop with one model segment and one raw segment records raw for the whole stop', async () => {
    // Breaks if: a stop that used a fallback is logged as model, or the stop line drops a segment's words.
    transcriptQueue = ['first sentence', 'second sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    healMode = 'error'
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    await flush()
    expect(drafts).toEqual(['first sentence second sentence'])
    expect(transcriptLog).toEqual([
      { raw: 'first sentence second sentence', cleaned: 'first sentence second sentence', source: 'raw', action: 'draft' },
    ])
    expectHostOnlyFetches()
  })

  it('a trigger prompt with a raw segment records raw, and the prompt after it records model', async () => {
    // Breaks if: the fallback leaks past the trigger into the next prompt's line, or the trigger line takes the source of a segment after it.
    healMode = 'error'
    transcriptQueue = ['first sentence', 'second sentence that is my prompt', 'third sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    healMode = 'model'
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence second sentence'])
    expect(transcriptLog).toEqual([
      { raw: 'first sentence second sentence that is my prompt', cleaned: 'first sentence second sentence', source: 'raw', action: 'submit' },
    ])
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    await flush()
    expect(drafts).toEqual(['third sentence'])
    expect(transcriptLog).toEqual([
      { raw: 'first sentence second sentence that is my prompt', cleaned: 'first sentence second sentence', source: 'raw', action: 'submit' },
      { raw: 'third sentence', cleaned: 'third sentence', source: 'model', action: 'draft' },
    ])
    expectHostOnlyFetches()
  })

  it('a failed trail write leaves the stop untouched and writes the draft', async () => {
    // The trail is a record, not a dependency: losing it must not lose the words.
    // Breaks if: a failed trail write stops the draft, or the stop throws instead of drafting.
    logFails = true
    transcriptQueue = ['um first sentence']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(drafts).toEqual(['first sentence'])
    expect(transcriptLog).toEqual([])
  })

  it('the first segment of a capture has an empty context, and a context under 30 words is every earlier raw word', async () => {
    // Breaks if: the first call carries words it never heard, or a short window drops or repeats earlier words.
    transcriptQueue = ['alpha beta gamma', 'delta epsilon', 'zeta']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    expect(healCalls).toEqual([
      { context: '', text: 'alpha beta gamma' },
      { context: 'alpha beta gamma', text: 'delta epsilon' },
      { context: 'alpha beta gamma delta epsilon', text: 'zeta' },
    ])
  })

  it('a long capture sends at most the last 30 earlier raw words as context, and only the newest segment as text', async () => {
    // Breaks if: the window grows past 30 words, the window is taken after the segment is appended (so a segment is its own context), or a call's text is more than its newest segment.
    const pieces = [1, 2, 3, 4, 5].map(segmentWords)
    transcriptQueue = [...pieces]
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    expect(healCalls).toHaveLength(5)
    expect(healCalls[0]).toEqual({ context: '', text: pieces[0] })
    expect(healCalls[1]).toEqual({ context: pieces[0], text: pieces[1] })
    // Before segment 5 there are 40 earlier words; the window is the last 30, which is segments 2 to 4 whole.
    expect(healCalls[4]).toEqual({ context: [pieces[1], pieces[2], pieces[3]].join(' '), text: pieces[4] })
    expect(healCalls[4]?.context.split(' ')).toHaveLength(30)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    // The window limits what the model sees, never the trail: the stop's raw line keeps all 50 words.
    expect(drafts).toEqual([pieces.join(' ')])
    expect(transcriptLog).toEqual([{ raw: pieces.join(' '), cleaned: pieces.join(' '), source: 'model', action: 'draft' }])
  })

  // The window is the last 30 raw words said before the newest segment. These four sizes sit on both
  // sides of that edge: 0 has no window, 29 and 30 fit whole, and 31 must be cut to its last 30.
  const wordCount = (text: string | undefined): number => (text === undefined || text === '' ? 0 : text.split(' ').length)
  for (const earlier of [0, 29, 30, 31]) {
    it(`a segment after ${String(earlier)} earlier raw words gets a context of ${String(Math.min(earlier, 30))} words`, async () => {
      // Breaks if: the window is one word short at 29 or 30, a 31-word history is not cut to its last 30 words, or the newest segment is counted among its own context words.
      const earlierWords = Array.from({ length: earlier }, (_, index) => `earlier${String(index + 1)}`)
      const newest = 'newest words'
      const pieces = earlier === 0 ? 1 : 2
      transcriptQueue = earlier === 0 ? [newest] : [earlierWords.join(' '), newest]
      await mountAndSettle()
      await startCapture()
      // The transcriber shifts transcriptQueue as it goes, so the loop bound is fixed before the loop.
      for (let piece = 0; piece < pieces; piece += 1) {
        await speak(1000)
        await advance(1000)
      }
      const last = healCalls[healCalls.length - 1]
      expect(last).toEqual({ context: earlierWords.slice(-30).join(' '), text: newest })
      expect(wordCount(last?.context)).toBe(Math.min(earlier, 30))
    })
  }

  it('two pieces are glued with one space, and the draft holds the model answers, not a copy of the context', async () => {
    // Breaks if: the draft repeats the context, the pieces are joined without a space, or the raw words replace the model's answer for a piece.
    transcriptQueue = ['read the config', 'config for the ssh socket']
    modelAnswers = ['read the config', 'for the ssh socket']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    expect(healCalls[1]).toEqual({ context: 'read the config', text: 'config for the ssh socket' })
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(drafts).toEqual(['read the config for the ssh socket'])
  })

  const SPLITS: Array<[string, string]> = [
    ['first sentence that', 'is my prompt'],
    ['first sentence that is', 'my prompt'],
    ['first sentence that is my', 'prompt'],
  ]
  for (const [first, second] of SPLITS) {
    it(`a trigger phrase split between segments as "${first}" and "${second}" fires, and the prompt excludes the phrase`, async () => {
      // Breaks if: the matcher runs on each segment alone instead of the glued text, so a phrase split across a boundary never fires.
      transcriptQueue = [first, second]
      await mountAndSettle()
      await startCapture()
      await speak(1000)
      await advance(1000)
      expect(sent).toEqual([])
      await speak(1000)
      await advance(1000)
      expect(sent).toEqual(['first sentence'])
      expect(drafts).toEqual([])
      expect(recording()).toBe(true)
    })
  }

  it('a phrase in the middle of a segment ends the prompt there: the words after it are never submitted, and stay raw in the trail', async () => {
    // Breaks if: the words after the phrase reach a sent prompt or the draft, or they are missing from the trail's raw text.
    transcriptQueue = ['first sentence that is my prompt and then more words']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence'])
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(sent).toEqual(['first sentence'])
    expect(drafts).toEqual([])
    expect(transcriptLog).toEqual([
      { raw: 'first sentence that is my prompt and then more words', cleaned: 'first sentence', source: 'model', action: 'submit' },
    ])
  })

  it('an all-filler segment answered with an empty prompt adds nothing, and is never replaced with its raw words', async () => {
    // Breaks if: an empty answer falls back to the raw "um", the empty segment's words reach the draft, or the trail marks the stop raw.
    transcriptQueue = ['um', 'first sentence', 'um']
    modelAnswers = ['', 'first sentence', '']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(healCalls.map(call => call.text)).toEqual(['um', 'first sentence', 'um'])
    expect(drafts).toEqual(['first sentence'])
    expect(transcriptLog).toEqual([{ raw: 'um first sentence um', cleaned: 'first sentence', source: 'model', action: 'draft' }])
  })

  it('a capture whose every segment is filler writes nothing to the composer, and the trail logs action none', async () => {
    // Breaks if: an all-filler capture drafts raw words or an empty draft, or its trail line loses the raw words.
    transcriptQueue = ['um']
    modelAnswers = ['']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(drafts).toEqual([])
    expect(sent).toEqual([])
    expect(transcriptLog).toEqual([{ raw: 'um', cleaned: '', source: 'model', action: 'none' }])
  })

  it('after a segment whose cleaned text differs from its raw words, and a segment cleaned to nothing, the next window holds the raw words and not the current segment', async () => {
    // Breaks if: the window is built from cleaned text instead of raw text, an empty cleaned segment drops its raw words from the window, or the current segment is part of its own window.
    transcriptQueue = ['um read the config', 'um uh', 'ssh socket path']
    modelAnswers = ['read the config', '', 'ssh socket path']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    expect(healCalls).toEqual([
      { context: '', text: 'um read the config' },
      { context: 'um read the config', text: 'um uh' },
      { context: 'um read the config um uh', text: 'ssh socket path' },
    ])
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(drafts).toEqual(['read the config ssh socket path'])
  })

  it('a truncated heal answer counts as a failure: its raw words stand in, and a trigger in them still fires', async () => {
    // Breaks if: a length-capped answer is used as the cleaned text (it may end mid-sentence), or a trigger in the raw words is missed.
    healMode = 'truncated'
    transcriptQueue = ['um first sentence that is my prompt']
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['um first sentence'])
    expect(recording()).toBe(true)
    expect(errorText()).toBeNull()
    expect(transcriptLog).toEqual([
      { raw: 'um first sentence that is my prompt', cleaned: 'um first sentence', source: 'raw', action: 'submit' },
    ])
    expectHostOnlyFetches()
  })

  it('a trigger submits at once with autoSubmit on too, and a trigger never writes the draft', async () => {
    // Breaks if: autoSubmit changes when a trigger submits, or a trigger writes the composer draft.
    transcriptQueue = ['first sentence that is my prompt']
    await mountAndSettle(AUTO_SUBMIT)
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(sent).toEqual(['first sentence'])
    expect(drafts).toEqual([])
    expect(recording()).toBe(true)
  })

  it('a trigger heard during the stop is cut and its tail dropped, and with autoSubmit on the prompt is submitted only at the stop', async () => {
    // Breaks if: the trigger submits before the stop's final step, the tail after the phrase reaches the prompt, or the trail records the wrong action.
    transcriptQueue = ['first sentence', 'second sentence that is my prompt and more']
    await mountAndSettle(AUTO_SUBMIT)
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    expect(sent).toEqual([])
    await act(async () => {
      button().click()
    })
    await flush()
    await flush()
    expect(sent).toEqual(['first sentence second sentence'])
    expect(drafts).toEqual([])
    expect(transcriptLog).toEqual([
      { raw: 'first sentence second sentence that is my prompt and more', cleaned: 'first sentence second sentence', source: 'model', action: 'submit' },
    ])
  })

  it('a segment that reaches the queue after a closing trigger changes neither the prompt nor the draft, stays raw in the trail, and makes no heal call', async () => {
    // Breaks if: a latched capture asks the model again, a late segment's words reach the prompt or the draft, or they are missing from the trail's raw text.
    transcriptQueue = ['first sentence', 'second sentence that is my prompt', 'late words']
    holdHeals()
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await speak(1000)
    expect(healCalls.map(call => call.text)).toEqual(['first sentence'])
    await act(async () => {
      button().click()
    })
    await flush()
    openHealGate()
    await flush()
    await flush()
    await flush()
    expect(healCalls).toEqual([
      { context: '', text: 'first sentence' },
      { context: 'first sentence', text: 'second sentence that is my prompt' },
    ])
    expect(drafts).toEqual(['first sentence second sentence'])
    expect(sent).toEqual([])
    expect(transcriptLog).toEqual([
      { raw: 'first sentence second sentence that is my prompt late words', cleaned: 'first sentence second sentence', source: 'model', action: 'draft' },
    ])
  })

  it('a stop with several queued segments makes one heal call per segment, each with its own window', async () => {
    // Breaks if: a queued segment gets no call of its own, the stop asks once for the whole transcript, or a window is taken before the segment before it is appended.
    transcriptQueue = ['alpha one', 'beta two', 'gamma three']
    holdHeals()
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await advance(1000)
    await speak(1000)
    await act(async () => {
      button().click()
    })
    await flush()
    openHealGate()
    await flush()
    await flush()
    await flush()
    expect(healCalls).toEqual([
      { context: '', text: 'alpha one' },
      { context: 'alpha one', text: 'beta two' },
      { context: 'alpha one beta two', text: 'gamma three' },
    ])
    expect(drafts).toEqual(['alpha one beta two gamma three'])
  })
})
