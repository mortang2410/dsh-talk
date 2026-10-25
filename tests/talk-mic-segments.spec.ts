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
import { healTranscript } from '../src/heal.ts'
import type { TalkHeal, TalkStatus, TalkTranscript, TalkVoiceTurn } from '../src/wire.ts'

const TICK_MS = 500
/** The capture rate. At 16 kHz a 1000-sample block is 62.5 ms, and no resampling happens. */
const SAMPLE_RATE = 16000
const BLOCK_SAMPLES = 1000
const SPEECH_LEVEL = 0.2
/** Under the energy floor (record.vad.energyThreshold is 0.01), so not speech. */
const SOFT_LEVEL = 0.005
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
  record: { enabled: true, hotkey: null, autoSubmit: false, vad: { silenceMs: 1000, energyThreshold: 0.01 } },
  interrupt: false,
  patchFile: null,
}

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
/** Microphone tracks stopped so far (every track the button releases). */
let trackStops = 0
/** When set, every host heal call waits on it before answering (the call is in flight). */
let healGate: Promise<void> | null = null
let openHealGate: () => void = () => {}
/** The microphone level the test feeds into the capture worklet. */
let micLevel = 0
/** What the host's model call does: answer like the model, or fail (the host fallback then decides). */
let healMode: 'model' | 'error' = 'model'
/** Text the button wrote into the composer draft. */
const drafts: string[] = []
/** Texts the button submitted. */
const sent: string[] = []
/** Every audio context the button opened, in order. */
const contexts: FakeAudioContext[] = []
/** Every capture worklet node the button created, in order. The test feeds blocks to their ports. */
const workletNodes: FakeWorkletNode[] = []

const FINAL_PHRASE = /\bthat(?: is |'s )my prompt\b[.!?]?\s*$/iu

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
  const raw = (body.messages[1]?.content ?? '').replace(/^raw transcript: /u, '')
  events.push(`heal:${raw}`)
  if (healMode === 'error') return { ok: false, status: 500, text: async () => '' }
  // A stand-in for the model's filler removal: "um" is dropped wherever it stands.
  const cleaned = raw.replace(/\bum\s+/giu, '')
  const trigger = FINAL_PHRASE.test(raw)
  const prompt = trigger ? cleaned.replace(FINAL_PHRASE, '').trim() : cleaned
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
  transcriptQueue = []
  micLevel = 0
  healMode = 'model'
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

  it('unmounting mid-capture releases the microphone and the capture graph, and a heal answer that arrives later submits nothing', async () => {
    // Blocker 1 (P-F6): the stale closure must not submit, and nothing may keep capturing.
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
    // Blocker 2 (P-F2): the answer lands after the trip; it must not submit. The stop heals the text
    // again, so the composer gets the text before the phrase, and the phrase itself is not written.
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
  })

  it('a trigger whose voice-turn arm answers after unmount submits nothing', async () => {
    // Review round 1 F4: the arm is awaited after the trigger decision, so an unmount during the
    // await must stop the submit that follows it.
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

  it('a stop writes the cleaned text, not the raw transcript, when the last heal call did not fire', async () => {
    // B4: the heal answer for the segment is kept, and the stop writes its cleaned prompt to the composer.
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
    expect(healCalls).toEqual(['um first sentence'])
    expect(drafts).toEqual(['first sentence'])
    expect(sent).toEqual([])
  })

  it('a stop heals the text no heal call judged, and writes its cleaned form', async () => {
    // B4: the open segment's words arrive after the stop began, so no heal call judged them. The stop asks heal once.
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
    expect(healCalls).toEqual(['first part', 'first part um last words'])
    expect(drafts).toEqual(['first part last words'])
    expect(sent).toEqual([])
  })

  it('a stop keeps the raw text when the heal call fails and the text has no trigger phrase', async () => {
    // B4: the stop-time heal fails too, so the words the speaker said are written as they were transcribed.
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

  it('a stop heals the text whose heal call failed during the capture', async () => {
    // B4: the heal call for the segment failed, so no answer covers its words. The stop heals them once the host answers.
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
    expect(healCalls).toEqual(['um first sentence', 'um first sentence'])
    expect(drafts).toEqual(['first sentence'])
  })

  it('a stop after the trigger phrase writes the text before the phrase to the composer, and sends nothing', async () => {
    // B4: a trigger heard at the stop is not submitted (the capture is ending), and the phrase is not written.
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
    expect(healCalls).toEqual(['first sentence', 'first sentence second sentence that is my prompt'])
    expect(drafts).toEqual(['first sentence second sentence'])
    expect(sent).toEqual([])
  })

  it('a trigger heal still in flight at the stop is reused, not asked again, and sends nothing', async () => {
    // Review round 5, R5-1: the trigger answer arrives after the stop began. Its text before the phrase is the
    // stop's cleaned text, so the stop makes no second heal call and submits nothing.
    transcriptQueue = ['one thought that is my prompt']
    holdHeals()
    await mountAndSettle()
    await startCapture()
    await speak(1000)
    await advance(1000)
    expect(healCalls).toEqual(['one thought that is my prompt'])
    await act(async () => {
      button().click()
    })
    await flush()
    openHealGate()
    await flush()
    await flush()
    expect(healCalls).toEqual(['one thought that is my prompt'])
    expect(drafts).toEqual(['one thought'])
    expect(sent).toEqual([])
  })

  it('a stop submits the words said so far when the autoSubmit setting is on', async () => {
    // B14: the host path honours record.autoSubmit at the stop, as the Web Speech path does.
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
})
