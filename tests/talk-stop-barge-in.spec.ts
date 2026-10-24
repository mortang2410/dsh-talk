// @vitest-environment jsdom
/**
 * Barge-in while a reply is audible. The microphone is acquired when a reply starts and released when it
 * ends by any route. Barge-in comes from the listener alone: an analyser on the reply's microphone, or on a
 * running capture's stream. A capture decides segmentation only, and a press on the mic stops the reply first.
 * The microphone, audio context and recorder are stubbed. Each AudioContext has its own frame queue: each
 * analyser read takes the next level, and silence once the queue is empty.
 *
 * @module dsh-talk/test/talk-stop-barge-in.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { TalkMicButton } from '../src/client/TalkMicButton.tsx'
import { SPEECH_POLL_MS } from '../src/client/speechPoll.ts'
import {
  advanceSpeechRun,
  LISTENER_CONSECUTIVE_FRAMES,
  LISTENER_FLOOR_MAX,
  LISTENER_FLOOR_MIN,
  LISTENER_ONSET_MS,
  LISTENER_TICK_MS,
  listenerFloor,
} from '../src/client/vad.ts'
import type { TalkAudio, TalkLatest, TalkStatus } from '../src/wire.ts'

const SESSION = 'session-barge-in-under-test'
const LABEL = 'Stop playback, or cancel the running turn'
const TICK_MS = 500
const MIC_OPTIONS = { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
const SETTINGS: TalkStatus = {
  stt: { engine: 'whisper', resolved: 'whisper', language: 'en-US', silenceFinaliseMs: 1500, funasrUrl: null, whisperModel: null },
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

/** An audio element that stays audible until a test fires an end event or the stop control stops it. */
class FiringAudio {
  static instances: FiringAudio[] = []
  pauses = 0
  plays = 0
  duration = Number.NaN
  private handlers = new Map<string, Array<() => void>>()
  constructor(_src: string) {
    FiringAudio.instances.push(this)
  }
  addEventListener(type: string, handler: () => void): void {
    this.handlers.set(type, [...(this.handlers.get(type) ?? []), handler])
  }
  fire(type: string): void {
    for (const handler of this.handlers.get(type) ?? []) handler()
  }
  play(): Promise<void> {
    this.plays += 1
    return Promise.resolve()
  }
  pause(): void {
    this.pauses += 1
  }
}

/** A microphone stream that counts how many of its tracks were stopped. */
class FakeStream {
  stopped = 0
  getTracks(): Array<{ stop: () => void }> {
    return [{ stop: () => { this.stopped += 1 } }]
  }
}

let micDenied = false
/** When true, the next microphone request is held until the test resolves it. */
let micDeferred = false
let deferredResolve: ((stream: FakeStream) => void) | null = null
let micStreams: FakeStream[] = []
const getUserMedia = vi.fn((_constraints?: unknown): Promise<FakeStream> => {
  if (micDenied) return Promise.reject(new Error('NotAllowedError'))
  if (micDeferred) {
    // One-shot: only the next request is held.
    micDeferred = false
    return new Promise<FakeStream>((resolve) => { deferredResolve = resolve })
  }
  const stream = new FakeStream()
  micStreams.push(stream)
  return Promise.resolve(stream)
})
const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')

class FakeContext {
  static instances: FakeContext[] = []
  /** The stream this context's source was built on. */
  source: unknown = undefined
  /** The levels the next analyser reads take, one per read. */
  queue: number[] = []
  closed = 0
  constructor() {
    FakeContext.instances.push(this)
  }
  createMediaStreamSource(stream: unknown): { connect(target: unknown): void } {
    this.source = stream
    return { connect() {} }
  }
  createAnalyser(): { fftSize: number; connect(target: unknown): void; getFloatTimeDomainData(buffer: Float32Array): void } {
    return {
      fftSize: 0,
      connect() {},
      getFloatTimeDomainData: (buffer: Float32Array) => {
        buffer.fill(this.queue.shift() ?? 0)
      },
    }
  }
  close(): Promise<void> {
    this.closed += 1
    return Promise.resolve()
  }
}

class FakeRecorder {
  static instances: FakeRecorder[] = []
  static isTypeSupported(_type: string): boolean {
    return true
  }
  state: 'inactive' | 'recording' = 'inactive'
  ondataavailable: ((event: { data: { size: number } }) => void) | null = null
  onstop: (() => void) | null = null
  constructor(_stream: unknown, _options: unknown) {
    FakeRecorder.instances.push(this)
  }
  start(): void {
    this.state = 'recording'
  }
  stop(): void {
    this.state = 'inactive'
    this.onstop?.()
  }
}

interface TurnState {
  running: boolean
  listeners: Array<(running: boolean) => void>
  cancels: string[]
}
let turn: TurnState = { running: false, listeners: [], cancels: [] }
let remoteLatest: TalkLatest | null = null
let container: HTMLElement
let root: Root | undefined
const originalCreate = (URL as unknown as { createObjectURL?: unknown }).createObjectURL
const originalRevoke = (URL as unknown as { revokeObjectURL?: unknown }).revokeObjectURL

function utterance(utteranceId: string): TalkLatest {
  return {
    sessionId: SESSION,
    utteranceId,
    engine: 'edge-tts',
    text: `spoken ${utteranceId}`,
    audioBytes: 3,
    reason: 'speak-tool',
    logged: true,
    appended: 1,
    skipped: 0,
  }
}

function mount(): void {
  const props = {
    status: async () => SETTINGS,
    interrupt: async () => ({ stopped: false }),
    transcribe: async () => ({ text: '', engine: 'whisper', language: 'en' }),
    audio: async (utteranceId: string) => {
      const entry: TalkAudio = { utteranceId, mime: 'audio/mpeg', data: 'AAAA', text: '', engine: 'edge-tts' }
      return entry
    },
    latest: async () => remoteLatest,
    setDraft: () => {},
    send: async () => {},
    sessionId: SESSION,
    stopLabel: LABEL,
    watchTurn: (_sessionId: string, listener: (running: boolean) => void) => {
      turn.listeners.push(listener)
      listener(turn.running)
      return () => {
        turn.listeners = turn.listeners.filter((candidate) => candidate !== listener)
      }
    },
    cancelTurn: async (sessionId: string) => {
      turn.cancels.push(sessionId)
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

async function advance(ms: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

function setRunning(running: boolean): void {
  turn.running = running
  act(() => {
    for (const listener of turn.listeners) listener(running)
  })
}

/** Publish one reply and let the poll start it. */
async function publish(utteranceId: string): Promise<void> {
  remoteLatest = utterance(utteranceId)
  await advance(SPEECH_POLL_MS)
}

function stopButton(): HTMLButtonElement | null {
  return container.querySelector<HTMLButtonElement>('[data-dsh-talk-stop]')
}

function pressStop(): void {
  act(() => {
    stopButton()?.click()
  })
}

async function pressMic(): Promise<void> {
  await act(async () => {
    container.querySelector<HTMLButtonElement>('[data-dsh-talk-mic]')?.click()
  })
}

/** A run of frames above the listener floor. */
function speech(count: number): number[] {
  return Array.from({ length: count }, () => 0.5)
}

/** The AudioContexts built on one stream, oldest first: a capture's analyser is built before a listener's on its stream. */
function contextsOn(stream: FakeStream | undefined): FakeContext[] {
  return FakeContext.instances.filter((context) => context.source === stream)
}

/** Set the analyser levels of one AudioContext on a stream (index 0 is the oldest). */
function queueOn(stream: FakeStream | undefined, levels: number[], index = 0): void {
  const context = contextsOn(stream)[index]
  if (context === undefined) throw new Error('no AudioContext on this stream yet')
  context.queue = levels
}

/** Resolve the held microphone request with a fresh stream, at the moment the test chooses. */
async function resolveDeferredMic(): Promise<FakeStream> {
  const stream = new FakeStream()
  micStreams.push(stream)
  const resolve = deferredResolve
  deferredResolve = null
  micDeferred = false
  await act(async () => {
    resolve?.(stream)
    await vi.advanceTimersByTimeAsync(0)
  })
  return stream
}

describe('listener run (pure)', () => {
  it('a speech frame extends the run and a quiet frame resets it to zero', () => {
    expect(advanceSpeechRun(0, true)).toBe(1)
    expect(advanceSpeechRun(2, true)).toBe(3)
    expect(advanceSpeechRun(3, false)).toBe(0)
  })

  it('sustained speech takes several frames, one frame is never enough', () => {
    expect(LISTENER_CONSECUTIVE_FRAMES).toBeGreaterThan(1)
    expect(LISTENER_TICK_MS * LISTENER_CONSECUTIVE_FRAMES).toBeGreaterThanOrEqual(300)
    expect(LISTENER_ONSET_MS).toBeGreaterThan(0)
  })
})

describe('listener floor (pure, clamped)', () => {
  it('the default threshold 0.01 gives the 0.05 floor, unchanged', () => {
    expect(listenerFloor(0.01)).toBeCloseTo(0.05, 10)
  })

  it('a threshold of 0.2 is held at the ceiling, not 1.0, so speech can still trip the listener', () => {
    expect(listenerFloor(0.2)).toBe(LISTENER_FLOOR_MAX)
    expect(listenerFloor(0.2)).toBeLessThan(1)
  })

  it('a threshold of 0 is raised to the minimum, so silence-level noise cannot trip the listener', () => {
    expect(listenerFloor(0)).toBe(LISTENER_FLOOR_MIN)
  })
})

describe('barge-in while a reply is audible', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => false })
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } })
    vi.stubGlobal('Audio', FiringAudio)
    vi.stubGlobal('AudioContext', FakeContext)
    vi.stubGlobal('MediaRecorder', FakeRecorder)
    Object.assign(URL, { createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} })
    FiringAudio.instances = []
    FakeContext.instances = []
    FakeRecorder.instances = []
    micStreams = []
    micDenied = false
    micDeferred = false
    deferredResolve = null
    getUserMedia.mockClear()
    turn = { running: false, listeners: [], cancels: [] }
    remoteLatest = null
    container = document.createElement('div')
    document.body.append(container)
  })

  afterEach(() => {
    unmount()
    container.remove()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    if (originalMediaDevices !== undefined) Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices)
    else Reflect.deleteProperty(navigator, 'mediaDevices')
    Reflect.deleteProperty(document, 'hidden')
    Object.assign(URL, { createObjectURL: originalCreate, revokeObjectURL: originalRevoke })
  })

  it('acquires the microphone when a reply starts and releases it on stop, without reopening while idle', async () => {
    mount()
    await advance(0)
    expect(getUserMedia).not.toHaveBeenCalled()

    await publish('u1')
    expect(getUserMedia).toHaveBeenCalledTimes(1)
    expect(micStreams[0]?.stopped).toBe(0)

    pressStop()
    expect(micStreams[0]?.stopped).toBe(1)

    await advance(5000)
    expect(getUserMedia).toHaveBeenCalledTimes(1)
  })

  it('requests the microphone with echo cancellation, noise suppression and gain control on', async () => {
    mount()
    await advance(0)
    await publish('u1')
    expect(getUserMedia).toHaveBeenCalledWith({ audio: MIC_OPTIONS })
  })

  it('the capture microphone request carries the same three options', async () => {
    mount()
    await advance(0)
    await pressMic()
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledWith({ audio: MIC_OPTIONS })
  })

  it.each(['ended', 'error', 'abort'] as const)('releases the microphone when the reply ends by %s', async (event) => {
    mount()
    await advance(0)
    await publish('u1')
    expect(micStreams[0]?.stopped).toBe(0)

    act(() => {
      FiringAudio.instances[0]?.fire(event)
    })
    expect(micStreams[0]?.stopped).toBe(1)
    expect(stopButton()).toBeNull()
  })

  it('one speech frame does not barge in, even a single frame well above the floor', async () => {
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    queueOn(micStreams[0], [0.5])
    await advance(2000)
    expect(FiringAudio.instances[0]?.pauses).toBe(0)
    expect(micStreams[0]?.stopped).toBe(0)
  })

  it('the required run of consecutive speech frames does barge in', async () => {
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    queueOn(micStreams[0], speech(LISTENER_CONSECUTIVE_FRAMES))
    await advance(LISTENER_ONSET_MS + LISTENER_TICK_MS * LISTENER_CONSECUTIVE_FRAMES + LISTENER_TICK_MS)
    expect(FiringAudio.instances[0]?.pauses).toBe(1)
    expect(micStreams[0]?.stopped).toBe(1)
  })

  it('one fewer than the required run does not barge in', async () => {
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    queueOn(micStreams[0], speech(LISTENER_CONSECUTIVE_FRAMES - 1))
    await advance(3000)
    expect(FiringAudio.instances[0]?.pauses).toBe(0)
  })

  it('a quiet frame between speech frames resets the count', async () => {
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    // Two speech frames, one quiet, then three speech frames: no run of four, so no barge-in.
    queueOn(micStreams[0], [0.5, 0.5, 0, 0.5, 0.5, 0.5])
    await advance(3000)
    expect(FiringAudio.instances[0]?.pauses).toBe(0)
    // A fresh run of four speech frames, with nothing carried over from before the quiet frame, does barge in.
    queueOn(micStreams[0], speech(LISTENER_CONSECUTIVE_FRAMES))
    await advance(1000)
    expect(FiringAudio.instances[0]?.pauses).toBe(1)
  })

  it('frames during the playback onset window are not read and do not barge in', async () => {
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    queueOn(micStreams[0], speech(LISTENER_CONSECUTIVE_FRAMES))
    await advance(LISTENER_ONSET_MS - LISTENER_TICK_MS)
    expect(FiringAudio.instances[0]?.pauses).toBe(0)
    expect(contextsOn(micStreams[0])[0]?.queue.length).toBe(LISTENER_CONSECUTIVE_FRAMES)
  })

  it('a new reply starts its count from zero, so one speech frame after a stop does not barge in', async () => {
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    // Three speech frames read at 500, 600 and 700 ms after reply one starts. Then the reply is stopped by
    // hand at 700 ms, before any quiet frame could reset the count of three.
    queueOn(micStreams[0], [0.5, 0.5, 0.5])
    await advance(LISTENER_ONSET_MS + LISTENER_TICK_MS * 2)
    pressStop()
    expect(FiringAudio.instances[0]?.pauses).toBe(1)

    // Reply two: a single speech frame after its onset must not complete a run carried over from reply one.
    await publish('u2')
    await advance(0)
    queueOn(micStreams[1], [0.5])
    await advance(LISTENER_ONSET_MS + LISTENER_TICK_MS * 3)
    expect(FiringAudio.instances[1]?.pauses).toBe(0)
  })

  it('a late listener does not carry the run of the reply it interrupted: one speech frame after its onset does not barge in', async () => {
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    // Reply one reaches a run of four and is interrupted by it, so the run is left at the threshold.
    queueOn(micStreams[0], speech(LISTENER_CONSECUTIVE_FRAMES))
    await advance(LISTENER_ONSET_MS + LISTENER_TICK_MS * LISTENER_CONSECUTIVE_FRAMES + LISTENER_TICK_MS)
    expect(FiringAudio.instances[0]?.pauses).toBe(1)

    // Reply two's microphone request is held past its onset window, so its listener opens late.
    micDeferred = true
    await publish('u2')
    await advance(LISTENER_ONSET_MS + LISTENER_TICK_MS)
    expect(getUserMedia).toHaveBeenCalledTimes(2)
    const late = await resolveDeferredMic()
    await advance(0)
    queueOn(late, [0.5])
    await advance(TICK_MS)
    expect(FiringAudio.instances[1]?.pauses).toBe(0)
  })

  it('with a turn running, sustained speech during the reply stops it and cancels the turn once', async () => {
    mount()
    await advance(0)
    setRunning(true)
    await publish('u1')
    await advance(0)

    queueOn(micStreams[0], speech(20))
    await advance(TICK_MS * 4)
    expect(FiringAudio.instances[0]?.pauses).toBe(1)
    expect(turn.cancels).toEqual([SESSION])

    await advance(2000)
    expect(turn.cancels).toEqual([SESSION])
  })

  it('speech with nothing audible opens no microphone and cancels nothing', async () => {
    mount()
    await advance(0)
    setRunning(true)
    await advance(2000)
    expect(getUserMedia).not.toHaveBeenCalled()
    expect(turn.cancels).toEqual([])
  })

  it('a denied microphone logs once, and the next reply does not retry the request', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    micDenied = true
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledTimes(1)

    pressStop()
    await publish('u2')
    await advance(0)
    expect(FiringAudio.instances.length).toBe(2)
    expect(getUserMedia).toHaveBeenCalledTimes(1)
    expect(warn).toHaveBeenCalledTimes(1)
    warn.mockRestore()
  })

  it('pressing the mic while a reply plays stops that playback at once, opens the capture on its own microphone, and cancels nothing', async () => {
    mount()
    await advance(0)
    setRunning(true)
    await publish('u1')
    await advance(0)
    const player = FiringAudio.instances[0]
    expect(player?.pauses).toBe(0)

    await pressMic()
    await advance(0)
    expect(player?.pauses).toBe(1)
    expect(stopButton()?.dataset.playing).toBe('false')
    expect(micStreams[0]?.stopped).toBe(1)
    expect(getUserMedia).toHaveBeenCalledTimes(2)
    expect(turn.cancels).toEqual([])
  })

  it('a capture tick never cancels the turn: the reply echo in the capture stream does not barge in', async () => {
    mount()
    await advance(0)
    setRunning(true)
    await pressMic()
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledTimes(1)
    // The reply starts while the capture runs: its listener reads the capture's stream, with no second microphone.
    await publish('u1')
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledTimes(1)
    const player = FiringAudio.instances[0]

    // The capture (oldest context on its stream) reads sustained speech, the echo level. The listener reads silence.
    queueOn(micStreams[0], speech(20), 0)
    await advance(TICK_MS * 4)
    expect(turn.cancels).toEqual([])
    expect(player?.pauses).toBe(0)
  })

  it('during a capture, sustained speech on the listener still stops the reply and cancels the turn once', async () => {
    mount()
    await advance(0)
    setRunning(true)
    await pressMic()
    await advance(0)
    await publish('u1')
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledTimes(1)

    // The listener (the newer context on the capture stream) reads sustained speech after its onset.
    queueOn(micStreams[0], speech(20), 1)
    await advance(TICK_MS * 4)
    expect(FiringAudio.instances[0]?.pauses).toBe(1)
    expect(turn.cancels).toEqual([SESSION])
    expect(getUserMedia).toHaveBeenCalledTimes(1)
  })

  it('a listener request still held when the capture starts is released on arrival, and the capture keeps its own microphone', async () => {
    mount()
    await advance(0)
    micDeferred = true
    await publish('u1')
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledTimes(1)

    // The press stops the reply. The capture opens its own microphone while the listener request is still held.
    await pressMic()
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledTimes(2)
    expect(FiringAudio.instances[0]?.pauses).toBe(1)

    // A reply plays during the capture: its listener reads the capture's stream, with no third request.
    await publish('u2')
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledTimes(2)

    // The held request resolves now, during the capture: its stream is stopped on arrival, the capture's is not.
    const held = await resolveDeferredMic()
    expect(held.stopped).toBe(1)
    expect(micStreams[0]?.stopped).toBe(0)
  })

  it('the stop control carries the localized label as title and aria-label', async () => {
    mount()
    await advance(0)
    setRunning(true)
    expect(stopButton()?.getAttribute('aria-label')).toBe(LABEL)
    expect(stopButton()?.getAttribute('title')).toBe(LABEL)
  })
})
