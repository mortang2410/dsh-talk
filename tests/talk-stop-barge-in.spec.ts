// @vitest-environment jsdom
/**
 * Barge-in while a reply is audible. The microphone is acquired when a reply starts,
 * released when it ends by any route, and never held while idle. A capture takes over the
 * listener's stream. Speech reaches the stop through the real component: the listener tick
 * (the reply's own microphone) and the capture's segment tick. The microphone, audio
 * context and recorder are stubbed; the tests drive the analyser level directly.
 *
 * @module dsh-talk/test/talk-stop-barge-in.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { TalkMicButton } from '../src/client/TalkMicButton.tsx'
import { SPEECH_POLL_MS } from '../src/client/speechPoll.ts'
import type { TalkAudio, TalkLatest, TalkStatus } from '../src/wire.ts'

const SESSION = 'session-barge-in-under-test'
const LABEL = 'Stop playback, or cancel the running turn'
const TICK_MS = 500
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

/** The analyser reads a constant level: above the energy threshold is speech, zero is silence. */
let level = 0
let micDenied = false
let micStreams: FakeStream[] = []
const getUserMedia = vi.fn(async (): Promise<FakeStream> => {
  if (micDenied) throw new Error('NotAllowedError')
  const stream = new FakeStream()
  micStreams.push(stream)
  return stream
})
const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')

class FakeContext {
  static instances: FakeContext[] = []
  closed = 0
  constructor() {
    FakeContext.instances.push(this)
  }
  createMediaStreamSource(_stream: unknown): { connect(target: unknown): void } {
    return { connect() {} }
  }
  createAnalyser(): { fftSize: number; connect(target: unknown): void; getFloatTimeDomainData(buffer: Float32Array): void } {
    return { fftSize: 0, connect() {}, getFloatTimeDomainData(buffer: Float32Array) { buffer.fill(level) } }
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
    level = 0
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

  it('listener path: speech during an audible reply reaches the stop and releases the microphone', async () => {
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    expect(micStreams.length).toBe(1)

    level = 0.5
    await advance(TICK_MS)
    expect(FiringAudio.instances[0]?.pauses).toBe(1)
    expect(turn.cancels).toEqual([])
    expect(micStreams[0]?.stopped).toBe(1)
  })

  it('with a turn running, speech during the reply stops it and cancels the turn once', async () => {
    mount()
    await advance(0)
    setRunning(true)
    await publish('u1')
    await advance(0)

    level = 0.5
    await advance(TICK_MS)
    expect(FiringAudio.instances[0]?.pauses).toBe(1)
    expect(turn.cancels).toEqual([SESSION])

    level = 0.5
    await advance(2000)
    expect(turn.cancels).toEqual([SESSION])
  })

  it('speech with nothing audible opens no microphone and cancels nothing', async () => {
    mount()
    await advance(0)
    setRunning(true)
    level = 0.5
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

  it('capture path: the capture takes the listener stream, and speech through the segment tick stops the reply', async () => {
    mount()
    await advance(0)
    await publish('u1')
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledTimes(1)
    const player = FiringAudio.instances[0]

    await pressMic()
    await advance(0)
    expect(getUserMedia).toHaveBeenCalledTimes(1)

    level = 0.5
    await advance(TICK_MS)
    expect(player?.pauses).toBe(1)
    expect(getUserMedia).toHaveBeenCalledTimes(1)

    level = 0
    await pressMic()
    await advance(0)
    expect(micStreams[0]?.stopped).toBe(1)
  })

  it('the stop control carries the localized label as title and aria-label', async () => {
    mount()
    await advance(0)
    setRunning(true)
    expect(stopButton()?.getAttribute('aria-label')).toBe(LABEL)
    expect(stopButton()?.getAttribute('title')).toBe(LABEL)
  })
})
