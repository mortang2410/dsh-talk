// @vitest-environment jsdom
/**
 * The two-stage stop control and the barge-in decision. The pure rules are
 * checked directly. The component tests render the real record button under
 * jsdom with the remote surface and the turn services stubbed, and check the
 * button wiring: one press stops audible playback and leaves the turn running,
 * the next press cancels a running turn, the control shows only while it can
 * act, and a stopped reply does not hold the next reply.
 *
 * @module dsh-talk/test/talk-stop-control.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { TalkMicButton } from '../src/client/TalkMicButton.tsx'
import { planBargeIn, planStopPress, stopControlVisible, turnCancellable } from '../src/client/present.ts'
import { SPEECH_POLL_MS } from '../src/client/speechPoll.ts'
import type { TalkAudio, TalkLatest, TalkStatus } from '../src/wire.ts'

const SESSION = 'session-stop-under-test'
const SETTINGS: TalkStatus = {
  stt: { engine: 'web', resolved: 'web', language: 'en-US', silenceFinaliseMs: 1500, funasrUrl: null, whisperModel: null },
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
  record: { enabled: true, hotkey: null, maxSeconds: 60, ceilingSeconds: 600, autoSubmit: false, vad: { enabled: false, silenceMs: 1000, energyThreshold: 0.01 } },
  interrupt: false,
  patchFile: null,
}

/**
 * An audio element that stays audible until something stops it. It fires no
 * events of its own, so a reply ends only through the stop control or the
 * chain's bound. `pause()` is recorded, not turned into an event, so the test
 * fails if the stop does not end the task itself.
 */
class FakeAudio {
  static instances: FakeAudio[] = []
  pauses = 0
  plays = 0
  duration = Number.NaN
  constructor(_src: string) {
    FakeAudio.instances.push(this)
  }
  addEventListener(_type: string, _handler: () => void): void {}
  play(): Promise<void> {
    this.plays += 1
    return Promise.resolve()
  }
  pause(): void {
    this.pauses += 1
  }
}

interface TurnState {
  running: boolean
  /** A pending interaction (an approval) awaits an answer on the session. */
  pending: boolean
  listeners: Array<(cancellable: boolean) => void>
  /** Session ids passed to cancelTurn, one entry per call. */
  cancels: string[]
}
let turn: TurnState = { running: false, pending: false, listeners: [], cancels: [] }

/** The turn signal the built-in guard gives (present.ts turnCancellable), with the session facts this spec sets. */
function cancellable(): boolean {
  return turnCancellable({ running: turn.running, removed: false, subagentMode: null, pendingInteraction: turn.pending, openTurn: turn.running })
}

let remoteLatest: TalkLatest | null = null
let container: HTMLElement
let root: Root | undefined
let clock = 0
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
    watchTurn: (_sessionId: string, listener: (cancellable: boolean) => void) => {
      turn.listeners.push(listener)
      listener(cancellable())
      return () => {
        turn.listeners = turn.listeners.filter((candidate) => candidate !== listener)
      }
    },
    cancelTurn: async (sessionId: string) => {
      if (!cancellable()) return
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
  clock += ms
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

/** Set the session's running flag and notify the mounted button, as the host's turn state does. */
function setRunning(running: boolean): void {
  turn.running = running
  act(() => {
    for (const listener of turn.listeners) listener(cancellable())
  })
}

/** Set or clear a pending interaction and notify the mounted button, as the session status does. */
function setPending(pending: boolean): void {
  turn.pending = pending
  act(() => {
    for (const listener of turn.listeners) listener(cancellable())
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

describe('stop control decisions (pure)', () => {
  it('a press while audio is audible stops it and never cancels the turn', () => {
    const plan = planStopPress({ playing: true, turnRunning: true, armed: false })
    expect(plan.action).toBe('stop-playback')
    expect(plan.armed).toBe(true)
  })

  it('with nothing audible, the first press arms and the second press cancels a running turn', () => {
    const first = planStopPress({ playing: false, turnRunning: true, armed: false })
    expect(first.action).toBe('arm')
    const second = planStopPress({ playing: false, turnRunning: true, armed: first.armed })
    expect(second.action).toBe('cancel-turn')
    expect(second.armed).toBe(false)
  })

  it('with no running turn, a press cancels nothing', () => {
    expect(planStopPress({ playing: false, turnRunning: false, armed: true }).action).toBe('none')
    expect(planStopPress({ playing: false, turnRunning: false, armed: false }).action).toBe('none')
  })

  it('barge-in stops audible playback and cancels only when a turn is running', () => {
    expect(planBargeIn({ playing: true, turnRunning: true })).toEqual({ stopPlayback: true, cancelTurn: true })
    expect(planBargeIn({ playing: true, turnRunning: false })).toEqual({ stopPlayback: true, cancelTurn: false })
  })

  it('barge-in with nothing audible cancels nothing', () => {
    expect(planBargeIn({ playing: false, turnRunning: true })).toEqual({ stopPlayback: false, cancelTurn: false })
    expect(planBargeIn({ playing: false, turnRunning: false })).toEqual({ stopPlayback: false, cancelTurn: false })
  })

  it('the control is visible only while audio plays or a turn runs', () => {
    expect(stopControlVisible({ playing: false, turnRunning: false })).toBe(false)
    expect(stopControlVisible({ playing: true, turnRunning: false })).toBe(true)
    expect(stopControlVisible({ playing: false, turnRunning: true })).toBe(true)
    expect(stopControlVisible({ playing: true, turnRunning: true })).toBe(true)
  })
})

describe('turnCancellable (the built-in Escape guard, pure)', () => {
  const OPEN = { running: true, removed: false, subagentMode: null, pendingInteraction: false, openTurn: true }

  it('allows a cancel when every guard passes, for a session or a continuable subagent', () => {
    expect(turnCancellable(OPEN)).toBe(true)
    expect(turnCancellable({ ...OPEN, subagentMode: 'continuable' })).toBe(true)
  })

  it('refuses when the session is not running', () => {
    expect(turnCancellable({ ...OPEN, running: false })).toBe(false)
  })

  it('refuses when the session was removed', () => {
    expect(turnCancellable({ ...OPEN, removed: true })).toBe(false)
  })

  it('refuses a subagent that is not continuable', () => {
    expect(turnCancellable({ ...OPEN, subagentMode: 'oneshot' })).toBe(false)
  })

  it('refuses while a pending interaction awaits an answer', () => {
    expect(turnCancellable({ ...OPEN, pendingInteraction: true })).toBe(false)
  })

  it('refuses when no turn is open', () => {
    expect(turnCancellable({ ...OPEN, openTurn: false })).toBe(false)
  })
})

describe('stop control (rendered button, stubbed player and turn services)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => false })
    vi.stubGlobal('Audio', FakeAudio)
    Object.assign(URL, { createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} })
    FakeAudio.instances = []
    turn = { running: false, pending: false, listeners: [], cancels: [] }
    remoteLatest = null
    clock = 0
    container = document.createElement('div')
    document.body.append(container)
  })

  afterEach(() => {
    unmount()
    container.remove()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    Reflect.deleteProperty(document, 'hidden')
    Object.assign(URL, { createObjectURL: originalCreate, revokeObjectURL: originalRevoke })
  })

  it('is visible only while audio plays or a turn runs', async () => {
    mount()
    await advance(0)
    expect(stopButton()).toBeNull()

    setRunning(true)
    expect(stopButton()).not.toBeNull()
    setRunning(false)
    expect(stopButton()).toBeNull()

    await publish('u1')
    expect(stopButton()).not.toBeNull()
    pressStop()
    expect(stopButton()).toBeNull()
  })

  it('the first press stops playback and leaves the turn running; the second press cancels it once', async () => {
    mount()
    await advance(0)
    setRunning(true)
    await publish('u1')
    const player = FakeAudio.instances[0]
    expect(player).toBeDefined()

    pressStop()
    expect(player?.pauses).toBe(1)
    expect(turn.cancels).toEqual([])
    expect(stopButton()?.dataset.turnRunning).toBe('true')

    pressStop()
    expect(turn.cancels).toEqual([SESSION])
  })

  it('a press with no running turn stops playback and cancels nothing', async () => {
    mount()
    await advance(0)
    await publish('u1')
    pressStop()
    expect(FakeAudio.instances[0]?.pauses).toBe(1)
    expect(turn.cancels).toEqual([])
    expect(stopButton()).toBeNull()
  })

  it('with a pending interaction, the stop control does not cancel the turn, even on a second press', async () => {
    mount()
    await advance(0)
    setRunning(true)
    setPending(true)
    expect(stopButton()).toBeNull()
    await publish('u1')
    pressStop()
    expect(FakeAudio.instances[0]?.pauses).toBe(1)
    pressStop()
    expect(turn.cancels).toEqual([])
  })

  it('an armed stop lapses after its 500 ms window: a later press with no audio arms again and cancels nothing', async () => {
    mount()
    await advance(0)
    setRunning(true)
    await publish('u1')
    pressStop()
    expect(FakeAudio.instances[0]?.pauses).toBe(1)
    expect(stopButton()?.dataset.armed).toBe('true')

    await advance(600)
    expect(stopButton()?.dataset.armed).toBe('false')

    pressStop()
    expect(turn.cancels).toEqual([])
    expect(stopButton()?.dataset.armed).toBe('true')
  })

  it('after a stop, the next reply plays normally', async () => {
    mount()
    await advance(0)
    setRunning(true)
    await publish('u1')
    pressStop()

    await publish('u2')
    expect(FakeAudio.instances.length).toBe(2)
    expect(FakeAudio.instances[1]?.plays).toBe(1)
  })
})
