// @vitest-environment jsdom
/**
 * The record button's playback wiring, rendered under jsdom with the remote
 * surface stubbed: the button polls `latest` for the session id it was given,
 * requests `audio` once per new utterance id, does not replay after a remount
 * (a page reload), stops polling on unmount, and follows page visibility: a
 * reply that lands while hidden plays once on return. A player that fires
 * pause or abort, or fires nothing at all, does not hold the next reply. The
 * poll rules themselves are covered by `speech-poll.spec.ts`.
 *
 * @module dsh-talk/test/talk-mic-playback.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { TalkMicButton } from '../src/client/TalkMicButton.tsx'
import { PLAYBACK_CAP_MS, SPEECH_POLL_MS } from '../src/client/speechPoll.ts'
import type { TalkAudio, TalkLatest, TalkStatus } from '../src/wire.ts'

const SESSION = 'session-mic-under-test'
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
  record: { enabled: true, hotkey: null, maxSeconds: 60, autoSubmit: false, vad: { enabled: false, silenceMs: 1000, energyThreshold: 0.01 } },
  interrupt: false,
  patchFile: null,
}

/** What the stubbed remote surface has seen and returned. */
interface Remote {
  latestValue: TalkLatest | null
  latestCalls: string[]
  audioCalls: string[]
}

/** Audio elements the button constructed (one per played utterance). */
const constructedAudio: string[] = []

/**
 * What each successive player does: fire ended, fire pause, fire abort, or fire
 * nothing (a stalled element). Consumed in construction order; default 'end'.
 */
type PlayerScript = 'end' | 'pause' | 'abort' | 'hang' | 'hang-with-duration'
const playerScripts: PlayerScript[] = []

/** Clip length the 'hang-with-duration' script reports, in seconds. */
const HANG_DURATION_S = 2

class FakeAudio {
  private readonly handlers = new Map<string, Array<() => void>>()
  private readonly script: PlayerScript
  /** NaN until the script reports a length, as a real element does before loadedmetadata. */
  duration = Number.NaN
  constructor(src: string) {
    constructedAudio.push(src)
    this.script = playerScripts.shift() ?? 'end'
  }
  addEventListener(type: string, handler: () => void): void {
    this.handlers.set(type, [...(this.handlers.get(type) ?? []), handler])
  }
  play(): Promise<void> {
    // The script fires on the next microtask, the way a real element does after play().
    return Promise.resolve().then(() => {
      if (this.script === 'hang') return
      if (this.script === 'hang-with-duration') {
        this.duration = HANG_DURATION_S
        for (const handler of this.handlers.get('loadedmetadata') ?? []) handler()
        return
      }
      const event = this.script === 'end' ? 'ended' : this.script
      for (const handler of this.handlers.get(event) ?? []) handler()
    })
  }
}

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

let container: HTMLElement
let root: Root | undefined
let remote: Remote
let hiddenNow = false
const originalCreate = (URL as unknown as { createObjectURL?: unknown }).createObjectURL
const originalRevoke = (URL as unknown as { revokeObjectURL?: unknown }).revokeObjectURL

function mount(): void {
  const props = {
    status: async () => SETTINGS,
    interrupt: async () => ({ stopped: false }),
    transcribe: async () => ({ text: '', engine: 'whisper', language: 'en' }),
    audio: async (utteranceId: string) => {
      remote.audioCalls.push(utteranceId)
      const entry: TalkAudio = { utteranceId, mime: 'audio/mpeg', data: 'AAAA', text: '', engine: 'edge-tts' }
      return entry
    },
    latest: async (sessionId: string) => {
      remote.latestCalls.push(sessionId)
      return remote.latestValue
    },
    setDraft: () => {},
    send: async () => {},
    sessionId: SESSION,
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

/** Change page visibility and fire the event the browser fires on a change. */
function setHidden(value: boolean): void {
  hiddenNow = value
  act(() => {
    document.dispatchEvent(new Event('visibilitychange'))
  })
}

async function advance(ms: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

describe('TalkMicButton playback wiring (stubbed talk remote)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    hiddenNow = false
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hiddenNow })
    vi.stubGlobal('Audio', FakeAudio)
    Object.assign(URL, { createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} })
    constructedAudio.length = 0
    playerScripts.length = 0
    remote = { latestValue: null, latestCalls: [], audioCalls: [] }
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

  it('polls for its own session and requests audio once per new utterance id', async () => {
    mount()
    await advance(0)
    remote.latestValue = utterance('u1')
    await advance(SPEECH_POLL_MS)
    expect(remote.audioCalls).toEqual(['u1'])
    expect(constructedAudio).toHaveLength(1)
    await advance(SPEECH_POLL_MS * 3)
    expect(remote.audioCalls).toEqual(['u1'])
    expect(constructedAudio).toHaveLength(1)
    expect(new Set(remote.latestCalls)).toEqual(new Set([SESSION]))
  })

  it('a remount (page reload) does not replay the utterance cached before it, and still plays a new one', async () => {
    remote.latestValue = utterance('before-reload')
    mount()
    await advance(0)
    unmount()
    mount()
    await advance(SPEECH_POLL_MS * 2)
    expect(remote.audioCalls).toEqual([])
    expect(constructedAudio).toHaveLength(0)
    // Positive control: the poll is alive after the remount, so a new reply plays.
    remote.latestValue = utterance('after-reload')
    await advance(SPEECH_POLL_MS * 2)
    expect(remote.audioCalls).toEqual(['after-reload'])
    expect(constructedAudio).toHaveLength(1)
  })

  it('unmounting stops the poll', async () => {
    mount()
    await advance(0)
    const callsBeforeUnmount = remote.latestCalls.length
    expect(callsBeforeUnmount).toBeGreaterThan(0)
    unmount()
    await advance(SPEECH_POLL_MS * 5)
    expect(remote.latestCalls.length).toBe(callsBeforeUnmount)
  })

  it('a reply that lands while the page is hidden plays once on return, and nothing is requested or played while hidden', async () => {
    remote.latestValue = utterance('before')
    mount()
    await advance(0)
    setHidden(true)
    const callsWhenHidden = remote.latestCalls.length
    remote.latestValue = utterance('while-hidden')
    await advance(SPEECH_POLL_MS * 4)
    expect(remote.latestCalls.length).toBe(callsWhenHidden)
    expect(remote.audioCalls).toEqual([])
    expect(constructedAudio).toHaveLength(0)
    setHidden(false)
    await advance(SPEECH_POLL_MS)
    expect(remote.audioCalls).toEqual(['while-hidden'])
    expect(constructedAudio).toHaveLength(1)
    await advance(SPEECH_POLL_MS * 3)
    expect(constructedAudio).toHaveLength(1)
  })

  it('returning to a page whose last reply is still current does not replay it', async () => {
    // Replay-on-return guard, the component twin of the speech-poll test. It
    // catches a seen set cleared on hide; a reversed seed rule passes it.
    remote.latestValue = null
    mount()
    await advance(0)
    remote.latestValue = utterance('last')
    await advance(SPEECH_POLL_MS)
    expect(constructedAudio).toHaveLength(1)
    setHidden(true)
    setHidden(false)
    await advance(SPEECH_POLL_MS * 3)
    expect(remote.audioCalls).toEqual(['last'])
    expect(constructedAudio).toHaveLength(1)
  })

  it('a player that fires pause or abort releases the next reply at once, without waiting for the cap', async () => {
    playerScripts.push('pause', 'abort', 'end')
    remote.latestValue = null
    mount()
    await advance(0)
    remote.latestValue = utterance('p1')
    await advance(SPEECH_POLL_MS * 2)
    expect(constructedAudio).toHaveLength(1)
    remote.latestValue = utterance('a1')
    await advance(SPEECH_POLL_MS * 2)
    expect(constructedAudio).toHaveLength(2)
    remote.latestValue = utterance('e1')
    await advance(SPEECH_POLL_MS * 2)
    expect(constructedAudio).toHaveLength(3)
  })

  it('a player that fires neither ended, error, pause nor abort does not block the next reply: the 120 s cap releases it', async () => {
    playerScripts.push('hang', 'end')
    remote.latestValue = null
    mount()
    await advance(0)
    remote.latestValue = utterance('h1')
    await advance(SPEECH_POLL_MS * 2)
    expect(constructedAudio).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advance(SPEECH_POLL_MS * 2)
    expect(constructedAudio).toHaveLength(1)
    await advance(PLAYBACK_CAP_MS)
    expect(constructedAudio).toHaveLength(2)
    expect(remote.audioCalls).toEqual(['h1', 'n1'])
  })

  it('a clip that never ends is released at its reported duration plus the margin, well before the cap', async () => {
    // Pins the loadedmetadata wiring: the duration must reach the chain.
    playerScripts.push('hang-with-duration', 'end')
    remote.latestValue = null
    mount()
    await advance(0)
    remote.latestValue = utterance('d1')
    await advance(SPEECH_POLL_MS * 2)
    expect(constructedAudio).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advance(SPEECH_POLL_MS * 2)
    expect(remote.audioCalls).toEqual(['d1', 'n1'])
    expect(constructedAudio).toHaveLength(1)
    // d1 started at 0.7 s with a 2 s duration, so it is released at 0.7 + 2 + 5 = 7.7 s.
    await advance(4_800)
    expect(constructedAudio).toHaveLength(1)
    await advance(200)
    expect(constructedAudio).toHaveLength(2)
  })

  it('a browser reply longer than the cap is held for its estimate plus the margin, so the next reply does not speak over it', async () => {
    class FakeUtterance {
      text: string
      rate = 1
      pitch = 1
      voice: unknown = undefined
      onend: (() => void) | null = null
      onerror: (() => void) | null = null
      constructor(text: string) {
        this.text = text
      }
    }
    const spoken: FakeUtterance[] = []
    vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance)
    vi.stubGlobal('speechSynthesis', {
      getVoices: () => [],
      speak: (utterance: FakeUtterance) => {
        spoken.push(utterance)
      },
    })
    remote.latestValue = null
    mount()
    await advance(0)
    // 1500 characters at rate 1 are estimated at 150 s. The browser end never fires.
    remote.latestValue = { ...utterance('b1'), engine: 'browser', text: 'a'.repeat(1500) }
    await advance(SPEECH_POLL_MS)
    expect(spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advance(SPEECH_POLL_MS)
    expect(remote.audioCalls).toEqual(['n1'])
    expect(constructedAudio).toHaveLength(0)
    // Past the 120 s cap the browser reply still holds the next reply.
    await advance(PLAYBACK_CAP_MS)
    expect(constructedAudio).toHaveLength(0)
    await advance(30_000)
    expect(constructedAudio).toHaveLength(0)
    // Released at the estimate plus the 5 s margin: 0.7 s + 155 s = 155.7 s.
    await advance(5_000)
    expect(constructedAudio).toHaveLength(1)
  })
})
