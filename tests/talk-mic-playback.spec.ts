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
  record: { enabled: true, hotkey: null, autoSubmit: false, vad: { silenceMs: 1000, energyThreshold: 0.01 } },
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
/** The browser speech flag as each audio element was constructed; true means speech was still audible. */
const speakingAtAudioStart: boolean[] = []

/**
 * What each successive player does: fire ended, fire pause, fire abort, or fire
 * nothing (a stalled element). Consumed in construction order; default 'end'.
 */
type PlayerScript = 'end' | 'pause' | 'abort' | 'hang' | 'hang-with-duration' | 'reject'
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
    speakingAtAudioStart.push(speech.speaking)
    this.script = playerScripts.shift() ?? 'end'
  }
  addEventListener(type: string, handler: () => void): void {
    this.handlers.set(type, [...(this.handlers.get(type) ?? []), handler])
  }
  play(): Promise<void> {
    // A refused start rejects play(), as Safari does when auto-play is blocked.
    if (this.script === 'reject') return Promise.reject(Object.assign(new Error('not allowed'), { name: 'NotAllowedError' }))
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

function mount(settings: TalkStatus = SETTINGS): void {
  const props = {
    status: async () => settings,
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

/** Fake milliseconds since the test's mount; advance keeps it current. */
let clock = 0

async function advance(ms: number): Promise<void> {
  clock += ms
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

/** Advance until the fake clock reads `ms`. */
async function advanceTo(ms: number): Promise<void> {
  await advance(ms - clock)
}

/** The utterance object the button passes to the stubbed `speechSynthesis.speak`. */
class FakeUtterance {
  text: string
  rate = 1
  pitch = 1
  voice: unknown = undefined
  onstart: (() => void) | null = null
  onend: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor(text: string) {
    this.text = text
  }
}

/**
 * The stubbed browser speech engine. `speaking` is what `speechSynthesis.speaking`
 * reads. When `speaksOnSpeak` is true, speak() sets `speaking` true, as a real
 * engine does; a test then sets it false itself, or leaves it true to show that
 * only the utterance's own end can release the chain.
 */
interface FakeSpeech {
  speaking: boolean
  speaksOnSpeak: boolean
  spoken: FakeUtterance[]
  /** How many times the button read `speechSynthesis.speaking`. */
  reads: number
  /** How many times the button called `speechSynthesis.cancel()`. A cancel clears the flag, as the engine does. */
  cancels: number
  /** Calls to speak() and cancel() in the order they happened: `speak:<text>` or `cancel`. */
  order: string[]
}
let speech: FakeSpeech = { speaking: false, speaksOnSpeak: false, spoken: [], reads: 0, cancels: 0, order: [] }

function stubSpeech(speaksOnSpeak: boolean): void {
  speech = { speaking: false, speaksOnSpeak, spoken: [], reads: 0, cancels: 0, order: [] }
  vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance)
  vi.stubGlobal('speechSynthesis', {
    getVoices: () => [],
    speak: (utterance: FakeUtterance) => {
      speech.spoken.push(utterance)
      speech.order.push(`speak:${utterance.text}`)
      if (speech.speaksOnSpeak) speech.speaking = true
    },
    cancel: () => {
      speech.cancels += 1
      speech.order.push('cancel')
      speech.speaking = false
    },
    get speaking(): boolean {
      speech.reads += 1
      return speech.speaking
    },
  })
}

/**
 * Stop the speaking poll from ticking, as a throttled or hidden tab does. Its
 * interval is created but its callback never runs, so the flag is not read and
 * only the chain's bound can release the reply. Call it before the reply starts.
 * The mount's own poll was created earlier, so it keeps running.
 */
function stallSpeakingPoll(): void {
  const realSetInterval = globalThis.setInterval
  vi.stubGlobal('setInterval', (_handler: () => void, ms: number) => realSetInterval(() => {}, ms))
}

/** A browser-engine reply of `characters` characters, as talk/latest reports it. */
function browserReply(utteranceId: string, characters: number): TalkLatest {
  return { ...utterance(utteranceId), engine: 'browser', text: 'a'.repeat(characters) }
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
    speakingAtAudioStart.length = 0
    playerScripts.length = 0
    clock = 0
    stubSpeech(false)
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

  it('a browser reply with no end signal is held for its ceiling plus the margin, so the next reply does not speak over it', async () => {
    // 1500 characters at rate 1: ceiling 450 s, bound 455 s. Neither end nor
    // error fires, and speaking never turns true, so only the ceiling releases.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 1500)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(SPEECH_POLL_MS * 2)
    expect(remote.audioCalls).toEqual(['n1'])
    expect(constructedAudio).toHaveLength(0)
    // Past the 120 s cap the browser reply still holds the next reply.
    await advanceTo(PLAYBACK_CAP_MS)
    expect(constructedAudio).toHaveLength(0)
    // Released at 0.7 s + 455 s = 455.7 s.
    await advanceTo(455_000)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(456_000)
    expect(constructedAudio).toHaveLength(1)
  })

  it('end releases the chain: the next reply plays about 3 s in, not at the ceiling', async () => {
    // speaking stays true, so only the utterance's own end can release the chain.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(2_999)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(3_000)
    speech.spoken[0]?.onend?.()
    await advanceTo(3_500)
    expect(constructedAudio).toHaveLength(1)
    expect(remote.audioCalls).toEqual(['n1'])
    // The speaking poll stops with the reply: no further reads after the end.
    const readsAtEnd = speech.reads
    await advanceTo(5_000)
    expect(speech.reads).toBe(readsAtEnd)
  })

  it('slow speech does not overlap: 1500 characters with end at 300 s hold the next reply until 300 s', async () => {
    // Reviewer probe P-F1a. 1500 characters at 5 per second take 300 s; the
    // previous estimate (155 s bound) released the next reply at about 155.7 s.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 1500)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(299_900)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(300_000)
    speech.speaking = false
    speech.spoken[0]?.onend?.()
    await advanceTo(301_000)
    expect(constructedAudio).toHaveLength(1)
  })

  it('speaking going false releases the chain when end never fires: the next reply plays by 3.5 s', async () => {
    // Pins the speaking poll on its own: onend is never fired.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(2_999)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(3_000)
    speech.speaking = false
    await advanceTo(3_500)
    expect(constructedAudio).toHaveLength(1)
  })

  it('a wedged engine is released at the 125 s floor: 200 characters, no end, no error, speaking never true', async () => {
    // ceiling max(120 s, 200 * 300 ms) = 120 s, bound 125 s, started at 0.7 s: released at 125.7 s.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(125_000)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(125_800)
    expect(constructedAudio).toHaveLength(1)
  })

  it('a wedged engine is clamped at 605 s: 20000 characters, no end, no error, speaking never true', async () => {
    // ceiling min(600 s, 20000 * 300 ms) = 600 s, bound 605 s, started at 0.7 s: released at 605.7 s.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 20_000)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(605_000)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(605_800)
    expect(constructedAudio).toHaveLength(1)
  })

  it('a long reply at rate 1 is not cut off at 605 s: 2500 characters, speech runs to 625 s, the next reply waits for its end', async () => {
    // Ceiling min(600 s, 2500 * 300 ms) = 600 s. The flag stays true, so each true read re-arms the bound.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 2500)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(625_000)
    expect(constructedAudio).toHaveLength(0)
    speech.speaking = false
    speech.spoken[0]?.onend?.()
    await advanceTo(626_000)
    expect(constructedAudio).toHaveLength(1)
  })

  it('a long reply at rate 0.5 is not cut off at 605 s: 1500 characters, speech runs to 750 s', async () => {
    // Ceiling min(600 s, 1500 * 300 ms / 0.5) = 600 s. The reply speaks past the first bound and is still held.
    stubSpeech(true)
    remote.latestValue = null
    mount({ ...SETTINGS, tts: { ...SETTINGS.tts, browser: { voiceName: null, rate: 0.5, pitch: 1 } } })
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 1500)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(749_000)
    expect(constructedAudio).toHaveLength(0)
    speech.speaking = false
    speech.spoken[0]?.onend?.()
    await advanceTo(751_000)
    expect(constructedAudio).toHaveLength(1)
  })

  it('the bound moves forward on each true read: a reply that speaks past its 125 s bound still holds the next reply', async () => {
    // 200 characters: ceiling 120 s, bound 125 s. Without the re-arm the next reply would start at 125.8 s.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(125_800)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(199_000)
    expect(constructedAudio).toHaveLength(0)
    speech.speaking = false
    speech.spoken[0]?.onend?.()
    await advanceTo(200_500)
    expect(constructedAudio).toHaveLength(1)
  })

  it('a flag stuck at true is released by the 900 s absolute cap, and the speaking poll stops with it', async () => {
    // The cap is armed at speak (0.7 s), so it fires at 900.7 s; the flag never reads false.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(900_000)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(901_000)
    expect(constructedAudio).toHaveLength(1)
    const readsAtRelease = speech.reads
    await advanceTo(1_000_000)
    expect(speech.reads).toBe(readsAtRelease)
  })

  it('after the chain releases a browser reply, the speaking flag is no longer read (interval leak, R4-4)', async () => {
    // 200 characters, no end, speaking never true: released at 125.7 s. The first tick after
    // release returns before it reads the flag, so the read count must not grow after that.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(125_800)
    expect(constructedAudio).toHaveLength(1)
    const readsAtRelease = speech.reads
    await advanceTo(400_000)
    expect(speech.reads).toBe(readsAtRelease)
  })

  it('a gap in the flag between two queued replies does not release the second (R4-3; assumes the engine reports the flag false between utterances, not measured)', async () => {
    // b0 fires onstart and ends through its own end signal, so no cap is involved. That sets the
    // engine's start latch, and b1 is spoken as soon as b0 ends.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b0', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.speaking = true
    speech.spoken[0]?.onstart?.()
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS * 2)
    remote.latestValue = utterance('n1')
    await advanceTo(2_000)
    speech.speaking = false
    speech.spoken[0]?.onend?.()
    await advanceTo(2_100)
    expect(speech.spoken).toHaveLength(2)
    expect(constructedAudio).toHaveLength(0)
    // The gap: the flag reads false before b1 starts. b1 has not started, so it is not released.
    await advanceTo(2_500)
    expect(constructedAudio).toHaveLength(0)
    // b1 starts, its flag goes false after its own start, and only then is it released.
    speech.speaking = true
    speech.spoken[1]?.onstart?.()
    await advanceTo(3_000)
    speech.speaking = false
    await advanceTo(3_500)
    expect(constructedAudio).toHaveLength(1)
  })

  it('an error while the utterance is still queued releases the reply at the error (R4-5)', async () => {
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(2_000)
    // The flag never turns true: the utterance is queued behind other speech.
    speech.spoken[0]?.onerror?.()
    await advanceTo(2_500)
    expect(constructedAudio).toHaveLength(1)
  })

  it('an unreadable speaking flag does not end the reply early: it holds until the 125 s bound (R4-6)', async () => {
    stubSpeech(false)
    vi.stubGlobal('speechSynthesis', {
      getVoices: () => [],
      speak: (utterance: FakeUtterance) => {
        speech.spoken.push(utterance)
      },
      get speaking(): boolean {
        throw new Error('unreadable')
      },
    })
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(120_000)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(125_800)
    expect(constructedAudio).toHaveLength(1)
  })

  it('onstart ties the flag to this reply: a false flag before its onstart does not release it, and after onstart one false read does', async () => {
    // The flag reads true, then false, before the engine fires onstart for this reply. Before onstart,
    // the false read must not release it (a fallback false read cannot count yet: it is due at 2.7 s).
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.speaking = true
    remote.latestValue = utterance('n1')
    await advanceTo(1_000)
    speech.speaking = false
    await advanceTo(1_500)
    expect(constructedAudio).toHaveLength(0)
    speech.speaking = true
    speech.spoken[0]?.onstart?.()
    await advanceTo(2_000)
    speech.speaking = false
    await advanceTo(2_500)
    expect(constructedAudio).toHaveLength(1)
  })

  it('an engine that never fires onstart: the flag seen true, then false on two reads after the 2 s fallback, releases the reply', async () => {
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.speaking = true
    remote.latestValue = utterance('n1')
    await advanceTo(1_000)
    speech.speaking = false
    // The fallback is 2 s after speak (0.7 s), so it is due at 2.7 s. The speaking reads at 2.7 s and
    // 2.95 s are the two consecutive false reads, 250 ms apart, so the reply is released at 2.95 s.
    await advanceTo(2_500)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(2_900)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(3_200)
    expect(constructedAudio).toHaveLength(1)
  })

  it('an engine that never fires onstart: one false read after the fallback does not release the reply', async () => {
    // Speaking polls run at 0.95 s, 1.2 s, ... 2.95 s, 3.2 s. The 2 s fallback is due at 2.7 s,
    // so the read at 2.95 s is the first false read that counts. The flag reads true again at 3.2 s.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.speaking = true
    remote.latestValue = utterance('n1')
    await advanceTo(2_900)
    speech.speaking = false
    await advanceTo(2_960)
    speech.speaking = true
    await advanceTo(3_500)
    expect(constructedAudio).toHaveLength(0)
  })

  it('the start latch holds (P6-5): after an earlier onstart, a reply that never fires onstart is not released by two false reads after its fallback', async () => {
    // b0 fires onstart, which sets the engine's latch. b1 never fires onstart. Its flag is seen true, then reads
    // false twice after its 2 s fallback. With the latch, the fallback is off, so b1 is held to its bound.
    // Removing the `!engine.firesStart` guard (mutant M6e) releases b1 before the 5 s check, and this test fails.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b0', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.speaking = true
    speech.spoken[0]?.onstart?.()
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(2_000)
    speech.speaking = false
    speech.spoken[0]?.onend?.()
    await advanceTo(2_100)
    expect(speech.spoken).toHaveLength(2)
    speech.speaking = true
    remote.latestValue = utterance('n1')
    await advanceTo(4_500)
    speech.speaking = false
    await advanceTo(5_000)
    expect(constructedAudio).toHaveLength(0)
  })

  it('the cap does not cancel a reply whose onstart never fired, even while the flag reads true: the next clip starts over it (named limit 1)', async () => {
    // The flag stays true, and this engine never fires onstart for b1. The cap (armed at the 0.7 s speak)
    // releases b1 at 900.7 s. Its speech is unattributable: thisSpeaking is unknown, so no cancel runs,
    // and the next clip starts while the flag still reads true. This is the accepted overlap, asserted here.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    remote.latestValue = utterance('n1')
    await advanceTo(900_000)
    expect(constructedAudio).toHaveLength(0)
    expect(speech.cancels).toBe(0)
    await advanceTo(901_000)
    expect(speech.cancels).toBe(0)
    expect(constructedAudio).toHaveLength(1)
    expect(speakingAtAudioStart).toEqual([true])
    await advanceTo(1_000_000)
    expect(speech.cancels).toBe(0)
  })

  it('the cap cancels a reply whose onstart fired: cancel is called once, and the next reply starts only after it, not over it', async () => {
    // onstart fires for b1 at the 0.7 s speak, so thisSpeaking is true. The 900 s cap releases b1 while it is audible, and
    // the cancel runs first.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.spoken[0]?.onstart?.()
    remote.latestValue = utterance('n1')
    await advanceTo(900_000)
    expect(constructedAudio).toHaveLength(0)
    expect(speech.cancels).toBe(0)
    await advanceTo(901_000)
    expect(speech.cancels).toBe(1)
    expect(constructedAudio).toHaveLength(1)
    // The clip began after the cancel cleared the flag, so no audio played over the speech.
    expect(speakingAtAudioStart).toEqual([false])
    await advanceTo(1_000_000)
    expect(speech.cancels).toBe(1)
  })

  it('the cap does not cancel when the flag already reads false: the reply is released without cancel', async () => {
    // b0 fires onstart and ends, which sets the engine's start latch. b1 never fires onstart. Its flag
    // reads true until 800 s, then false. Its bound is re-armed by the last true read to about 925 s,
    // so the cap at about 901.4 s releases b1 while the flag reads false.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b0', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.speaking = true
    speech.spoken[0]?.onstart?.()
    speech.speaking = false
    speech.spoken[0]?.onend?.()
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS * 2)
    expect(speech.spoken).toHaveLength(2)
    speech.speaking = true
    remote.latestValue = utterance('n1')
    await advanceTo(800_000)
    expect(constructedAudio).toHaveLength(0)
    speech.speaking = false
    await advanceTo(900_000)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(902_000)
    expect(constructedAudio).toHaveLength(1)
    expect(speech.cancels).toBe(0)
  })

  it('after the chain releases a browser reply, the speaking poll and its timers are all cleared (R5-5: finish must clear the interval)', async () => {
    // Only this probe catches an interval that finish leaves running: the released branch returns before it reads the flag.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    const baseline = vi.getTimerCount()
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    await advanceTo(126_000)
    expect(vi.getTimerCount()).toBe(baseline)
  })

  it('the bound releases a reply whose flag still reads true: cancel is called once, at the bound, and the next clip starts after it', async () => {
    // The speaking poll is stalled, so the flag is never read. The 200-character reply is bounded at 125.7 s
    // (125 s ceiling plus margin, armed at its 0.7 s start) while the engine still speaks. onstart fires for b1,
    // so thisSpeaking is true and only the release hook can cancel it.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    stallSpeakingPoll()
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.spoken[0]?.onstart?.()
    remote.latestValue = utterance('n1')
    await advanceTo(125_700 - 1)
    expect(speech.cancels).toBe(0)
    expect(constructedAudio).toHaveLength(0)
    await advanceTo(125_700)
    expect(speech.cancels).toBe(1)
    await advanceTo(126_100)
    expect(constructedAudio).toHaveLength(1)
    // The clip began after the cancel cleared the flag, so no audio played over the speech.
    expect(speakingAtAudioStart).toEqual([false])
    expect(speech.cancels).toBe(1)
  })

  it('the reply ends by itself while the flag reads false: the bound never fires and nothing is cancelled', async () => {
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.spoken[0]?.onstart?.()
    speech.spoken[0]?.onend?.()
    remote.latestValue = utterance('n1')
    await advanceTo(SPEECH_POLL_MS * 3)
    expect(constructedAudio).toHaveLength(1)
    // The bound would have released at 125.7 s; advance past it.
    await advanceTo(130_000)
    expect(speech.cancels).toBe(0)
  })

  it('the reply ends by itself while another utterance keeps the flag true: the bound cancels nothing, so that speech is not cut', async () => {
    // The flag stays true after this reply's end, as it would while another talk button's utterance is audible.
    // onstart fires first, so thisSpeaking is true, and onend then clears it as the rule says. This test does not
    // separately observe that clear: once a task settles the chain drops its release handlers and finish() clears
    // the cap, so no cancel path can run after onend. Mutant M8d (onend keeps thisSpeaking) is therefore equivalent.
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    speech.spoken[0]?.onstart?.()
    speech.spoken[0]?.onend?.()
    await advanceTo(130_000)
    expect(speech.cancels).toBe(0)
  })

  it('the cancel on a bound release comes before the next reply is spoken: the order is speak, cancel, speak', async () => {
    stubSpeech(true)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    stallSpeakingPoll()
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    speech.spoken[0]?.onstart?.()
    remote.latestValue = { ...browserReply('n1', 10), text: 'next reply' }
    await advanceTo(125_700)
    expect(speech.order).toEqual([`speak:${'a'.repeat(200)}`, 'cancel', 'speak:next reply'])
  })

  it('the bound releases a reply whose flag reads false: the release hook does not cancel', async () => {
    // Stalled poll and a flag that never reads true: the guard in the hook is what keeps cancel from running.
    stubSpeech(false)
    remote.latestValue = null
    mount()
    await advanceTo(0)
    stallSpeakingPoll()
    remote.latestValue = browserReply('b1', 200)
    await advanceTo(SPEECH_POLL_MS)
    expect(speech.spoken).toHaveLength(1)
    await advanceTo(125_700 + SPEECH_POLL_MS)
    expect(speech.cancels).toBe(0)
  })

  it('a start the browser refuses is logged once, not retried, and releases the chain for the next reply', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const refusals = (): unknown[][] => warn.mock.calls.filter((call) => String(call[0]).includes('did not start'))
    mount()
    await advance(0)
    playerScripts.push('reject', 'end')
    remote.latestValue = utterance('blocked')
    await advance(SPEECH_POLL_MS)
    expect(refusals()).toHaveLength(1)
    expect(String(refusals()[0]?.[0])).toContain('blocked')
    await advance(SPEECH_POLL_MS * 3)
    expect(remote.audioCalls).toEqual(['blocked'])
    // Positive control: the refused reply released the chain, so the next reply plays.
    remote.latestValue = utterance('next')
    await advance(SPEECH_POLL_MS * 2)
    expect(remote.audioCalls).toEqual(['blocked', 'next'])
    expect(constructedAudio).toHaveLength(2)
    expect(refusals()).toHaveLength(1)
    warn.mockRestore()
  })
})
