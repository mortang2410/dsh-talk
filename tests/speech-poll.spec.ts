/**
 * The browser's `talk/latest` poll: each new utterance id is played once, in
 * arrival order, and the first successful read after mount is a seed that
 * plays nothing. Polling pauses while the page is hidden; a reply that lands
 * meanwhile plays once on return, and returning to a page whose last reply is
 * still current replays nothing. An evicted entry is silent. A read that never
 * settles times out and has its signal aborted, a player that never settles is
 * released by its bound, and a late audio fetch is dropped. The remote surface
 * (`latest`, `audio`) and the page hooks are stubs, and time is virtual (fake
 * timers), so no DOM or host is needed.
 *
 * @module dsh-talk/test/speech-poll.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  PLAYBACK_CAP_MS,
  PLAYBACK_MARGIN_MS,
  READ_TIMEOUT_MS,
  SPEECH_POLL_MS,
  startSpeechPoll,
  type SpeechPollDeps,
} from '../src/client/speechPoll.ts'
import type { TalkAudio, TalkLatest } from '../src/wire.ts'

const SESSION = 'session-under-test'

function utterance(utteranceId: string, engine: TalkLatest['engine'] = 'edge-tts', text = `spoken ${utteranceId}`): TalkLatest {
  return {
    sessionId: SESSION,
    utteranceId,
    engine,
    text,
    audioBytes: engine === 'browser' ? 0 : 3,
    reason: 'speak-tool',
    logged: true,
    appended: 1,
    skipped: 0,
  }
}

function audioFor(utteranceId: string): TalkAudio {
  return { utteranceId, mime: 'audio/mpeg', data: 'AAAA', text: `spoken ${utteranceId}`, engine: 'edge-tts' }
}

/** Stub remote surface and page hooks, with recorders for every call. */
function stubRemote() {
  const state: { latest: TalkLatest | null; failReads: number } = { latest: null, failReads: 0 }
  const evicted = new Set<string>()
  const latestSessions: string[] = []
  const audioRequests: string[] = []
  // Every audio entry handed to the player (the controller never hands it null).
  const played: string[] = []
  const spoken: string[] = []
  let hidden = false
  const listeners = new Set<() => void>()
  const deps: SpeechPollDeps = {
    latest: async (sessionId) => {
      latestSessions.push(sessionId)
      if (state.failReads > 0) {
        state.failReads -= 1
        throw new Error('transport down')
      }
      return state.latest
    },
    audio: async (utteranceId) => {
      audioRequests.push(utteranceId)
      return evicted.has(utteranceId) ? null : audioFor(utteranceId)
    },
    speakText: async (text) => {
      spoken.push(text)
    },
    playAudio: async (entry) => {
      played.push(entry.utteranceId)
    },
    isHidden: () => hidden,
    onVisibilityChange: (listener) => {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
  }
  return {
    state,
    evicted,
    deps,
    latestSessions,
    audioRequests,
    played,
    spoken,
    listenerCount: () => listeners.size,
    setHidden(value: boolean) {
      hidden = value
      for (const listener of listeners) listener()
    },
  }
}

describe('startSpeechPoll (talk/latest poll)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('polls every SPEECH_POLL_MS (700 ms) for the session it was started with', async () => {
    const remote = stubRemote()
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.latestSessions).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS - 1)
    expect(remote.latestSessions).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(remote.latestSessions).toHaveLength(2)
    expect(new Set(remote.latestSessions)).toEqual(new Set([SESSION]))
    expect(SPEECH_POLL_MS).toBe(700)
    stop()
  })

  it('seeds at mount: the utterance cached before the page loaded is not played', async () => {
    const remote = stubRemote()
    remote.state.latest = utterance('before-reload')
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 3)
    expect(remote.audioRequests).toEqual([])
    expect(remote.played).toEqual([])
    stop()
  })

  it('requests audio once per new utterance id and never again for an id already seen', async () => {
    const remote = stubRemote()
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('u1')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.audioRequests).toEqual(['u1'])
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 5)
    expect(remote.audioRequests).toEqual(['u1'])
    expect(remote.played).toEqual(['u1'])
    stop()
  })

  it('two utterances in a row each play once, in order, and the first is not replayed when the second arrives', async () => {
    const remote = stubRemote()
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('u1')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    remote.state.latest = utterance('u2')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 3)
    expect(remote.audioRequests).toEqual(['u1', 'u2'])
    expect(remote.played).toEqual(['u1', 'u2'])
    stop()
  })

  it('a burst: two replies inside one interval means only the newest is requested', async () => {
    // talk/latest keeps one utterance per session (service.ts:302-311), so
    // 'first' is overwritten before any poll can read it. This documents that
    // host limit; the client cannot recover 'first' without a wire change.
    const remote = stubRemote()
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('first')
    remote.state.latest = utterance('second')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.audioRequests).toEqual(['second'])
    expect(remote.played).toEqual(['second'])
    stop()
  })

  it('plays in arrival order even when a later audio response arrives first', async () => {
    const remote = stubRemote()
    let releaseFirst: ((entry: TalkAudio) => void) | undefined
    const deps: SpeechPollDeps = {
      ...remote.deps,
      audio: (utteranceId) => {
        remote.audioRequests.push(utteranceId)
        return utteranceId === 'slow'
          ? new Promise<TalkAudio>((resolve) => {
            releaseFirst = resolve
          })
          : Promise.resolve(audioFor(utteranceId))
      },
    }
    const stop = startSpeechPoll(SESSION, deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('slow')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    remote.state.latest = utterance('fast')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 2)
    expect(remote.audioRequests).toEqual(['slow', 'fast'])
    expect(remote.played).toEqual([])
    releaseFirst?.(audioFor('slow'))
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.played).toEqual(['slow', 'fast'])
    stop()
  })

  it('an evicted utterance (audio null) is silent: no console error, no warning, no retry, and the next reply still plays', async () => {
    const remote = stubRemote()
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    remote.evicted.add('gone')
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('gone')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 4)
    expect(remote.played).toEqual([])
    expect(remote.audioRequests).toEqual(['gone'])
    expect(errorSpy).not.toHaveBeenCalled()
    expect(warnSpy).not.toHaveBeenCalled()
    remote.state.latest = utterance('next')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual(['next'])
    stop()
  })

  it('a browser-engine utterance is spoken from its text and requests no audio', async () => {
    const remote = stubRemote()
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('b1', 'browser', 'hello there')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.spoken).toEqual(['hello there'])
    expect(remote.audioRequests).toEqual([])
    expect(remote.played).toEqual([])
    stop()
  })

  it('a failed read plays nothing, seeds nothing, and polling continues', async () => {
    const remote = stubRemote()
    remote.state.latest = utterance('old')
    remote.state.failReads = 1
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.played).toEqual([])
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual([])
    remote.state.latest = utterance('new')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual(['new'])
    stop()
  })

  it('a reply that lands while the page is hidden plays once on return, and nothing is read or played while hidden', async () => {
    const remote = stubRemote()
    remote.state.latest = utterance('before')
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.setHidden(true)
    const callsWhenHidden = remote.latestSessions.length
    remote.state.latest = utterance('while-hidden')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 5)
    expect(remote.latestSessions.length).toBe(callsWhenHidden)
    expect(remote.played).toEqual([])
    remote.setHidden(false)
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.played).toEqual(['while-hidden'])
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 3)
    expect(remote.played).toEqual(['while-hidden'])
    stop()
  })

  it('returning to a page whose last reply is still current does not replay it', async () => {
    // Replay-on-return guard. The seen set must survive a hide: a same-id
    // return must not request or play the reply again. The seed rule does not
    // reach this case (a reseed on return would also be silent), so this test
    // does not catch a reversed seed rule; the "while hidden" tests do.
    const remote = stubRemote()
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('last')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual(['last'])
    remote.setHidden(true)
    remote.setHidden(false)
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 3)
    expect(remote.audioRequests).toEqual(['last'])
    expect(remote.played).toEqual(['last'])
    stop()
  })

  it('mount while hidden seeds the reply present at mount: nothing plays on return, and a reply after return plays once', async () => {
    const remote = stubRemote()
    remote.state.latest = utterance('at-mount')
    remote.setHidden(true)
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 3)
    expect(remote.latestSessions).toEqual([])
    remote.setHidden(false)
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.audioRequests).toEqual([])
    expect(remote.played).toEqual([])
    remote.state.latest = utterance('after-return')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 3)
    expect(remote.audioRequests).toEqual(['after-return'])
    expect(remote.played).toEqual(['after-return'])
    stop()
  })

  it('a read still in flight when the page hides is dropped, and its reply plays once on return', async () => {
    const remote = stubRemote()
    let calls = 0
    let release: ((value: TalkLatest | null) => void) | undefined
    const deps: SpeechPollDeps = {
      ...remote.deps,
      latest: (sessionId, signal) => {
        calls += 1
        if (calls === 2) {
          return new Promise<TalkLatest | null>((resolve) => {
            release = resolve
          })
        }
        return remote.deps.latest(sessionId, signal)
      },
    }
    const stop = startSpeechPoll(SESSION, deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('in-flight-while-hidden')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(calls).toBe(2)
    remote.setHidden(true)
    release?.(utterance('in-flight-while-hidden'))
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.played).toEqual([])
    remote.setHidden(false)
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.played).toEqual(['in-flight-while-hidden'])
    stop()
  })

  it('stop() ends polling and removes the visibility listener', async () => {
    const remote = stubRemote()
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.listenerCount()).toBe(1)
    expect(vi.getTimerCount()).toBe(1)
    stop()
    expect(remote.listenerCount()).toBe(0)
    // The interval itself is cleared, not merely ignored by the stopped flag.
    expect(vi.getTimerCount()).toBe(0)
    const callsAtStop = remote.latestSessions.length
    remote.state.latest = utterance('after-unmount')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 5)
    expect(remote.latestSessions.length).toBe(callsAtStop)
    expect(remote.played).toEqual([])
  })

  it('stop() with audio still being fetched: the audio that resolves afterwards is never handed to the player', async () => {
    const remote = stubRemote()
    const held = new Map<string, (entry: TalkAudio) => void>()
    const deps: SpeechPollDeps = {
      ...remote.deps,
      audio: (utteranceId) => {
        remote.audioRequests.push(utteranceId)
        return new Promise<TalkAudio>((resolve) => {
          held.set(utteranceId, resolve)
        })
      },
    }
    const stop = startSpeechPoll(SESSION, deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('in-flight')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.audioRequests).toEqual(['in-flight'])
    stop()
    const callsAtStop = remote.latestSessions.length
    held.get('in-flight')?.(audioFor('in-flight'))
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 3)
    expect(remote.played).toEqual([])
    expect(remote.latestSessions.length).toBe(callsAtStop)
  })

  it('a read that never settles times out, and polling resumes', async () => {
    const remote = stubRemote()
    let calls = 0
    const deps: SpeechPollDeps = {
      ...remote.deps,
      latest: (sessionId, signal) => {
        calls += 1
        if (calls === 1) return new Promise<TalkLatest | null>(() => {})
        return remote.deps.latest(sessionId, signal)
      },
    }
    const stop = startSpeechPoll(SESSION, deps)
    await vi.advanceTimersByTimeAsync(0)
    expect(calls).toBe(1)
    await vi.advanceTimersByTimeAsync(READ_TIMEOUT_MS + SPEECH_POLL_MS * 3)
    expect(calls).toBeGreaterThan(2)
    remote.state.latest = utterance('after-hang')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 2)
    expect(remote.played).toEqual(['after-hang'])
    stop()
  })

  it('a player that fires neither ended, error nor pause does not block the next reply: the 120 s cap releases it', async () => {
    const remote = stubRemote()
    let firstPlayer = true
    const deps: SpeechPollDeps = {
      ...remote.deps,
      playAudio: (entry) => {
        remote.played.push(entry.utteranceId)
        if (firstPlayer) {
          firstPlayer = false
          return new Promise<void>(() => {})
        }
        return Promise.resolve()
      },
    }
    const stop = startSpeechPoll(SESSION, deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('u1')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual(['u1'])
    remote.state.latest = utterance('u2')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 3)
    expect(remote.audioRequests).toEqual(['u1', 'u2'])
    expect(remote.played).toEqual(['u1'])
    await vi.advanceTimersByTimeAsync(PLAYBACK_CAP_MS)
    expect(remote.played).toEqual(['u1', 'u2'])
    stop()
  })

  it('a known audio duration sets the wait to duration plus margin, not the 120 s cap', async () => {
    const remote = stubRemote()
    let firstPlayer = true
    const deps: SpeechPollDeps = {
      ...remote.deps,
      playAudio: (entry, onDuration) => {
        remote.played.push(entry.utteranceId)
        if (firstPlayer) {
          firstPlayer = false
          onDuration(2)
          return new Promise<void>(() => {})
        }
        return Promise.resolve()
      },
    }
    const stop = startSpeechPoll(SESSION, deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('u1')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    remote.state.latest = utterance('u2')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual(['u1'])
    await vi.advanceTimersByTimeAsync(2000 + PLAYBACK_MARGIN_MS)
    expect(remote.played).toEqual(['u1', 'u2'])
    stop()
  })

  it('a browser-engine reply waits for audio still playing, so two things never speak at once', async () => {
    const remote = stubRemote()
    let releaseAudio: (() => void) | undefined
    const deps: SpeechPollDeps = {
      ...remote.deps,
      playAudio: (entry) => {
        remote.played.push(entry.utteranceId)
        return new Promise<void>((resolve) => {
          releaseAudio = resolve
        })
      },
    }
    const stop = startSpeechPoll(SESSION, deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('u1')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual(['u1'])
    remote.state.latest = utterance('b1', 'browser', 'hello there')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 2)
    expect(remote.spoken).toEqual([])
    releaseAudio?.()
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.spoken).toEqual(['hello there'])
    stop()
  })

  it('an audio fetch that resolves after its bound released the chain is dropped, not played under the next reply', async () => {
    const remote = stubRemote()
    const held = new Map<string, (entry: TalkAudio) => void>()
    const deps: SpeechPollDeps = {
      ...remote.deps,
      audio: (utteranceId) => {
        remote.audioRequests.push(utteranceId)
        if (utteranceId === 'u2') {
          return new Promise<TalkAudio>((resolve) => {
            held.set(utteranceId, resolve)
          })
        }
        return Promise.resolve(audioFor(utteranceId))
      },
      playAudio: (entry) => {
        remote.played.push(entry.utteranceId)
        if (entry.utteranceId === 'u1') return new Promise<void>(() => {})
        return Promise.resolve()
      },
    }
    const stop = startSpeechPoll(SESSION, deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.state.latest = utterance('u1')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual(['u1'])
    remote.state.latest = utterance('u2')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    remote.state.latest = utterance('u3')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.audioRequests).toEqual(['u1', 'u2', 'u3'])
    // u1 hangs and releases at the cap. u2 starts then, is bounded by its own
    // cap, and its fetch stays held, so u3 waits behind it.
    await vi.advanceTimersByTimeAsync(PLAYBACK_CAP_MS)
    expect(remote.played).toEqual(['u1'])
    expect(held.has('u2')).toBe(true)
    // u2's bound releases at about 240 s, and u3 plays then.
    await vi.advanceTimersByTimeAsync(PLAYBACK_CAP_MS)
    expect(remote.played).toEqual(['u1', 'u3'])
    // u2's fetch resolves after the chain released it: the late audio is dropped.
    held.get('u2')?.(audioFor('u2'))
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual(['u1', 'u3'])
    stop()
  })

  it('a read past READ_TIMEOUT_MS has its signal aborted, so the poll can drop the request', async () => {
    const remote = stubRemote()
    const signals: AbortSignal[] = []
    const deps: SpeechPollDeps = {
      ...remote.deps,
      latest: (_sessionId, signal) => {
        signals.push(signal)
        return new Promise<TalkLatest | null>(() => {})
      },
    }
    const stop = startSpeechPoll(SESSION, deps)
    await vi.advanceTimersByTimeAsync(0)
    expect(signals).toHaveLength(1)
    expect(signals[0]?.aborted).toBe(false)
    await vi.advanceTimersByTimeAsync(READ_TIMEOUT_MS - 1)
    expect(signals[0]?.aborted).toBe(false)
    await vi.advanceTimersByTimeAsync(1)
    expect(signals[0]?.aborted).toBe(true)
    stop()
  })
})
