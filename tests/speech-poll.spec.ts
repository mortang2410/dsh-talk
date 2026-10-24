/**
 * The browser's `talk/latest` poll: each new utterance id is played once, in
 * arrival order; the first read after mount or after the page becomes visible
 * is a seed and plays nothing; an evicted entry is silent; polling pauses while
 * the page is hidden. The remote surface (`latest`, `audio`) and the page hooks
 * are stubs, and time is virtual (fake timers), so no DOM or host is needed.
 *
 * @module dsh-talk/test/speech-poll.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SPEECH_POLL_MS, startSpeechPoll, type SpeechPollDeps } from '../src/client/speechPoll.ts'
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
  const played: string[] = []
  // Every value handed to the player, including a null that must never arrive.
  const handed: Array<string | null> = []
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
    speakText: (text) => {
      spoken.push(text)
    },
    playAudio: async (entry) => {
      handed.push(entry === null ? null : entry.utteranceId)
      if (entry !== null) played.push(entry.utteranceId)
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
    handed,
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
    expect(remote.handed).toEqual(['next'])
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

  it('pauses while the page is hidden and resumes without replaying what was missed', async () => {
    const remote = stubRemote()
    remote.state.latest = utterance('before')
    const stop = startSpeechPoll(SESSION, remote.deps)
    await vi.advanceTimersByTimeAsync(0)
    remote.setHidden(true)
    const callsWhenHidden = remote.latestSessions.length
    remote.state.latest = utterance('missed-while-hidden')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS * 5)
    expect(remote.latestSessions.length).toBe(callsWhenHidden)
    remote.setHidden(false)
    await vi.advanceTimersByTimeAsync(0)
    expect(remote.played).toEqual([])
    remote.state.latest = utterance('after-visible')
    await vi.advanceTimersByTimeAsync(SPEECH_POLL_MS)
    expect(remote.played).toEqual(['after-visible'])
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
})
