/**
 * The `talk/latest` read carries the poll's abort signal to the request. The
 * poll already moves on after READ_TIMEOUT_MS; these specs check that the
 * request itself is cancelled. The stub host behaves like `fetch`: it rejects
 * when its signal aborts, and it hangs forever when no signal reaches it.
 * Time is virtual (fake timers), so no DOM or host is needed.
 *
 * @module dsh-talk/test/talk-latest-abort.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { apply } from '../src/client/index.ts'
import type { TalkMicInjected } from '../src/client/TalkMicButton.tsx'
import {
  READ_TIMEOUT_MS,
  SPEECH_POLL_MS,
  startSpeechPoll,
  type SpeechPollDeps,
} from '../src/client/speechPoll.ts'
import { TALK_INVOCATIONS, TALK_LATEST_DESCRIPTOR } from '../src/wire.ts'

const SESSION = 'session-abort'

/** One `talk/latest` request as the stub host received it. */
interface Request {
  signal: AbortSignal | undefined
  settled: boolean
}

/**
 * A stub host. `hang` never answers and never sees a signal it could cancel;
 * `cancellable` answers nothing, but rejects when its signal aborts (as `fetch`
 * does); `answer` returns at once with no utterance.
 */
function stubHost(mode: 'hang' | 'cancellable' | 'answer') {
  const requests: Request[] = []
  const talkLatest = (_sessionId: string, signal?: AbortSignal) => {
    const request: Request = { signal, settled: false }
    requests.push(request)
    if (mode === 'answer') {
      request.settled = true
      return Promise.resolve({ ok: true as const, value: null })
    }
    return new Promise<never>((_resolve, reject) => {
      if (signal === undefined) return
      signal.addEventListener('abort', () => {
        request.settled = true
        reject(signal.reason)
      }, { once: true })
    })
  }
  return {
    requests,
    talkLatest,
    started: () => requests.length,
    settled: () => requests.filter((request) => request.settled).length,
  }
}

/**
 * Mount the browser plugin against the stub host and return the injected
 * `latest` the speech poll calls, so the chain under test is the real one.
 */
async function clientLatest(talkLatest: ReturnType<typeof stubHost>['talkLatest']) {
  let injected: TalkMicInjected | undefined
  const slots = {
    inject: (_slot: string, factory: () => unknown) => factory(),
    register: (options: { id: string; inject?: () => unknown }) => {
      if (options.id === 'talk-mic') injected = options.inject?.() as TalkMicInjected
      return options
    },
  }
  const ctx = {
    effect: () => undefined,
    locale: { register: () => undefined },
    remote: { $mount: async () => undefined },
    get: (key: string) => (key === 'slots' ? slots : undefined),
    inject: (_keys: string[], factory: (scope: unknown) => void) => factory({
      remote: { talk: { latest: talkLatest } },
      locale: { bind: () => (key: string) => key },
      get: () => undefined,
    }),
  }
  await apply(ctx as unknown as Parameters<typeof apply>[0])
  if (injected === undefined) throw new Error('talk-mic slot was not registered')
  return injected.latest
}

/** Poll deps whose only live surface is the `latest` read under test. */
function pollDeps(latest: SpeechPollDeps['latest']): SpeechPollDeps {
  return {
    latest,
    audio: async () => null,
    speakText: async () => undefined,
    playAudio: async () => undefined,
    isHidden: () => false,
    onVisibilityChange: () => () => undefined,
  }
}

describe('talk/latest carries the abort signal to the request', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('declares the signal as cancellation on the wire descriptor, never as a wire field', () => {
    expect(TALK_LATEST_DESCRIPTOR.cancellation).toEqual({ parameter: 'signal' })
    expect(TALK_LATEST_DESCRIPTOR.parameters.map((parameter) => parameter.wire)).toEqual(['sessionId'])
    expect(TALK_INVOCATIONS).toContain(TALK_LATEST_DESCRIPTOR)
  })

  it('a read past READ_TIMEOUT_MS has the signal the host received aborted, not merely ignored', async () => {
    const host = stubHost('cancellable')
    const stop = startSpeechPoll(SESSION, pollDeps(await clientLatest(host.talkLatest)))
    await vi.advanceTimersByTimeAsync(0)
    expect(host.started()).toBe(1)
    const received = host.requests[0]?.signal
    expect(received).toBeDefined()
    expect(received?.aborted).toBe(false)
    await vi.advanceTimersByTimeAsync(READ_TIMEOUT_MS - 1)
    expect(received?.aborted).toBe(false)
    await vi.advanceTimersByTimeAsync(1)
    expect(received?.aborted).toBe(true)
    expect(host.requests[0]?.settled).toBe(true)
    stop()
  })

  it('after timeouts against a hung host, never more than one read is in flight', async () => {
    const host = stubHost('cancellable')
    const stop = startSpeechPoll(SESSION, pollDeps(await clientLatest(host.talkLatest)))
    // Five timeouts plus one poll interval: the accumulation the ticket names would show here.
    const horizon = 5 * (READ_TIMEOUT_MS + SPEECH_POLL_MS) + SPEECH_POLL_MS
    for (let elapsed = 0; elapsed < horizon; elapsed += 100) {
      await vi.advanceTimersByTimeAsync(100)
      expect(host.started() - host.settled()).toBeLessThanOrEqual(1)
    }
    expect(host.started()).toBeGreaterThan(1)
    stop()
  })

  it('a read that completes normally is never aborted, so the fix does not abort everything', async () => {
    const host = stubHost('answer')
    const stop = startSpeechPoll(SESSION, pollDeps(await clientLatest(host.talkLatest)))
    await vi.advanceTimersByTimeAsync(0)
    expect(host.started()).toBe(1)
    expect(host.requests[0]?.signal).toBeDefined()
    await vi.advanceTimersByTimeAsync(READ_TIMEOUT_MS * 2)
    expect(host.started()).toBeGreaterThan(1)
    expect(host.requests.every((request) => request.signal?.aborted === false)).toBe(true)
    stop()
  })
})
