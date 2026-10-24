/**
 * Speech delivery by polling. The browser asks the host for the newest
 * utterance of ONE session (`talk/latest`) on a fixed interval and plays each
 * utterance id it has not seen. The host writes the utterance cache before it
 * attempts the session-log append, so the audio survives even when the
 * `dsh-talk/speech` event never reaches the log (it does not on a stock host).
 *
 * The controller takes its remote calls and its visibility hooks as
 * parameters, so it runs without a DOM: the component supplies the real
 * `talk/*` bindings and `document`, and the spec supplies stubs.
 *
 * @module dsh-talk/client/speechPoll
 */

import type { TalkAudio, TalkLatest } from '../wire.ts'

/**
 * Milliseconds between polls. 700 ms keeps the wait for a reply to about one
 * interval after the host has cached it, and each poll is one small JSON call.
 */
export const SPEECH_POLL_MS = 700

/**
 * Longest a `talk/latest` read may take before it counts as failed. Without
 * this bound one read that never settles would hold the in-flight flag and
 * stop polling for the rest of the mount.
 */
export const READ_TIMEOUT_MS = 5_000

/**
 * Longest one playback (audio or browser voice) may hold the chain when its
 * duration is unknown. A player that never fires ended, error, pause or abort
 * must not silence every later reply.
 */
export const PLAYBACK_CAP_MS = 120_000

/** Slack added to a known audio duration before the playback wait is released. */
export const PLAYBACK_MARGIN_MS = 5_000

/**
 * Browser-voice ceiling per character, in milliseconds, at rate 1. Deliberately
 * slower than real speech (about 4 to 5 characters per second for Chinese, so
 * about 200 to 250 ms per character): the ceiling only fires for an engine that
 * never ends its speech, and the speaking poll ends a normal reply.
 */
export const CEILING_MS_PER_CHAR = 300

/** Longest browser-voice ceiling, in milliseconds: the hard clamp on a wedged engine's hold. */
export const PLAYBACK_CEILING_MAX_MS = 600_000

/**
 * Milliseconds between reads of `speechSynthesis.speaking` while a browser
 * reply is spoken. The reply ends on the first read that is false after a read
 * that was true.
 */
export const SPEAKING_POLL_MS = 250

/** How many utterance ids are remembered for the play-once rule (oldest dropped first). */
const SEEN_CAP = 64

/** The remote surface and page hooks the controller needs. */
export interface SpeechPollDeps {
  /**
   * Read the newest utterance of one session; rejects on transport failure.
   * `signal` aborts when the read passes {@link READ_TIMEOUT_MS}; an
   * implementation that cannot cancel its request may ignore it.
   */
  latest(sessionId: string, signal: AbortSignal): Promise<TalkLatest | null>
  /** Fetch the cached audio of one utterance; null when evicted or never cached. */
  audio(utteranceId: string): Promise<TalkAudio | null>
  /**
   * Speak text with the browser voice; resolves when the speech ends or fails.
   * Calls onDuration with an upper bound in seconds before speaking, so the
   * chain is bounded even when the engine never reports the end. The bound is
   * not the expected length: the speech itself normally ends the wait first.
   */
  speakText(text: string, onDuration: (seconds: number) => void): Promise<void>
  /**
   * Play decoded audio; resolves when playback ends, fails, pauses or aborts.
   * Calls onDuration with the length in seconds once the player knows it.
   */
  playAudio(entry: TalkAudio, onDuration: (seconds: number) => void): Promise<void>
  /** True while the page is hidden. */
  isHidden(): boolean
  /** Subscribe to visibility changes; returns the unsubscribe function. */
  onVisibilityChange(listener: () => void): () => void
}

/** What the chain hands a playback task. */
interface PlaybackContext {
  /**
   * Reports an upper bound in seconds; the chain's bound becomes that value plus
   * {@link PLAYBACK_MARGIN_MS}. The bound releases the chain only when the task
   * has not settled first, so a reported value that is too large costs time on a
   * stalled engine, and one that is too small lets speech overlap the next reply.
   */
  onDuration(seconds: number): void
  /** True once the chain has released this task, because its bound elapsed or it finished. */
  released(): boolean
}

/**
 * Resolve when `task` settles or when its bound elapses, whichever is first.
 * The bound is {@link PLAYBACK_CAP_MS} until the task reports a duration, then
 * the duration plus {@link PLAYBACK_MARGIN_MS}.
 *
 * @param task - the playback or speech to wait for; it receives the chain context.
 */
function boundedPlayback(task: (ctx: PlaybackContext) => Promise<void>): Promise<void> {
  return new Promise<void>((resolve) => {
    let settled = false
    // Set when the chain releases this task. A late reply is dropped rather
    // than played over the next one: two replies at once is the failure a
    // listener hears, so a late reply after a stall is the lesser cost.
    let released = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const done = (): void => {
      if (settled) return
      settled = true
      released = true
      if (timer !== undefined) clearTimeout(timer)
      resolve()
    }
    const arm = (ms: number): void => {
      if (settled) return
      if (timer !== undefined) clearTimeout(timer)
      timer = setTimeout(done, ms)
    }
    arm(PLAYBACK_CAP_MS)
    const ctx: PlaybackContext = {
      onDuration: (seconds) => {
        if (Number.isFinite(seconds) && seconds > 0) arm(seconds * 1000 + PLAYBACK_MARGIN_MS)
      },
      released: () => released,
    }
    let pending: Promise<void>
    try {
      pending = task(ctx)
    } catch {
      pending = Promise.resolve()
    }
    pending.then(done, done)
  })
}

/**
 * Run a read and reject if it has not settled within `ms`. On the timeout the
 * signal passed to `start` is aborted. The late result, if any, is ignored.
 *
 * @param start - starts the read with its signal; a synchronous throw counts as a rejection.
 * @param ms - the bound in milliseconds.
 */
function withTimeout<T>(start: (signal: AbortSignal) => Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const controller = new AbortController()
    const timer = setTimeout(() => {
      controller.abort()
      reject(new Error('talk/latest read timed out'))
    }, ms)
    Promise.resolve()
      .then(() => start(controller.signal))
      .then(
        (value) => {
          clearTimeout(timer)
          resolve(value)
        },
        (error: unknown) => {
          clearTimeout(timer)
          reject(error)
        },
      )
  })
}

/**
 * Start polling for one session's utterances.
 *
 * The first successful read after mount is a seed: its utterance id is marked
 * seen and nothing is played, so a page reload does not replay the last reply.
 * Each later utterance id is requested through `audio` (or spoken, for the
 * browser engine) exactly once. All sound goes through one chain, so replies
 * play in arrival order and never overlap. Polling pauses while the page is
 * hidden, and a read that resolves while hidden is dropped. When the page
 * becomes visible again, the next read plays any reply that is not yet seen,
 * so a reply that landed while hidden is played once on return.
 *
 * @param sessionId - the session whose utterances to follow.
 * @param deps - remote calls and page hooks.
 * @param intervalMs - poll interval; defaults to {@link SPEECH_POLL_MS}.
 * @returns a function that stops polling, removes the visibility listener, and
 *   keeps any audio that has not yet started from being handed to the player.
 */
export function startSpeechPoll(sessionId: string, deps: SpeechPollDeps, intervalMs: number = SPEECH_POLL_MS): () => void {
  let stopped = false
  let seeded = false
  let inFlight = false
  let timer: ReturnType<typeof setInterval> | undefined
  const seen = new Set<string>()
  let playing: Promise<void> = Promise.resolve()

  const remember = (utteranceId: string): void => {
    seen.add(utteranceId)
    while (seen.size > SEEN_CAP) {
      const oldest = seen.values().next().value
      if (oldest === undefined) break
      seen.delete(oldest)
    }
  }

  // One chain for everything that makes sound, audio and browser voice alike,
  // so two replies never speak at once. A task whose turn comes after stop()
  // does not run.
  const enqueue = (task: (ctx: PlaybackContext) => Promise<void>): void => {
    playing = playing
      .then(() => (stopped ? undefined : boundedPlayback(task)))
      .catch(() => undefined)
  }

  // The audio request starts now, while the cache still holds the entry; the
  // playback waits its turn behind earlier replies.
  const enqueueAudio = (utteranceId: string): void => {
    const fetched = (async () => deps.audio(utteranceId))().catch(() => null)
    enqueue(async (ctx) => {
      const entry = await fetched
      // Audio that resolves after stop() is never handed to the player.
      if (entry === null || stopped) return
      // Audio that resolves after the chain released this task is dropped.
      if (ctx.released()) return
      await deps.playAudio(entry, ctx.onDuration)
    })
  }

  const tick = async (): Promise<void> => {
    if (stopped || inFlight) return
    inFlight = true
    let latest: TalkLatest | null = null
    let failed = false
    try {
      latest = await withTimeout((signal) => deps.latest(sessionId, signal), READ_TIMEOUT_MS)
    } catch {
      failed = true
    }
    inFlight = false
    // A read that resolves while the page is hidden is dropped, not played:
    // the first read after the page is visible again picks the reply up.
    if (failed || stopped || deps.isHidden()) return
    if (!seeded) {
      // Seed: the first successful read after mount marks its reply seen and
      // plays nothing, so a reload never replays. A mount while hidden seeds on
      // the first read after the page is visible, so the reply present at mount
      // is not played either. Replies that arrive after mount play once.
      seeded = true
      if (latest !== null) remember(latest.utteranceId)
      return
    }
    if (latest === null || seen.has(latest.utteranceId)) return
    remember(latest.utteranceId)
    if (latest.engine === 'browser') {
      const text = latest.text
      enqueue((ctx) => deps.speakText(text, ctx.onDuration))
      return
    }
    enqueueAudio(latest.utteranceId)
  }

  const halt = (): void => {
    if (timer !== undefined) {
      clearInterval(timer)
      timer = undefined
    }
  }

  const run = (): void => {
    timer = setInterval(() => {
      void tick()
    }, intervalMs)
    void tick()
  }

  const onVisibility = (): void => {
    if (stopped) return
    if (deps.isHidden()) {
      halt()
      return
    }
    if (timer !== undefined) return
    // No reseed here: the seen set still holds the last reply, so the next
    // read plays only a reply that landed while hidden.
    run()
  }

  const unsubscribe = deps.onVisibilityChange(onVisibility)
  if (!deps.isHidden()) run()

  return () => {
    stopped = true
    halt()
    unsubscribe()
  }
}
