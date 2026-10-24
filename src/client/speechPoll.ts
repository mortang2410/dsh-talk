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

/** How many utterance ids are remembered for the play-once rule (oldest dropped first). */
const SEEN_CAP = 64

/** The remote surface and page hooks the controller needs. */
export interface SpeechPollDeps {
  /** Read the newest utterance of one session; rejects on transport failure. */
  latest(sessionId: string): Promise<TalkLatest | null>
  /** Fetch the cached audio of one utterance; null when evicted or never cached. */
  audio(utteranceId: string): Promise<TalkAudio | null>
  /** Speak text with the browser voice (browser engine: no audio is cached). */
  speakText(text: string): void
  /** Play decoded audio; resolves when playback ends or fails. */
  playAudio(entry: TalkAudio): Promise<void>
  /** True while the page is hidden. */
  isHidden(): boolean
  /** Subscribe to visibility changes; returns the unsubscribe function. */
  onVisibilityChange(listener: () => void): () => void
}

/**
 * Start polling for one session's utterances.
 *
 * The first successful read is a seed: its utterance id is marked seen and
 * nothing is played, so a page reload does not replay the last reply. Each
 * later utterance id is requested through `audio` (or spoken, for the browser
 * engine) exactly once. Audio plays in arrival order. Polling pauses while
 * the page is hidden; on becoming visible again the next read is a new seed,
 * so utterances that landed while hidden are skipped rather than replayed.
 *
 * @param sessionId - the session whose utterances to follow.
 * @param deps - remote calls and page hooks.
 * @param intervalMs - poll interval; defaults to {@link SPEECH_POLL_MS}.
 * @returns a function that stops polling and removes the visibility listener.
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

  // The audio request starts now, while the cache still holds the entry; the
  // playback itself waits its turn behind any earlier utterance.
  const enqueueAudio = (utteranceId: string): void => {
    const fetched = (async () => deps.audio(utteranceId))().catch(() => null)
    playing = playing
      .then(() => fetched)
      .then(entry => (entry === null ? undefined : deps.playAudio(entry)))
      .catch(() => undefined)
  }

  const tick = async (): Promise<void> => {
    if (stopped || inFlight) return
    inFlight = true
    let latest: TalkLatest | null = null
    let failed = false
    try {
      latest = await deps.latest(sessionId)
    } catch {
      failed = true
    }
    inFlight = false
    if (failed || stopped) return
    if (!seeded) {
      seeded = true
      if (latest !== null) remember(latest.utteranceId)
      return
    }
    if (latest === null || seen.has(latest.utteranceId)) return
    remember(latest.utteranceId)
    if (latest.engine === 'browser') {
      try {
        deps.speakText(latest.text)
      } catch {
        // The browser voice is unavailable; the utterance is marked seen anyway.
      }
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
    seeded = false
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
