/**
 * The composer microphone button: when the host resolved the `web` STT engine,
 * transcribe with the browser's Web Speech API (interim results included);
 * otherwise record with MediaRecorder, convert the recording to 16 kHz mono WAV
 * (wav.ts), and send it through the host's `talk/transcribe` to the local engine.
 * Either way the result fills the draft (or sends directly when auto-submit is
 * on), and play each new utterance of
 * this session by polling `talk/latest` (browser voice for the browser engine,
 * cached audio through `talk/audio` for the local engines; see `speechPoll.ts`).
 * Pressing record also interrupts host playback (speak-to-interrupt).
 *
 * @module dsh-talk/client/TalkMicButton
 */

import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { InjectFace, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import type {} from '../projection.ts'
import { foldMic, initMic, planBargeIn, planStopPress, stopControlVisible, type MicViewModel } from './present.ts'
import { advanceSilence, CAPTURE_SILENCE_TIMEOUT_MS, isSpeechFrame, rmsOf, silenceTrips } from './vad.ts'
import { selectBrowserVoice } from './browserVoice.ts'
import {
  CEILING_MS_PER_CHAR,
  ENGINE_START_FALLBACK_MS,
  PLAYBACK_ABSOLUTE_MAX_MS,
  PLAYBACK_CAP_MS,
  PLAYBACK_CEILING_MAX_MS,
  SPEAKING_POLL_MS,
  startSpeechPoll,
} from './speechPoll.ts'
import { recordingToWav16kMono } from './wav.ts'
import type { TalkAudio, TalkHeal, TalkInterruptResult, TalkLatest, TalkStatus, TalkTranscript } from '../wire.ts'

/** Registration-side injected face: the recorder's host/browser bindings. */
export interface TalkMicInjected {
  /** Read the effective settings snapshot once. */
  status: () => Promise<TalkStatus>
  /** Stop in-flight host synthesis. */
  interrupt: () => Promise<TalkInterruptResult>
  /** Host transcription for the local STT engines. */
  transcribe: (audioData: string) => Promise<TalkTranscript>
  /** Heal the accumulated raw transcript on the host; the host makes the one model call. */
  heal: (rawTranscript: string) => Promise<TalkHeal>
  /** Serve one cached utterance's audio. */
  audio: (utteranceId: string) => Promise<TalkAudio | null>
  /** Read the newest utterance of one session (playback poll). */
  latest: (sessionId: string, signal?: AbortSignal) => Promise<TalkLatest | null>
  /** Write the transcription into the composer draft. */
  setDraft: (text: string) => void
  /** Send the transcription as a user message. */
  send: (text: string) => Promise<void>
  /**
   * Resolve the current master input facade for one session id through
   * `sessions.scope(id)` + `conversation.input.for(actx)`; undefined when
   * either service is unavailable (rc.8 builds without the scope exchange).
   */
  forSession?: (id: unknown) => { setDraft(text: string): void; submit(): void } | undefined
  /**
   * Watch whether one session has a running turn. The listener gets true while
   * the turn runs and false once it ends, and is called at once with the current
   * value. Returns the unsubscribe. Absent when the host lacks the turn services.
   */
  watchTurn?: (sessionId: string, listener: (running: boolean) => void) => () => void
  /**
   * Cancel the running turn of one session through the scoped conversation
   * service (the same call the built-in stop button makes). Preserves the queue.
   */
  cancelTurn?: (sessionId: string) => Promise<void>
  /** The stop control's localized `title` and `aria-label`, resolved by the plugin's locale dictionary. */
  stopLabel?: string
}

/** How one playing reply is stopped. `begin` registers it; the returned function ends it. */
export type PlaybackBegin = (stop: () => void) => () => void

/** Full component props assembled by the input-slot renderer. */
export type TalkMicProps = PropsRuntime<'conversation.input.left'> & InjectFace<TalkMicInjected>

/** Structural face of the standard-kit input actions current master feeds every session-scope slot component. */
interface RuntimeInputActions {
  setDraft(text: string): void
  submit(): void
}

/** Minimal structural face of the Web Speech API (Chrome/Edge). */
interface SpeechRecognitionLike {
  lang: string
  interimResults: boolean
  maxAlternatives: number
  continuous: boolean
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>>; resultIndex: number }) => void) | null
  onend: (() => void) | null
  onerror: ((event: { error?: string }) => void) | null
  start: () => void
  stop: () => void
  abort: () => void
}

function webRecognition(): SpeechRecognitionLike | undefined {
  const scope = globalThis as unknown as {
    SpeechRecognition?: new () => SpeechRecognitionLike
    webkitSpeechRecognition?: new () => SpeechRecognitionLike
  }
  const Ctor = scope.SpeechRecognition ?? scope.webkitSpeechRecognition
  if (Ctor === undefined) return undefined
  try {
    return new Ctor()
  } catch {
    return undefined
  }
}

/** Base64 of raw bytes, for the `talk/transcribe` wire field. */
function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000))
  }
  return btoa(binary)
}

/** A failure's message, whitespace collapsed and capped so it fits the composer. */
function detailOf(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error)
  return raw.replace(/\s+/gu, ' ').trim().slice(0, 200)
}

/**
 * Detector tick, in milliseconds: the cadence of the voice-activity reads in a segmented capture.
 *
 * Named limit (documented, not fixed): segment onset clips up to one tick. A segment starts at
 * the first tick that reads speech, with no pre-roll, so up to 500 ms at the start of each
 * utterance can be missing from its recording.
 */
const TICK_MS = 500

/**
 * One segmented capture, from the press until its final text is placed. The microphone
 * stays open; each speech run becomes its own MediaRecorder segment, transcribed in order.
 */
interface SegmentedCapture {
  stream: MediaStream
  mime: string
  /** The recorder of the segment being captured, or null between segments. */
  recorder: MediaRecorder | null
  /** Segments started so far; the 120-second timeout runs only once this is above zero. */
  segments: number
  /** When the capture started (Date.now), for the hard ceiling and the elapsed display. */
  startedAt: number
  /** The raw transcripts of every segment so far, joined with single spaces. */
  accumulated: string
  /** Silence, in milliseconds, inside the current segment (ends it at vad.silenceMs). */
  pauseMs: number
  /** Silence, in milliseconds, since the last speech frame (the 120-second timeout). */
  idleMs: number
  /**
   * Segments whose `onstop` has not fired yet. It rises once when a segment starts and falls
   * once, in that segment's own `onstop`, so it cannot go negative. A browser-ended segment
   * (the recorder stopped by itself) falls the same way as one this code stopped.
   */
  segmentsOpen: number
  /** Set once the capture is ending (trigger, timeout, stop or ceiling). Nothing new is detected or asked of heal. */
  closing: boolean
  /** Set on unmount. Nothing from this capture is transcribed into the draft or submitted after it. */
  abandoned: boolean
  /** Set once the final step has been queued, so it runs once. */
  finalized: boolean
  /** Set when the heal call decided the trigger and the prompt was submitted. */
  done: boolean
  /** Serial chain: every segment and the final step run here, one after another. */
  queue: Promise<void>
  timer: ReturnType<typeof setInterval> | null
}

function supportedMime(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined
  const candidates = ['audio/wav', 'audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
  for (const candidate of candidates) {
    try {
      if (MediaRecorder.isTypeSupported(candidate)) return candidate
    } catch {
      // isTypeSupported can throw on hostile environments; try the next.
    }
  }
  return undefined
}

/** What one mount has learned about the browser engine: whether it fires `onstart`. */
interface BrowserEngineState {
  firesStart: boolean
}

/**
 * Read `speechSynthesis.speaking`.
 *
 * @returns the flag, or null when the engine cannot be read.
 */
function readSpeaking(): boolean | null {
  try {
    return speechSynthesis.speaking === true
  } catch {
    return null
  }
}

/**
 * Speak text with the browser voice. Settings come from the `talk/status`
 * snapshot (`tts.browser`): `talk/latest` carries no per-call voice, rate or
 * pitch, so a speak-tool `voice` override is not applied on this path.
 *
 * The reply ends on the first of four signals: the utterance's `end` or
 * `error`, the absolute cap (PLAYBACK_ABSOLUTE_MAX_MS), or
 * `speechSynthesis.speaking` going false. The flag is page-wide, so it counts
 * only after this utterance's `onstart`. For an engine that never fires
 * `onstart`, it counts once ENGINE_START_FALLBACK_MS has passed since `speak()`
 * and the flag has been seen true, and then only on two consecutive false
 * reads, SPEAKING_POLL_MS apart. The poll stops once `released` reports that
 * the chain has moved on.
 *
 * A release for any reason cancels this reply's speech, and only this reply's.
 * That covers the absolute cap (PLAYBACK_ABSOLUTE_MAX_MS) and the chain's bound,
 * which releases the reply when the speaking poll has stopped ticking, as in a
 * throttled or hidden tab. The flag `speechSynthesis.speaking` is page-wide, so
 * it cannot say whose speech is audible. The cancel therefore runs only when
 * `thisSpeaking` is true: set by this utterance's `onstart`, cleared by its
 * `onend` and `onerror`. When `thisSpeaking` is false, nothing is cancelled. When
 * it is still unknown because the engine never fired `onstart`, nothing is
 * cancelled either. The release handler runs before the chain starts the next
 * reply, so the cancel never cuts a later reply. `speechSynthesis.cancel()`
 * clears the page's whole speech queue, so on a page with more than one mounted
 * talk button it can also stop another button's utterance that is queued.
 *
 * Named limits, accepted in round 8 and not fixed:
 * 1. An engine that never fires `onstart`: no cancel is made, so a reply that
 *    runs past its bound can overlap the next clip.
 * 2. An engine that fires `onstart` only for its first utterance: later replies
 *    are held off the two-read fallback and wait for their bound instead of
 *    releasing on a false read. The bound is the length-based ceiling plus
 *    PLAYBACK_MARGIN_MS (5 s): 125 s for a 200-character reply, longer for a longer one.
 *    This is a hold, not an overlap.
 * 3. A reply longer than PLAYBACK_ABSOLUTE_MAX_MS (900 s) is cut at the cap, so
 *    its tail is never spoken and the listener is not told.
 * 4. An engine that fires `onstart` but never `onend` or `onerror` leaves
 *    `thisSpeaking` true for good. If other speech then keeps the page flag true,
 *    the cap can still cancel that other speech, because the flag cannot say
 *    whose it is. There is no in-page fix for this: the reply's other
 *    per-utterance events (`boundary`, `mark`, `pause`, `resume`) show activity
 *    but never report an end, and the flag cannot say whose speech it is.
 * 5. A reply that is released while still queued behind other speech, before its
 *    own `onstart`, is never cancelled, so its utterance can play late.
 *
 * Before speaking, `onDuration` receives an inactivity ceiling in seconds:
 * `min(PLAYBACK_CEILING_MAX_MS, max(PLAYBACK_CAP_MS, text.length *
 * CEILING_MS_PER_CHAR / rate))`. Every poll read that is true passes it again,
 * so the bound fires only when the flag stops reporting speech for that long.
 *
 * @param text - the sanitized utterance text from `talk/latest`.
 * @param delivery - the browser delivery settings, when the snapshot has loaded.
 * @param onDuration - receives the inactivity ceiling in seconds, now and on each true read.
 * @param released - true once the chain has moved on; the poll stops then.
 * @param onRelease - registers the handler that cancels audible speech when the chain's bound releases the reply.
 * @param engine - this mount's record of whether the engine fires `onstart`.
 * @returns a promise that resolves when the speech ends or fails, so the poller
 *   can keep it in order with the audio replies.
 */
function speakWithBrowserVoice(
  text: string,
  delivery: TalkStatus['tts']['browser'] | undefined,
  onDuration: (seconds: number) => void,
  released: () => boolean,
  onRelease: (handler: () => void) => void,
  engine: BrowserEngineState,
  begin: PlaybackBegin,
): Promise<void> {
  if (typeof speechSynthesis === 'undefined') {
    console.warn('[dsh-talk] speechSynthesis unavailable')
    return Promise.resolve()
  }
  return new Promise<void>((resolve) => {
    const utterance = new SpeechSynthesisUtterance(text)
    const voice = selectBrowserVoice(speechSynthesis.getVoices(), delivery?.voiceName ?? undefined)
    if (voice !== undefined) utterance.voice = voice
    if (delivery !== undefined) {
      utterance.rate = delivery.rate
      utterance.pitch = delivery.pitch
    }
    const rate = Math.max(0.1, delivery?.rate ?? 1)
    const ceilingMs = Math.min(
      PLAYBACK_CEILING_MAX_MS,
      Math.max(PLAYBACK_CAP_MS, (text.length * CEILING_MS_PER_CHAR) / rate),
    )
    const ceilingSeconds = ceilingMs / 1000
    onDuration(ceilingSeconds)
    let finished = false
    // Ends this reply's entry in the playback registry; set just before speak().
    let endPlayback: () => void = () => {}
    let started = false
    // This utterance's own state: true from its onstart, false after its onend or onerror,
    // undefined while the engine has not fired onstart for it. Only true allows a cancel.
    let thisSpeaking: boolean | undefined
    let fallbackDue = false
    let seenSpeaking = false
    // Consecutive false reads counted on the never-onstart fallback; a true read resets it.
    let falseReads = 0
    let timer: ReturnType<typeof setInterval> | undefined
    let startFallback: ReturnType<typeof setTimeout> | undefined
    let absoluteCap: ReturnType<typeof setTimeout> | undefined
    // Every path ends here, so the poll and both timers are always cleared.
    const finish = (): void => {
      if (finished) return
      finished = true
      endPlayback()
      if (timer !== undefined) clearInterval(timer)
      if (startFallback !== undefined) clearTimeout(startFallback)
      if (absoluteCap !== undefined) clearTimeout(absoluteCap)
      resolve()
    }
    // Cancel this reply's speech, and only when this utterance is known to be speaking.
    // A page-wide flag that reads true may be another reply's speech, so it is not enough.
    const cancelOwnSpeech = (): void => {
      if (thisSpeaking !== true || readSpeaking() !== true) return
      try {
        speechSynthesis.cancel()
      } catch {
        // The engine refused the cancel; the reply still releases.
      }
    }
    onRelease(() => {
      cancelOwnSpeech()
      // Clear the poll and the cap now. Otherwise the cap could fire later and
      // cancel a reply that started after this one.
      finish()
    })
    // The stop control: cancel this reply's own audible speech (if known) and end the task.
    const stopNow = (): void => {
      cancelOwnSpeech()
      finish()
    }
    utterance.onstart = () => {
      started = true
      thisSpeaking = true
      engine.firesStart = true
    }
    utterance.onend = () => {
      thisSpeaking = false
      finish()
    }
    utterance.onerror = () => {
      thisSpeaking = false
      finish()
    }
    endPlayback = begin(stopNow)
    speechSynthesis.speak(utterance)
    // The utterance may have ended synchronously; never start a poll after that.
    if (finished) return
    startFallback = setTimeout(() => {
      fallbackDue = true
    }, ENGINE_START_FALLBACK_MS)
    absoluteCap = setTimeout(() => {
      // The chain releases this reply now. Cancel only this reply's own speech, then release.
      cancelOwnSpeech()
      finish()
    }, PLAYBACK_ABSOLUTE_MAX_MS)
    timer = setInterval(() => {
      // The chain has moved on (its bound elapsed): stop reading the flag.
      if (released()) {
        finish()
        return
      }
      const speaking = readSpeaking()
      // An unreadable flag is not a finished reply: wait for the next read, the ceiling or the cap.
      if (speaking === null) return
      if (speaking) {
        seenSpeaking = true
        falseReads = 0
        // Re-arm the chain's bound: the ceiling is an inactivity timeout, not a deadline.
        onDuration(ceilingSeconds)
        return
      }
      if (started) {
        finish()
        return
      }
      // The fallback applies only to an engine that has never fired `onstart`, and needs two false reads in a row.
      if (seenSpeaking && fallbackDue && !engine.firesStart) {
        falseReads += 1
        if (falseReads >= 2) finish()
      }
    }, SPEAKING_POLL_MS)
  })
}

/**
 * Play one cached audio entry. Resolves when playback ends, fails, pauses or
 * aborts, so the poller can keep utterances in arrival order. The poller also
 * bounds the wait, so a player that fires none of these still releases the
 * chain. Playback failure is silent by design: the utterance is already marked
 * seen and must not be retried.
 *
 * @param entry - the audio served by `talk/audio`.
 * @param onDuration - receives the audio length in seconds once the player knows it.
 */
function playAudioEntry(
  entry: TalkAudio,
  onDuration: (seconds: number) => void,
  onRelease: (handler: () => void) => void,
  begin: PlaybackBegin,
): Promise<void> {
  return new Promise<void>((resolve) => {
    const binary = atob(entry.data)
    const bytes = new Uint8Array(binary.length)
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
    const url = URL.createObjectURL(new Blob([bytes], { type: entry.mime }))
    const player = new Audio(url)
    let finished = false
    // Ends this reply's entry in the playback registry; set just before play().
    let endPlayback: () => void = () => {}
    const finish = (): void => {
      if (finished) return
      finished = true
      endPlayback()
      URL.revokeObjectURL(url)
      resolve()
    }
    // Stop this reply now: pause the element and end the task. A later pause event finds it finished.
    const stopNow = (): void => {
      try {
        player.pause()
      } catch {
        // A player that cannot pause is still ended below.
      }
      finish()
    }
    // The chain's bound released this playback: stop it so it cannot overlap the next reply.
    // Intended: a bound release pauses audio still playing, so the next reply never overlaps it.
    onRelease(stopNow)
    player.addEventListener('ended', finish)
    player.addEventListener('error', finish)
    player.addEventListener('pause', finish)
    player.addEventListener('abort', finish)
    player.addEventListener('loadedmetadata', () => onDuration(player.duration))
    endPlayback = begin(stopNow)
    void Promise.resolve(player.play()).catch(finish)
  })
}

/**
 * The record button itself. All browser capability reads are feature-detected
 * and degrade to a disabled button with no throw.
 *
 * @param props - injected bindings plus the framework standard kit.
 * @returns the button element.
 */
export function TalkMicButton(props: TalkMicProps): ReactNode {
  const { status, interrupt, transcribe, heal, audio, latest, setDraft, send, forSession, watchTurn, cancelTurn, stopLabel = '' } = props
  const inputActions = (props as Partial<Record<'inputActions', RuntimeInputActions>>).inputActions
  const sessionIdProp = (props as Partial<Record<'sessionId', unknown>>).sessionId
  const writeDraft = (text: string): void => {
    if (inputActions !== undefined) {
      inputActions.setDraft(text)
      return
    }
    const facade = sessionIdProp !== undefined ? forSession?.(sessionIdProp) : undefined
    if (facade !== undefined) {
      facade.setDraft(text)
      return
    }
    console.warn('[dsh-talk] falling back to rc.8 injected setDraft')
    setDraft(text)
  }
  const submitText = async (text: string): Promise<void> => {
    if (inputActions !== undefined) {
      inputActions.setDraft(text)
      inputActions.submit()
      return
    }
    const facade = sessionIdProp !== undefined ? forSession?.(sessionIdProp) : undefined
    if (facade !== undefined) {
      facade.setDraft(text)
      facade.submit()
      return
    }
    await send(text)
  }
  const [mic, setMic] = useState<MicViewModel>(initMic)
  const [recording, setRecording] = useState(false)
  const [settings, setSettings] = useState<TalkStatus | undefined>(undefined)
  // Stop control state. The refs are read by the detector and the press handler; the state drives the render.
  const [playing, setPlaying] = useState(false)
  const [turnRunning, setTurnRunning] = useState(false)
  const playingRef = useRef(false)
  const turnRunningRef = useRef(false)
  /** True after a press that stopped nothing, so the next press with a running turn cancels it. */
  const armedRef = useRef(false)
  /** The reply currently audible, or null. Cleared on every path that ends playback. */
  const playbackRef = useRef<{ stop: () => void } | null>(null)
  const cancellingRef = useRef(false)
  /**
   * The barge-in listener: the microphone and its analyser while a reply is audible and no capture
   * runs. Set by openListener, cleared by detachListener (hand-over) or releaseListener (release).
   */
  const listenRef = useRef<{ stream: MediaStream; context: AudioContext; analyser: AnalyserNode; timer: ReturnType<typeof setInterval> | null } | null>(null)
  /** True while the listener's getUserMedia request is open, so a second request is never started. */
  const listenPendingRef = useRef(false)
  /** Set once on a denied or failed microphone request. Barge-in then stays off for this mount. */
  const listenUnavailableRef = useRef(false)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startedRef = useRef<number>(0)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const captureRef = useRef<SegmentedCapture | null>(null)
  /** The legacy single recording's microphone stream, released on unmount. */
  const legacyStreamRef = useRef<MediaStream | null>(null)
  /** False once this button unmounted: no late transcript or answer may reach the composer. */
  const liveRef = useRef(true)
  const micRef = useRef(mic)
  micRef.current = mic
  const settingsRef = useRef(settings)
  settingsRef.current = settings

  // One-shot settings snapshot (cap, engines, auto-submit, hotkey).
  useEffect(() => {
    let cancelled = false
    void status().then(snapshot => {
      if (!cancelled) setSettings(snapshot)
    }).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [status])

  // Silence is not an error: an empty transcript leaves the composer as it was.
  const finishWithText = (text: string): void => {
    if (!liveRef.current) return
    const trimmed = text.trim()
    setMic(foldMic(micRef.current, trimmed === '' ? { kind: 'reset' } : { kind: 'transcribed', text: trimmed }))
    if (trimmed === '') return
    if (settingsRef.current?.record.autoSubmit === true) {
      void submitText(trimmed)
    } else {
      writeDraft(trimmed)
    }
  }

  // Convert one MediaRecorder recording to 16 kHz mono WAV in the browser, post
  // it to the host engine, and name the failing stage in the composer.
  const transcribeRecording = async (blob: Blob): Promise<void> => {
    let wav: Uint8Array
    try {
      wav = await recordingToWav16kMono(await blob.arrayBuffer())
    } catch (error) {
      setMic(foldMic(micRef.current, { kind: 'failed', message: `could not convert the recording to 16 kHz WAV: ${detailOf(error)}` }))
      return
    }
    try {
      const transcript = await transcribe(bytesToBase64(wav))
      finishWithText(transcript.text)
    } catch (error) {
      setMic(foldMic(micRef.current, { kind: 'failed', message: `transcription failed: ${detailOf(error)}` }))
    }
  }

  const stopRecognition = (): void => {
    recognitionRef.current?.stop()
    recognitionRef.current = null
  }

  const stopRecorder = (): void => {
    const recorder = recorderRef.current
    recorderRef.current = null
    if (recorder !== null && recorder.state !== 'inactive') {
      try {
        recorder.stop()
      } catch {
        // The recorder already stopped; the dataavailable handler finished us.
      }
    }
  }

  // Feature-detected voice-activity detection: attach an AnalyserNode to the
  // live stream when the recorder path is used. When it is present, the capture
  // is segmented (startSegmentedCapture). Absent AudioContext means one recording
  // that the user stops by hand.
  const setupVad = (stream: MediaStream, vad: TalkStatus['record']['vad']): void => {
    if (!vad.enabled) return
    const Ctor = (globalThis as unknown as { AudioContext?: typeof AudioContext }).AudioContext
    if (Ctor === undefined) return
    try {
      const context = new Ctor()
      const source = context.createMediaStreamSource(stream)
      const analyser = context.createAnalyser()
      analyser.fftSize = 2048
      source.connect(analyser)
      audioContextRef.current = context
      analyserRef.current = analyser
    } catch {
      // VAD degrades to one recording, stopped by hand.
    }
  }

  const teardownVad = (): void => {
    analyserRef.current = null
    const context = audioContextRef.current
    audioContextRef.current = null
    if (context !== null) {
      try {
        void context.close()
      } catch {
        // The context already closed.
      }
    }
  }

  // Serial chain for one segmented capture: segments and the final step run in order.
  const enqueue = (capture: SegmentedCapture, job: () => Promise<void>): void => {
    capture.queue = capture.queue.then(job).catch(() => {})
  }

  // The final step: release the microphone and, unless the trigger already sent the
  // prompt, put the accumulated transcript in the composer unsent.
  const finalizeCapture = (capture: SegmentedCapture): void => {
    capture.stream.getTracks().forEach(track => track.stop())
    if (captureRef.current === capture) captureRef.current = null
    syncListening()
    if (capture.done || capture.abandoned) return
    const text = capture.accumulated.trim()
    if (text === '') {
      setMic(foldMic(micRef.current, { kind: 'reset' }))
      return
    }
    writeDraft(text)
    setMic(foldMic(micRef.current, { kind: 'transcribed', text }))
  }

  // Queue the final step once the capture is ending and no recorder stop is still pending.
  const maybeFinalize = (capture: SegmentedCapture): void => {
    if (!capture.closing || capture.finalized || capture.segmentsOpen > 0) return
    capture.finalized = true
    enqueue(capture, async () => {
      finalizeCapture(capture)
    })
  }

  // One segment: convert, transcribe, log the raw text, then ask the heal call. A capture that
  // is ending keeps the segment's words in the draft but asks heal nothing and submits nothing.
  // An unmounted capture keeps nothing from its late segments.
  const processSegment = async (capture: SegmentedCapture, blob: Blob): Promise<void> => {
    if (capture.done || capture.abandoned) return
    let wav: Uint8Array
    try {
      wav = await recordingToWav16kMono(await blob.arrayBuffer())
    } catch (error) {
      setMic(foldMic(micRef.current, { kind: 'failed', message: `could not convert the recording to 16 kHz WAV: ${detailOf(error)}` }))
      return
    }
    let raw: string
    try {
      raw = (await transcribe(bytesToBase64(wav))).text
    } catch (error) {
      // The segment's audio is not kept for a retry, so its words are lost. The capture stays open.
      setMic(foldMic(micRef.current, { kind: 'failed', message: `transcription failed, so this segment's words are lost: ${detailOf(error)}` }))
      return
    }
    if (capture.abandoned) return
    console.log('[dsh-talk] segment raw transcript:', raw)
    const text = raw.trim()
    if (text === '') return
    capture.accumulated = capture.accumulated === '' ? text : `${capture.accumulated} ${text}`
    if (capture.closing) return
    let decision: TalkHeal
    try {
      // The host runs the one model call (and its literal-phrase fallback) on the accumulated text.
      decision = await heal(capture.accumulated)
    } catch (error) {
      // The host's fallback already ran and missed (or the call failed): keep the capture open.
      if (capture.abandoned) return
      setMic(foldMic(micRef.current, { kind: 'failed', message: `heal failed: ${detailOf(error)}` }))
      return
    }
    // A late answer is dropped, not acted on: the capture ended (trip, stop or ceiling) or
    // unmounted while the call was in flight. Its transcript is already in the draft.
    if (capture.closing || capture.abandoned) return
    if (!decision.trigger) return
    capture.done = true
    closeCapture(capture)
    const prompt = decision.prompt.trim()
    if (prompt === '') {
      setMic(foldMic(micRef.current, { kind: 'reset' }))
      return
    }
    setMic(foldMic(micRef.current, { kind: 'transcribed', text: prompt }))
    void submitText(prompt)
  }

  const startSegment = (capture: SegmentedCapture): void => {
    let recorder: MediaRecorder
    try {
      recorder = new MediaRecorder(capture.stream, { mimeType: capture.mime })
    } catch {
      setMic(foldMic(micRef.current, { kind: 'failed', message: 'microphone capture failed' }))
      return
    }
    // The chunks belong to this recorder only, so a late onstop never reads the next segment.
    const chunks: Blob[] = []
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data)
    }
    // This segment's own count entry. Its onstop releases it once, so the total cannot go negative.
    let counted = true
    capture.segmentsOpen += 1
    recorder.onstop = () => {
      if (counted) {
        counted = false
        capture.segmentsOpen -= 1
      }
      // A stop this code did not request (the browser ended the recorder) ends the segment
      // too. `capture.recorder` is left alone here: `isRecording` already treats an inactive
      // recorder as not recording, and every path that clears the field clears it outright.
      const blob = new Blob(chunks, { type: capture.mime })
      enqueue(capture, () => processSegment(capture, blob))
      maybeFinalize(capture)
    }
    try {
      recorder.start()
    } catch {
      counted = false
      capture.segmentsOpen -= 1
      setMic(foldMic(micRef.current, { kind: 'failed', message: 'microphone capture failed' }))
      return
    }
    capture.recorder = recorder
    capture.segments += 1
    capture.pauseMs = 0
  }

  // Ends the current segment, if any. Its onstop queues the transcription.
  const stopCurrentSegment = (capture: SegmentedCapture): void => {
    const recorder = capture.recorder
    capture.recorder = null
    capture.pauseMs = 0
    // An inactive recorder has stopped or is about to: its onstop releases the count, so nothing is added here.
    if (recorder === null || recorder.state === 'inactive') return
    try {
      recorder.stop()
    } catch {
      // stop() refused. Its onstop still follows a stop the browser accepted; if it does not, the count stays up.
    }
  }

  // Ends the capture: no more detection, the current segment is flushed, then the final step runs.
  const closeCapture = (capture: SegmentedCapture): void => {
    if (capture.closing) return
    capture.closing = true
    if (capture.timer !== null) {
      clearInterval(capture.timer)
      capture.timer = null
    }
    setRecording(false)
    teardownVad()
    setMic(foldMic(micRef.current, { kind: 'stopped' }))
    stopCurrentSegment(capture)
    maybeFinalize(capture)
  }

  // One detector tick. Speech opens a segment; a pause of vad.silenceMs ends it. The
  // 120-second timeout counts silence since the last speech, and only after a segment exists.
  // A recorder is taking audio only while it is neither inactive nor stopped on this code's request.
  const isRecording = (capture: SegmentedCapture): boolean =>
    capture.recorder !== null && capture.recorder.state !== 'inactive'

  const segmentTick = (capture: SegmentedCapture, record: TalkStatus['record']): void => {
    const analyser = analyserRef.current
    if (analyser === null || capture.closing) return
    // The hard ceiling ends the capture as the silence trip does: the transcript goes to the composer unsent.
    if (Date.now() - capture.startedAt >= record.ceilingSeconds * 1000) {
      closeCapture(capture)
      return
    }
    const buffer = new Float32Array(analyser.fftSize)
    analyser.getFloatTimeDomainData(buffer)
    const speech = isSpeechFrame(rmsOf(buffer), record.vad.energyThreshold)
    capture.idleMs = advanceSilence(capture.idleMs, TICK_MS, speech)
    // Barge-in reuses this capture's detector: speech while audio is audible stops it.
    if (speech && playingRef.current) bargeIn()
    if (speech) {
      capture.pauseMs = 0
      // A recorder that stopped by itself is not taking audio, so speech starts a new segment at once.
      if (!isRecording(capture)) startSegment(capture)
    } else if (isRecording(capture)) {
      capture.pauseMs = advanceSilence(capture.pauseMs, TICK_MS, false)
      if (silenceTrips(capture.pauseMs, record.vad.silenceMs)) stopCurrentSegment(capture)
    }
    if (capture.segments > 0 && silenceTrips(capture.idleMs, CAPTURE_SILENCE_TIMEOUT_MS)) closeCapture(capture)
  }

  const startSegmentedCapture = (stream: MediaStream, mime: string, record: TalkStatus['record']): void => {
    const capture: SegmentedCapture = {
      stream,
      mime,
      recorder: null,
      segments: 0,
      startedAt: Date.now(),
      accumulated: '',
      pauseMs: 0,
      idleMs: 0,
      segmentsOpen: 0,
      closing: false,
      abandoned: false,
      finalized: false,
      done: false,
      queue: Promise.resolve(),
      timer: null,
    }
    captureRef.current = capture
    setMic(foldMic(micRef.current, { kind: 'press', maxSeconds: record.ceilingSeconds }))
    setRecording(true)
    capture.timer = setInterval(() => {
      const elapsed = Math.ceil((Date.now() - capture.startedAt) / 1000)
      setMic(foldMic(micRef.current, { kind: 'tick', elapsedSec: elapsed, maxSeconds: record.ceilingSeconds }))
      segmentTick(capture, record)
    }, TICK_MS)
  }

  const stop = (): void => {
    const capture = captureRef.current
    if (capture !== null) {
      closeCapture(capture)
      return
    }
    setRecording(false)
    if (timerRef.current !== null) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (recognitionRef.current !== null) stopRecognition()
    else stopRecorder()
    teardownVad()
    setMic(foldMic(micRef.current, { kind: 'stopped' }))
  }

  // Register one audible reply. Its end clears the entry, but only while it is still the current one.
  const beginPlayback: PlaybackBegin = (stop) => {
    const handle = { stop }
    playbackRef.current = handle
    playingRef.current = true
    armedRef.current = false
    setPlaying(true)
    syncListening()
    return () => {
      if (playbackRef.current !== handle) return
      playbackRef.current = null
      playingRef.current = false
      setPlaying(false)
      syncListening()
    }
  }

  // Stop the audible reply, if any. The turn keeps running.
  const stopPlayback = (): void => {
    const handle = playbackRef.current
    if (handle === null) return
    playbackRef.current = null
    playingRef.current = false
    setPlaying(false)
    handle.stop()
    syncListening()
  }

  // Cancel this session's running turn through the scoped conversation service.
  // Limit: a pending approval does not stop this cancel; it matches the built-in Escape stop, which also skips sessions with a pending interaction.
  const cancelTurnNow = async (): Promise<void> => {
    if (cancelTurn === undefined || typeof sessionIdProp !== 'string' || sessionIdProp === '') return
    if (cancellingRef.current) return
    cancellingRef.current = true
    try {
      await cancelTurn(sessionIdProp)
    } catch (error) {
      if (liveRef.current) setMic(foldMic(micRef.current, { kind: 'failed', message: `cancel failed: ${detailOf(error)}` }))
    } finally {
      cancellingRef.current = false
    }
  }

  // Speech detected in an open capture: stop audible playback, and cancel the turn only when audio was audible.
  const bargeIn = (): void => {
    const plan = planBargeIn({ playing: playingRef.current, turnRunning: turnRunningRef.current })
    if (plan.stopPlayback) stopPlayback()
    if (plan.cancelTurn) void cancelTurnNow()
  }

  // The stop button. One press stops audible playback; with nothing audible, the first press arms and the second cancels.
  const onStopPress = (): void => {
    const plan = planStopPress({ playing: playingRef.current, turnRunning: turnRunningRef.current, armed: armedRef.current })
    armedRef.current = plan.armed
    if (plan.action === 'stop-playback') stopPlayback()
    else if (plan.action === 'cancel-turn') void cancelTurnNow()
  }

  // Take the listener's stream off the listener without stopping its tracks: a capture takes it over.
  const detachListener = (): MediaStream | null => {
    const listener = listenRef.current
    if (listener === null) return null
    listenRef.current = null
    if (listener.timer !== null) clearInterval(listener.timer)
    try {
      void Promise.resolve(listener.context.close()).catch(() => {})
    } catch {
      // The context already closed.
    }
    return listener.stream
  }

  const releaseListener = (): void => {
    detachListener()?.getTracks().forEach(track => track.stop())
  }

  // Whether the barge-in listener may hold the microphone right now.
  const listenWanted = (): boolean =>
    liveRef.current
    && playingRef.current
    && captureRef.current === null
    && settingsRef.current?.record.vad.enabled === true
    && !listenUnavailableRef.current
    && navigator.mediaDevices !== undefined

  // One barge-in tick: speech in the listener's analyser while audio is audible stops it.
  const listenTick = (): void => {
    const listener = listenRef.current
    if (listener === null || !playingRef.current) return
    const buffer = new Float32Array(listener.analyser.fftSize)
    listener.analyser.getFloatTimeDomainData(buffer)
    if (isSpeechFrame(rmsOf(buffer), settingsRef.current?.record.vad.energyThreshold ?? 0.01)) bargeIn()
  }

  const openListener = (stream: MediaStream): void => {
    const Ctor = (globalThis as unknown as { AudioContext?: typeof AudioContext }).AudioContext
    if (Ctor === undefined) {
      stream.getTracks().forEach(track => track.stop())
      listenUnavailableRef.current = true
      console.warn('[dsh-talk] barge-in unavailable: AudioContext is missing')
      return
    }
    try {
      const context = new Ctor()
      const source = context.createMediaStreamSource(stream)
      const analyser = context.createAnalyser()
      analyser.fftSize = 2048
      source.connect(analyser)
      listenRef.current = { stream, context, analyser, timer: setInterval(listenTick, TICK_MS) }
    } catch {
      stream.getTracks().forEach(track => track.stop())
    }
  }

  // The privacy contract: the microphone is live only while a reply is audible or a capture is running, and is released otherwise.
  // A listen request still pending when a press starts opens a second stream for the press's capture; the pending stream is stopped on arrival.
  const syncListening = (): void => {
    if (!listenWanted()) {
      releaseListener()
      return
    }
    if (listenRef.current !== null || listenPendingRef.current) return
    listenPendingRef.current = true
    let request: Promise<MediaStream>
    try {
      request = navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (error) {
      request = Promise.reject(error)
    }
    void request.then(
      (stream) => {
        listenPendingRef.current = false
        if (!listenWanted() || listenRef.current !== null) {
          stream.getTracks().forEach(track => track.stop())
          return
        }
        openListener(stream)
      },
      () => {
        listenPendingRef.current = false
        if (listenUnavailableRef.current) return
        listenUnavailableRef.current = true
        console.warn('[dsh-talk] barge-in unavailable: microphone not granted')
      },
    )
  }

  const start = async (): Promise<void> => {
    const snapshot = settingsRef.current
    if (snapshot === undefined || snapshot.record.enabled !== true) return
    // Speak-to-interrupt: starting to talk stops host playback.
    if (snapshot.interrupt) void interrupt()
    // The browser's Web Speech runs only when the host resolved the 'web'
    // engine. A FunASR or whisper profile must record and post the audio, or
    // the configured engine never sees it.
    const recognition = snapshot.stt.resolved === 'web' ? webRecognition() : undefined
    if (recognition !== undefined) {
      // An 'auto' language must not assign the empty string: Chrome handles
      // '' badly on some setups (observed as a silent immediate no-speech).
      // Fall back to the browser locale instead.
      const lang = snapshot.stt.language === 'auto'
        ? (navigator.language || 'en-US')
        : snapshot.stt.language
      recognition.lang = lang
      recognition.interimResults = true
      recognition.continuous = true
      recognition.maxAlternatives = 1
      let lastFull = ''
      let recognitionFailed = false
      const silenceFinaliseMs = snapshot.stt.silenceFinaliseMs
      let silenceTimer: ReturnType<typeof setTimeout> | null = null
      const clearSilence = (): void => {
        if (silenceTimer !== null) {
          clearTimeout(silenceTimer)
          silenceTimer = null
        }
      }
      const armSilence = (): void => {
        clearSilence()
        silenceTimer = setTimeout(() => {
          silenceTimer = null
          if (recognitionRef.current !== recognition) return
          try {
            recognition.stop()
          } catch {
            // The recognizer already ended; onend handled finalisation.
          }
        }, silenceFinaliseMs)
      }
      armSilence()
      // Web Speech re-reports the whole results collection on every event, so
      // rebuild the complete transcript from index 0 each time instead of
      // accumulating fragments across events (an accumulator duplicates text
      // whenever a result is revised or re-finalised).
      const buildTranscript = (event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }): string => {
        const parts: string[] = []
        for (let index = 0; index < event.results.length; index += 1) {
          const text = event.results[index]?.[0]?.transcript ?? ''
          if (text.trim() !== '') parts.push(text.trim())
        }
        return parts.join(' ')
      }
      recognition.onresult = (event) => {
        lastFull = buildTranscript(event)
        armSilence()
        if (lastFull !== '') writeDraft(lastFull)
      }
      recognition.onend = () => {
        clearSilence()
        recognitionRef.current = null
        setRecording(false)
        if (!recognitionFailed) finishWithText(lastFull)
      }
      recognition.onerror = (event) => {
        clearSilence()
        // 'no-speech' is silence: onend still follows and finishes with no text and no error.
        if (event.error === 'no-speech') return
        recognitionFailed = true
        recognitionRef.current = null
        setRecording(false)
        console.warn(`[dsh-talk] speech recognition error: ${String(event.error ?? 'unknown')}`)
        setMic(foldMic(micRef.current, { kind: 'failed', message: `web speech recognition failed (${String(event.error ?? 'unknown')})` }))
      }
      recognitionRef.current = recognition
      try {
        recognition.start()
      } catch {
        recognitionRef.current = null
        setMic(foldMic(micRef.current, { kind: 'failed', message: 'web speech recognition unavailable' }))
        return
      }
      setMic(foldMic(micRef.current, { kind: 'press', maxSeconds: snapshot.record.maxSeconds }))
      setRecording(true)
      return
    }
    const mime = supportedMime()
    if (mime === undefined || navigator.mediaDevices === undefined) {
      setMic(foldMic(micRef.current, { kind: 'failed', message: 'MediaRecorder is unavailable in this browser' }))
      return
    }
    let stream: MediaStream
    // A barge-in listener already holds the microphone: the capture takes its stream, so the microphone is never opened twice.
    const handedOver = detachListener()
    if (handedOver !== null) {
      stream = handedOver
    } else {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      } catch {
        setMic(foldMic(micRef.current, { kind: 'failed', message: 'microphone permission denied' }))
        return
      }
    }
    // The button unmounted while the permission prompt was open: release the microphone at once.
    if (!liveRef.current) {
      stream.getTracks().forEach(track => track.stop())
      return
    }
    setupVad(stream, snapshot.record.vad)
    if (analyserRef.current !== null) {
      startSegmentedCapture(stream, mime, snapshot.record)
      return
    }
    let recorder: MediaRecorder
    try {
      recorder = new MediaRecorder(stream, { mimeType: mime })
    } catch {
      stream.getTracks().forEach(track => track.stop())
      setMic(foldMic(micRef.current, { kind: 'failed', message: 'microphone capture failed' }))
      return
    }
    chunksRef.current = []
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data)
    }
    recorder.onstop = () => {
      stream.getTracks().forEach(track => track.stop())
      const blob = new Blob(chunksRef.current, { type: mime })
      chunksRef.current = []
      void transcribeRecording(blob)
    }
    recorderRef.current = recorder
    legacyStreamRef.current = stream
    try {
      recorder.start()
    } catch {
      recorderRef.current = null
      setMic(foldMic(micRef.current, { kind: 'failed', message: 'microphone capture failed' }))
      return
    }
    startedRef.current = Date.now()
    setMic(foldMic(micRef.current, { kind: 'press', maxSeconds: snapshot.record.maxSeconds }))
    setRecording(true)
    timerRef.current = setInterval(() => {
      const elapsed = Math.ceil((Date.now() - startedRef.current) / 1000)
      setMic(foldMic(micRef.current, { kind: 'tick', elapsedSec: elapsed, maxSeconds: settingsRef.current?.record.maxSeconds ?? 60 }))
      if (elapsed >= (settingsRef.current?.record.maxSeconds ?? 60)) stop()
    }, 500)
  }

  // Unmount: stop the detector interval, the segment recorder and the legacy recorder, release
  // every microphone track, and mark the capture abandoned. The submit path reads the same live
  // capture object this cleanup mutates, so a heal answer that arrives later is dropped.
  useEffect(() => {
    liveRef.current = true
    return () => {
      liveRef.current = false
      const capture = captureRef.current
      captureRef.current = null
      if (capture !== null) {
        capture.abandoned = true
        capture.closing = true
        if (capture.timer !== null) {
          clearInterval(capture.timer)
          capture.timer = null
        }
        const segment = capture.recorder
        capture.recorder = null
        capture.stream.getTracks().forEach(track => track.stop())
        if (segment !== null && segment.state !== 'inactive') {
          try {
            segment.stop()
          } catch {
            // The recorder already stopped.
          }
        }
      }
      if (timerRef.current !== null) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      stopRecorder()
      releaseListener()
      legacyStreamRef.current?.getTracks().forEach(track => track.stop())
      legacyStreamRef.current = null
      if (recognitionRef.current !== null) {
        try {
          recognitionRef.current.abort()
        } catch {
          // The recognizer already ended.
        }
        recognitionRef.current = null
      }
      teardownVad()
    }
    // Once per mount: the cleanup reads refs only, so the first render's closures are current.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Optional toggle hotkey.
  useEffect(() => {
    const hotkey = settings?.record.hotkey
    if (hotkey === null || hotkey === undefined || hotkey === '') return
    const listener = (event: KeyboardEvent): void => {
      if (event.altKey && event.key.toLowerCase() === hotkey.toLowerCase().slice('alt+'.length)) {
        event.preventDefault()
        if (recording) stop()
        else void start()
      }
    }
    window.addEventListener('keydown', listener)
    return () => {
      window.removeEventListener('keydown', listener)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings, recording])

  // Turn state for the stop control: follow this session's running turn while it is mounted.
  // Without the host's turn services the watch is absent, so no turn is ever reported running.
  useEffect(() => {
    if (typeof sessionIdProp !== 'string' || sessionIdProp === '' || watchTurn === undefined) return undefined
    const unwatch = watchTurn(sessionIdProp, (running) => {
      turnRunningRef.current = running
      if (!running) armedRef.current = false
      setTurnRunning(running)
    })
    return () => {
      unwatch()
      turnRunningRef.current = false
      armedRef.current = false
      setTurnRunning(false)
    }
  }, [sessionIdProp, watchTurn])

  // Playback: poll talk/latest for this session and play each new utterance
  // once (speechPoll.ts). The session id arrives as a slot prop; without it
  // there is no session to ask about, so no poll starts and the mic still works.
  // The talk:speech projection is no longer a playback trigger: on a host that
  // does write the event, both paths would play the same reply twice.
  useEffect(() => {
    if (typeof sessionIdProp !== 'string' || sessionIdProp === '') return undefined
    // Per mount: once this page's engine has fired `onstart`, the start fallback is off.
    const browserEngine: BrowserEngineState = { firesStart: false }
    return startSpeechPoll(sessionIdProp, {
      latest,
      audio,
      speakText: (text, onDuration, released, onRelease) =>
        speakWithBrowserVoice(text, settingsRef.current?.tts.browser, onDuration, released, onRelease, browserEngine, beginPlayback),
      playAudio: (entry, onDuration, onRelease) => playAudioEntry(entry, onDuration, onRelease, beginPlayback),
      isHidden: () => typeof document !== 'undefined' && document.hidden,
      onVisibilityChange: (listener) => {
        if (typeof document === 'undefined') return () => {}
        document.addEventListener('visibilitychange', listener)
        return () => document.removeEventListener('visibilitychange', listener)
      },
    })
  }, [sessionIdProp, latest, audio])

  const disabled = settings === undefined || settings.record.enabled !== true
  const label = recording
    ? (mic.phase === 'transcribing' ? '…' : '⏹')
    : '🎙'
  return (
    <>
      <button
        type="button"
        data-dsh-talk-mic
        data-recording={recording ? 'true' : 'false'}
        data-disabled={disabled ? 'true' : 'false'}
        title={mic.phase === 'error' ? mic.error : undefined}
        disabled={disabled}
        onClick={() => {
          if (recording) stop()
          else start()
        }}
      >
        {label}
      </button>
      {stopControlVisible({ playing, turnRunning })
        ? (
          <button
            type="button"
            data-dsh-talk-stop
            data-playing={playing ? 'true' : 'false'}
            data-turn-running={turnRunning ? 'true' : 'false'}
            aria-label={stopLabel}
            title={stopLabel}
            onClick={onStopPress}
          >
            ■
          </button>
        )
        : null}
      {mic.phase === 'error' && mic.error !== undefined
        ? <span data-dsh-talk-mic-error role="alert">{mic.error}</span>
        : null}
    </>
  )
}
