/**
 * The composer microphone button: when the host resolved the `web` STT engine,
 * transcribe with the browser's Web Speech API (interim results included);
 * otherwise capture the microphone through an AudioWorklet (capture-worklet.ts), cut
 * speech segments with the pure segmenter (speech-segmenter.ts, pre-roll included),
 * convert each segment to 16 kHz mono WAV (wav.ts), and send it through
 * the host's `talk/transcribe` to the local engine.
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
import {
  advanceSpeechRun,
  isSpeechFrame,
  LISTENER_CONSECUTIVE_FRAMES,
  LISTENER_ONSET_MS,
  LISTENER_FLOOR,
  LISTENER_TICK_MS,
  rmsOf,
} from './vad.ts'
import { CAPTURE_FLUSH_MESSAGE, CAPTURE_WORKLET_NAME, CAPTURE_WORKLET_SOURCE } from './capture-worklet.ts'
import { createSegmenter, feedBlock, flushSegment, type Segmenter } from './speech-segmenter.ts'
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
import { encodeWav16kMono } from './wav.ts'
import type { TalkAudio, TalkHeal, TalkInterruptResult, TalkLatest, TalkStatus, TalkTranscript, TalkVoiceTurn } from '../wire.ts'

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
  /**
   * Arm the session's next prompt as a microphone turn on the host. Absent in
   * builds without the voice-turn service; the prompt is then sent unarmed.
   */
  voiceTurn?: (sessionId: string) => Promise<TalkVoiceTurn>
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
  /** Accessible label while armed: the next press cancels the running turn. */
  cancelLabel?: string
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

/** Close an AudioContext, ignoring a refusal (it already closed). */
function closeContext(context: AudioContext): void {
  try {
    void Promise.resolve(context.close()).catch(() => {})
  } catch {
    // The context already closed.
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
 * Display tick, in milliseconds: how often the elapsed counter of a segmented capture updates.
 * Detection does not run on this tick: the segmenter judges every audio block as it arrives.
 */
const TICK_MS = 500

/**
 * How long a stop waits for the worklet's answer with its partial block. The answer normally comes
 * within a few milliseconds (one render quantum is 128 samples, about 3 ms at 48 kHz). The wait is
 * only a fallback for a context that does not render, such as one still suspended.
 */
const FLUSH_REPLY_TIMEOUT_MS = 500

/**
 * How a segment is transcribed: at most TRANSCRIBE_ATTEMPTS calls in all, TRANSCRIBE_RETRY_MS apart.
 * One failed call does not lose the segment's words; only the last failure does.
 */
const TRANSCRIBE_ATTEMPTS = 3
const TRANSCRIBE_RETRY_MS = 1000

/** How long a first stop press stays armed: the built-in Escape first-press window (stopSequenceMs, 500 ms by default). */
const ARM_WINDOW_MS = 500

/**
 * Microphone constraints for both the capture and the barge-in listener. The browser's echo
 * cancellation removes most of the reply's own speaker output before the analyser reads it.
 */
const MIC_CONSTRAINTS: MediaStreamConstraints = {
  audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
}

/**
 * The audio graph of one capture: the microphone feeds the capture worklet, and the worklet's
 * output goes through a gain of zero to the destination. The worklet runs only while it
 * reaches the destination, and the zero gain keeps it silent.
 */
interface CaptureGraph {
  context: AudioContext
  source: MediaStreamAudioSourceNode
  node: AudioWorkletNode
  gain: GainNode
}

/**
 * One segmented capture, from the press until its final text is placed. The microphone
 * stays open. The segmenter cuts speech segments out of the block stream, and each one is
 * transcribed in order.
 */
interface SegmentedCapture {
  stream: MediaStream
  graph: CaptureGraph
  /** Judges each audio block and cuts the segments, pre-roll included (speech-segmenter.ts). */
  segmenter: Segmenter
  /** When the capture started (Date.now), for the elapsed display. */
  startedAt: number
  /** The raw transcripts of every segment so far, joined with single spaces. */
  accumulated: string
  /**
   * The last heal answer: the accumulated text it judged (`raw`) and the cleaned text it gave (`text`).
   * The stop reuses it when no words came in after it. Null after a trigger that started a new prompt;
   * a trigger answered during the stop stores the text before its phrase.
   */
  cleaned: { raw: string; text: string } | null
  /** Set once the capture is ending (stop or the 120-second timeout). Nothing new is detected or asked of heal. */
  closing: boolean
  /** Set on unmount. Nothing from this capture is transcribed into the draft or submitted after it. */
  abandoned: boolean
  /** Set once the final step has been queued, so it runs once. */
  finalized: boolean
  /** Set once the stop handshake has ended (the worklet answered, or the fallback fired): the graph is torn down. */
  finished: boolean
  /** The fallback timer of the stop handshake, cleared when the answer arrives. */
  flushTimer: ReturnType<typeof setTimeout> | null
  /** Serial chain: every segment and the final step run here, one after another. */
  queue: Promise<void>
  timer: ReturnType<typeof setInterval> | null
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
 * chain. Playback failure is silent to the user: the utterance is already marked
 * seen and must not be retried. A start the browser refuses is logged to the console.
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
    void Promise.resolve(player.play()).catch((error: unknown) => {
      // The browser refused to start this reply (Safari's auto-play block is one case). Logged, not retried.
      console.warn(`[dsh-talk] reply ${entry.utteranceId} did not start:`, error)
      finish()
    })
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
  const { status, interrupt, transcribe, heal, audio, latest, voiceTurn, setDraft, send, forSession, watchTurn, cancelTurn, stopLabel = '', cancelLabel = '' } = props
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
  const [armed, setArmedState] = useState(false)
  const armTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Set the armed state and its expiry. An armed press lapses after ARM_WINDOW_MS, so a press long after
  // the stop cannot cancel the turn on a stale arm.
  const setArmed = (value: boolean): void => {
    armedRef.current = value
    setArmedState(value)
    if (armTimerRef.current !== null) {
      clearTimeout(armTimerRef.current)
      armTimerRef.current = null
    }
    if (!value) return
    armTimerRef.current = setTimeout(() => {
      armTimerRef.current = null
      armedRef.current = false
      setArmedState(false)
    }, ARM_WINDOW_MS)
  }
  /** The reply currently audible, or null. Cleared on every path that ends playback. */
  const playbackRef = useRef<{ stop: () => void } | null>(null)
  const cancellingRef = useRef(false)
  /**
   * The barge-in listener: an analyser on a microphone stream while a reply is audible. An owned stream
   * is the listener's own; a non-owned one is the running capture's stream. Set by openListener, cleared by releaseListener.
   */
  const listenRef = useRef<{ stream: MediaStream; context: AudioContext; analyser: AnalyserNode; timer: ReturnType<typeof setInterval> | null; owned: boolean } | null>(null)
  /** True while the listener's getUserMedia request is open, so a second request is never started. */
  const listenPendingRef = useRef(false)
  /** Set once on a denied or failed microphone request. Barge-in then stays off for this mount. */
  const listenUnavailableRef = useRef(false)
  /** Consecutive listener frames above the listener floor. Reset on a quiet frame and on each playback start. */
  const listenRunRef = useRef(0)
  /** When the current reply started (ms, Date.now). The listener ignores the onset window after it. */
  const playStartedAtRef = useRef(0)

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const captureRef = useRef<SegmentedCapture | null>(null)
  /** True while a press is opening its capture (permission prompt, worklet load), so a second press starts nothing. */
  const startingRef = useRef(false)
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

  const stopRecognition = (): void => {
    recognitionRef.current?.stop()
    recognitionRef.current = null
  }

  // Open the capture graph on the live stream (see CaptureGraph). Returns null when the browser
  // has no AudioContext or AudioWorklet, or a step fails; the caller then refuses to record.
  const setupCapture = async (stream: MediaStream): Promise<CaptureGraph | null> => {
    const Ctor = (globalThis as unknown as { AudioContext?: typeof AudioContext }).AudioContext
    if (Ctor === undefined || typeof AudioWorkletNode === 'undefined') return null
    let context: AudioContext
    try {
      context = new Ctor()
    } catch {
      return null
    }
    try {
      const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET_SOURCE], { type: 'text/javascript' }))
      try {
        await context.audioWorklet.addModule(url)
      } finally {
        URL.revokeObjectURL(url)
      }
      const source = context.createMediaStreamSource(stream)
      const node = new AudioWorkletNode(context, CAPTURE_WORKLET_NAME)
      const gain = context.createGain()
      gain.gain.value = 0
      source.connect(node)
      node.connect(gain)
      gain.connect(context.destination)
      await context.resume()
      // A context that stays suspended after resume captures nothing, so recording would hear silence.
      if (context.state === 'suspended') {
        closeContext(context)
        return null
      }
      return { context, source, node, gain }
    } catch {
      closeContext(context)
      return null
    }
  }

  // Detach the capture graph and close its context. Safe to call more than once.
  const teardownGraph = (graph: CaptureGraph): void => {
    graph.node.port.onmessage = null
    for (const node of [graph.source, graph.node, graph.gain]) {
      try {
        node.disconnect()
      } catch {
        // Already disconnected.
      }
    }
    closeContext(graph.context)
  }

  // Serial chain for one segmented capture: segments and the final step run in order.
  const enqueue = (capture: SegmentedCapture, job: () => Promise<void>): void => {
    capture.queue = capture.queue.then(job).catch(() => {})
  }

  // The cleaned text of the words said since the last trigger, for the stop. The heal answer that judged
  // the same text is reused. Otherwise the host heals it now, with the same bounded call. A failed call
  // gives the raw text, so no word is lost. A trigger heard here gives the text before the phrase; the
  // stop submits it only when autoSubmit is on, and otherwise writes it to the composer.
  const finalText = async (capture: SegmentedCapture): Promise<string> => {
    const raw = capture.accumulated.trim()
    if (raw === '') return ''
    if (capture.cleaned !== null && capture.cleaned.raw === capture.accumulated) return capture.cleaned.text
    try {
      const decision = await heal(capture.accumulated)
      const text = decision.prompt.trim()
      return !decision.trigger && text === '' ? raw : text
    } catch {
      return raw
    }
  }

  // The final step: release the microphone, then put the words said since the last trigger in the composer,
  // cleaned. With the autoSubmit setting on, they are submitted instead, as the Web Speech path does.
  // A trigger clears that text, so no sent prompt is put back.
  const finalizeCapture = async (capture: SegmentedCapture): Promise<void> => {
    capture.stream.getTracks().forEach(track => track.stop())
    if (captureRef.current === capture) captureRef.current = null
    syncListening()
    if (capture.abandoned) return
    const text = await finalText(capture)
    if (capture.abandoned) return
    if (text === '') {
      setMic(foldMic(micRef.current, { kind: 'reset' }))
      return
    }
    setMic(foldMic(micRef.current, { kind: 'transcribed', text }))
    if (settingsRef.current?.record.autoSubmit === true) {
      void submitText(text)
      return
    }
    writeDraft(text)
  }

  // Queue the final step once. It runs after every segment already queued, so the last words reach the draft first.
  const queueFinalize = (capture: SegmentedCapture): void => {
    if (capture.finalized) return
    capture.finalized = true
    enqueue(capture, async () => {
      await finalizeCapture(capture)
    })
  }

  // One segment: encode, transcribe, log the raw text, then ask the heal call. A capture that
  // is ending keeps the segment's words for the stop (finalText), asks heal nothing, and so submits
  // nothing. An unmounted capture keeps nothing from its late segments.
  const processSegment = async (capture: SegmentedCapture, samples: Float32Array): Promise<void> => {
    if (capture.abandoned) return
    const body = bytesToBase64(encodeWav16kMono(samples))
    let raw: string | null = null
    let lastError: unknown = null
    // The segment's audio is kept for the retries (up to TRANSCRIBE_ATTEMPTS calls, TRANSCRIBE_RETRY_MS apart).
    for (let attempt = 1; attempt <= TRANSCRIBE_ATTEMPTS; attempt += 1) {
      if (attempt > 1) await new Promise<void>(resolve => setTimeout(resolve, TRANSCRIBE_RETRY_MS))
      if (capture.abandoned) return
      try {
        raw = (await transcribe(body)).text
        break
      } catch (error) {
        lastError = error
      }
    }
    if (raw === null) {
      // Every call failed, so the segment's words are lost and named. The capture stays open.
      setMic(foldMic(micRef.current, { kind: 'failed', message: `transcription failed, so this segment's words are lost: ${detailOf(lastError)}` }))
      return
    }
    if (capture.abandoned) return
    console.log('[dsh-talk] segment raw transcript:', raw)
    const text = raw.trim()
    if (text === '') {
      // The segment had no words. Say so, rather than show nothing after the speech was cut.
      setMic(foldMic(micRef.current, { kind: 'notice', message: 'no words recognised in that segment' }))
      return
    }
    // Words arrived, so the note about an earlier empty segment no longer applies.
    setMic(foldMic(micRef.current, { kind: 'notice', message: undefined }))
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
    if (capture.abandoned) return
    if (!decision.trigger) {
      // The cleaned text of this answer. The stop writes it, unless more words came in first.
      const prompt = decision.prompt.trim()
      capture.cleaned = { raw: capture.accumulated, text: prompt === '' ? capture.accumulated.trim() : prompt }
      return
    }
    // A trigger answered after the capture began ending is not acted on and not submitted. Its text
    // before the phrase becomes the stop's cleaned text, so the stop does not heal these words again.
    if (capture.closing) {
      capture.cleaned = { raw: capture.accumulated, text: decision.prompt.trim() }
      return
    }
    // The trigger sends what was said so far and starts the next prompt from empty. The
    // capture stays open: only a stop or the 120-second timeout ends it. The mic phase is not
    // touched here, because 'transcribed' and 'reset' would show the mic as idle while
    // the recorder keeps running.
    capture.accumulated = ''
    capture.cleaned = null
    const prompt = decision.prompt.trim()
    if (prompt === '') return
    // Arm the session before the prompt starts its turn, so the host knows this turn is spoken.
    // The await is required: the arm must reach the host before the prompt does.
    if (voiceTurn !== undefined && typeof sessionIdProp === 'string' && sessionIdProp !== '') {
      try {
        await voiceTurn(sessionIdProp)
      } catch (error) {
        console.warn('[dsh-talk] voice turn arm failed:', error)
      }
    }
    // The button may have unmounted while the arm was in flight: no prompt is submitted then.
    if (capture.abandoned) return
    void submitText(prompt)
  }

  // One audio block from the worklet, in order. The segmenter judges every block: a finished
  // segment is queued for transcription, and the 120-second silence timeout closes the capture.
  // Barge-in comes from the listener alone, so the reply's own echo in this stream cannot cancel the turn.
  // Blocks keep arriving until the stop handshake ends: the ones posted before its answer are still in the port queue.
  const onBlock = (capture: SegmentedCapture, block: Float32Array): void => {
    if (capture.finished || capture.abandoned) return
    for (const event of feedBlock(capture.segmenter, block)) {
      if (event.kind === 'timeout') {
        closeCapture(capture)
        return
      }
      const samples = event.samples
      enqueue(capture, () => processSegment(capture, samples))
    }
  }

  // The worklet's messages, in order: a full block, or the answer to the stop request.
  const onPortMessage = (capture: SegmentedCapture, data: unknown): void => {
    if (ArrayBuffer.isView(data)) {
      onBlock(capture, data as Float32Array)
      return
    }
    const reply = data as { kind?: unknown; samples?: unknown } | null
    if (reply !== null && typeof reply === 'object' && reply.kind === 'flushed' && reply.samples instanceof Float32Array) {
      finishCapture(capture, reply.samples)
    }
  }

  // Ends the capture: no more detection and no timeout. The worklet is asked for the partial block it
  // still holds (audio since its last full block); finishCapture runs when that block arrives, or when
  // the fallback timer fires if it never does.
  const closeCapture = (capture: SegmentedCapture): void => {
    if (capture.closing) return
    capture.closing = true
    if (capture.timer !== null) {
      clearInterval(capture.timer)
      capture.timer = null
    }
    setRecording(false)
    setMic(foldMic(micRef.current, { kind: 'stopped' }))
    capture.flushTimer = setTimeout(() => finishCapture(capture, null), FLUSH_REPLY_TIMEOUT_MS)
    capture.graph.node.port.postMessage(CAPTURE_FLUSH_MESSAGE)
  }

  // The end of the stop handshake, once: the partial block is judged, then the open segment (if any)
  // is flushed and transcribed, and the final step runs. Every queued segment runs before the final step.
  const finishCapture = (capture: SegmentedCapture, partial: Float32Array | null): void => {
    if (capture.finished) return
    capture.finished = true
    if (capture.flushTimer !== null) {
      clearTimeout(capture.flushTimer)
      capture.flushTimer = null
    }
    if (partial !== null && partial.length > 0 && !capture.abandoned) {
      for (const event of feedBlock(capture.segmenter, partial)) {
        if (event.kind === 'segment') {
          const samples = event.samples
          enqueue(capture, () => processSegment(capture, samples))
        }
      }
    }
    const tail = flushSegment(capture.segmenter)
    teardownGraph(capture.graph)
    if (tail !== null) enqueue(capture, () => processSegment(capture, tail))
    queueFinalize(capture)
  }

  const startSegmentedCapture = (stream: MediaStream, graph: CaptureGraph, record: TalkStatus['record']): void => {
    const capture: SegmentedCapture = {
      stream,
      graph,
      segmenter: createSegmenter({
        sampleRate: graph.context.sampleRate,
        threshold: record.vad.energyThreshold,
        silenceMs: record.vad.silenceMs,
      }),
      startedAt: Date.now(),
      accumulated: '',
      cleaned: null,
      closing: false,
      abandoned: false,
      finalized: false,
      finished: false,
      flushTimer: null,
      queue: Promise.resolve(),
      timer: null,
    }
    graph.node.port.onmessage = (event: MessageEvent<unknown>) => onPortMessage(capture, event.data)
    captureRef.current = capture
    // A listener that resolved during the permission wait, or an owned one still open, now hands off to this capture's stream.
    syncListening()
    setMic(foldMic(micRef.current, { kind: 'press' }))
    setRecording(true)
    capture.timer = setInterval(() => {
      const elapsed = Math.ceil((Date.now() - capture.startedAt) / 1000)
      setMic(foldMic(micRef.current, { kind: 'tick', elapsedSec: elapsed }))
    }, TICK_MS)
  }

  const stop = (): void => {
    const capture = captureRef.current
    if (capture !== null) {
      closeCapture(capture)
      return
    }
    setRecording(false)
    if (recognitionRef.current !== null) stopRecognition()
    setMic(foldMic(micRef.current, { kind: 'stopped' }))
  }

  // Register one audible reply. Its end clears the entry, but only while it is still the current one.
  const beginPlayback: PlaybackBegin = (stop) => {
    const handle = { stop }
    playbackRef.current = handle
    playingRef.current = true
    setArmed(false)
    // A new reply starts the listener's onset window and its consecutive-frame count from zero.
    playStartedAtRef.current = Date.now()
    listenRunRef.current = 0
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
  // The cancel is guarded by the same turn signal as the watch (index.ts readCancellable), which mirrors
  // the built-in Escape guard at dsh-client-ui-conversation lib/client.js:22677. A pending interaction,
  // a removed session, or a non-continuable subagent makes this cancel a no-op.
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

  // Speech detected by the listener: stop audible playback, and cancel the turn only when audio was audible.
  // Barge-in cancels in one stage, unlike the two-stage stop button: a voice cannot press a button twice.
  const bargeIn = (): void => {
    const plan = planBargeIn({ playing: playingRef.current, turnRunning: turnRunningRef.current, capturing: captureRef.current !== null })
    if (plan.stopPlayback) stopPlayback()
    if (plan.cancelTurn) void cancelTurnNow()
  }

  // The stop button. One press stops audible playback; with nothing audible, the first press arms and the second cancels.
  const onStopPress = (): void => {
    const plan = planStopPress({ playing: playingRef.current, turnRunning: turnRunningRef.current, armed: armedRef.current })
    setArmed(plan.armed)
    if (plan.action === 'stop-playback') stopPlayback()
    else if (plan.action === 'cancel-turn') void cancelTurnNow()
  }

  // Close the listener. Its microphone tracks stop only when the listener owns them; a listener on a capture's stream leaves them to the capture.
  const releaseListener = (): void => {
    const listener = listenRef.current
    if (listener === null) return
    listenRef.current = null
    if (listener.timer !== null) clearInterval(listener.timer)
    try {
      void Promise.resolve(listener.context.close()).catch(() => {})
    } catch {
      // The context already closed.
    }
    if (listener.owned) listener.stream.getTracks().forEach(track => track.stop())
  }

  // Whether the barge-in listener may run right now: a reply is audible, and the settings and microphone allow it.
  const listenWanted = (): boolean =>
    liveRef.current
    && playingRef.current
    && !listenUnavailableRef.current
    && navigator.mediaDevices !== undefined

  // One listener tick. Barge-in needs LISTENER_CONSECUTIVE_FRAMES frames in a row above the
  // listener floor, after the onset window. A silent frame or an onset-window tick resets the run.
  const listenTick = (): void => {
    const listener = listenRef.current
    if (listener === null) return
    if (!playingRef.current || Date.now() - playStartedAtRef.current < LISTENER_ONSET_MS) {
      listenRunRef.current = 0
      return
    }
    const buffer = new Float32Array(listener.analyser.fftSize)
    listener.analyser.getFloatTimeDomainData(buffer)
    const floor = LISTENER_FLOOR
    listenRunRef.current = advanceSpeechRun(listenRunRef.current, isSpeechFrame(rmsOf(buffer), floor))
    if (listenRunRef.current >= LISTENER_CONSECUTIVE_FRAMES) bargeIn()
  }

  // Open the listener on a stream. An owned stream is the listener's own microphone; a non-owned one
  // is a running capture's stream, which that capture stops when it ends.
  const openListener = (stream: MediaStream, owned: boolean): void => {
    const Ctor = (globalThis as unknown as { AudioContext?: typeof AudioContext }).AudioContext
    if (Ctor === undefined) {
      if (!owned) return
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
      listenRef.current = { stream, context, analyser, timer: setInterval(listenTick, LISTENER_TICK_MS), owned }
    } catch {
      if (owned) stream.getTracks().forEach(track => track.stop())
    }
  }

  // The privacy contract: the microphone is live only while a reply is audible or a capture is running, and is released otherwise.
  // During a capture the listener reads the capture's own stream and opens no second microphone. A listen request still pending
  // when a capture or a reply ends is stopped on arrival.
  // Two limits, documented and not fixed:
  // - An AudioContext created without a user gesture may start suspended. The analyser then reads silence, and barge-in is silently off until the first gesture.
  // - The first reply shows the microphone prompt. A denial leaves barge-in off for the mount.
  const syncListening = (): void => {
    if (!listenWanted()) {
      releaseListener()
      return
    }
    const capture = captureRef.current
    const current = listenRef.current
    if (capture !== null) {
      if (current?.owned === true) releaseListener()
      if (listenRef.current === null) openListener(capture.stream, false)
      return
    }
    if (current !== null && !current.owned) releaseListener()
    if (listenRef.current !== null || listenPendingRef.current) return
    listenPendingRef.current = true
    let request: Promise<MediaStream>
    try {
      request = navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS)
    } catch (error) {
      request = Promise.reject(error)
    }
    void request.then(
      (stream) => {
        listenPendingRef.current = false
        // A reply that ended, a capture that started, or a listener that opened meanwhile: this stream is not wanted.
        if (!listenWanted() || captureRef.current !== null || listenRef.current !== null) {
          stream.getTracks().forEach(track => track.stop())
          return
        }
        openListener(stream, true)
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
    // Pressing the mic is an explicit "I want to talk": an audible reply stops first, so the capture never hears it.
    if (playingRef.current) stopPlayback()
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
      setMic(foldMic(micRef.current, { kind: 'press' }))
      setRecording(true)
      return
    }
    if (navigator.mediaDevices === undefined) {
      setMic(foldMic(micRef.current, { kind: 'failed', message: 'microphone capture is unavailable in this browser' }))
      return
    }
    // A second press while this one still opens the capture starts nothing.
    if (startingRef.current) return
    startingRef.current = true
    try {
      let stream: MediaStream
      try {
        stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS)
      } catch {
        setMic(foldMic(micRef.current, { kind: 'failed', message: 'microphone permission denied' }))
        return
      }
      // The button unmounted while the permission prompt was open: release the microphone at once.
      if (!liveRef.current) {
        stream.getTracks().forEach(track => track.stop())
        return
      }
      const graph = await setupCapture(stream)
      // The button may also have unmounted while the worklet loaded: release everything at once.
      if (!liveRef.current || graph === null) {
        if (graph !== null) teardownGraph(graph)
        stream.getTracks().forEach(track => track.stop())
        if (liveRef.current) {
          setMic(foldMic(micRef.current, { kind: 'failed', message: 'voice-activity detection is unavailable in this browser' }))
        }
        return
      }
      startSegmentedCapture(stream, graph, snapshot.record)
    } finally {
      startingRef.current = false
    }
  }

  // Unmount: stop the display timer, tear down the capture graph, release every microphone
  // track, and mark the capture abandoned. The submit path reads the same live capture object
  // this cleanup mutates, so a heal answer that arrives later is dropped.
  useEffect(() => {
    liveRef.current = true
    return () => {
      liveRef.current = false
      const capture = captureRef.current
      captureRef.current = null
      if (capture !== null) {
        capture.abandoned = true
        capture.closing = true
        capture.finished = true
        if (capture.timer !== null) {
          clearInterval(capture.timer)
          capture.timer = null
        }
        if (capture.flushTimer !== null) {
          clearTimeout(capture.flushTimer)
          capture.flushTimer = null
        }
        capture.stream.getTracks().forEach(track => track.stop())
        teardownGraph(capture.graph)
      }
      releaseListener()
      if (armTimerRef.current !== null) {
        clearTimeout(armTimerRef.current)
        armTimerRef.current = null
      }
      if (recognitionRef.current !== null) {
        try {
          recognitionRef.current.abort()
        } catch {
          // The recognizer already ended.
        }
        recognitionRef.current = null
      }
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
      if (!running) setArmed(false)
      setTurnRunning(running)
    })
    return () => {
      unwatch()
      turnRunningRef.current = false
      setArmed(false)
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
            data-armed={armed ? 'true' : 'false'}
            aria-label={armed ? cancelLabel : stopLabel}
            title={armed ? cancelLabel : stopLabel}
            onClick={onStopPress}
          >
            ■
          </button>
        )
        : null}
      {mic.phase === 'error' && mic.error !== undefined
        ? <span data-dsh-talk-mic-error role="alert">{mic.error}</span>
        : null}
      {mic.notice !== undefined
        ? <span data-dsh-talk-mic-notice role="status">{mic.notice}</span>
        : null}
    </>
  )
}
