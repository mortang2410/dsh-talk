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
import { foldMic, initMic, type MicViewModel } from './present.ts'
import { advanceSilence, isSpeechFrame, rmsOf, silenceTrips } from './vad.ts'
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
import type { TalkAudio, TalkInterruptResult, TalkLatest, TalkStatus, TalkTranscript } from '../wire.ts'

/** Registration-side injected face: the recorder's host/browser bindings. */
export interface TalkMicInjected {
  /** Read the effective settings snapshot once. */
  status: () => Promise<TalkStatus>
  /** Stop in-flight host synthesis. */
  interrupt: () => Promise<TalkInterruptResult>
  /** Host transcription for the local STT engines. */
  transcribe: (audioData: string) => Promise<TalkTranscript>
  /** Serve one cached utterance's audio. */
  audio: (utteranceId: string) => Promise<TalkAudio | null>
  /** Read the newest utterance of one session (playback poll). */
  latest: (sessionId: string) => Promise<TalkLatest | null>
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
}

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
 * When the cap fires and the flag still reads true, the reply is audible and the
 * chain would start the next reply over it, so the cap calls
 * `speechSynthesis.cancel()` before it finishes. `cancel()` clears the page's
 * whole speech queue: on a page with more than one mounted talk button, it also
 * stops any other utterance that is queued or speaking. When the flag reads
 * false, the cap does not cancel.
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
 * @param engine - this mount's record of whether the engine fires `onstart`.
 * @returns a promise that resolves when the speech ends or fails, so the poller
 *   can keep it in order with the audio replies.
 */
function speakWithBrowserVoice(
  text: string,
  delivery: TalkStatus['tts']['browser'] | undefined,
  onDuration: (seconds: number) => void,
  released: () => boolean,
  engine: BrowserEngineState,
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
    let started = false
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
      if (timer !== undefined) clearInterval(timer)
      if (startFallback !== undefined) clearTimeout(startFallback)
      if (absoluteCap !== undefined) clearTimeout(absoluteCap)
      resolve()
    }
    utterance.onstart = () => {
      started = true
      engine.firesStart = true
    }
    utterance.onend = finish
    utterance.onerror = finish
    speechSynthesis.speak(utterance)
    // The utterance may have ended synchronously; never start a poll after that.
    if (finished) return
    startFallback = setTimeout(() => {
      fallbackDue = true
    }, ENGINE_START_FALLBACK_MS)
    absoluteCap = setTimeout(() => {
      // The chain releases this reply now. If the flag still reads true, the speech is audible: cancel it first.
      if (readSpeaking() === true) {
        try {
          speechSynthesis.cancel()
        } catch {
          // The engine refused the cancel; the reply still releases.
        }
      }
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
function playAudioEntry(entry: TalkAudio, onDuration: (seconds: number) => void): Promise<void> {
  return new Promise<void>((resolve) => {
    const binary = atob(entry.data)
    const bytes = new Uint8Array(binary.length)
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
    const url = URL.createObjectURL(new Blob([bytes], { type: entry.mime }))
    const player = new Audio(url)
    let finished = false
    const finish = (): void => {
      if (finished) return
      finished = true
      URL.revokeObjectURL(url)
      resolve()
    }
    player.addEventListener('ended', finish)
    player.addEventListener('error', finish)
    player.addEventListener('pause', finish)
    player.addEventListener('abort', finish)
    player.addEventListener('loadedmetadata', () => onDuration(player.duration))
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
  const { status, interrupt, transcribe, audio, latest, setDraft, send, forSession } = props
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

  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startedRef = useRef<number>(0)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const silenceMsRef = useRef<number>(0)
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
  // live stream when the recorder path is used. Absent AudioContext degrades
  // to no auto-end (the user stops manually).
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
      silenceMsRef.current = 0
    } catch {
      // VAD degrades to no auto-end.
    }
  }

  const teardownVad = (): void => {
    analyserRef.current = null
    silenceMsRef.current = 0
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

  const stop = (): void => {
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
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setMic(foldMic(micRef.current, { kind: 'failed', message: 'microphone permission denied' }))
      return
    }
    setupVad(stream, snapshot.record.vad)
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
      const analyser = analyserRef.current
      const vad = settingsRef.current?.record.vad
      if (analyser !== null && vad?.enabled === true) {
        const buffer = new Float32Array(analyser.fftSize)
        analyser.getFloatTimeDomainData(buffer)
        silenceMsRef.current = advanceSilence(silenceMsRef.current, 500, isSpeechFrame(rmsOf(buffer), vad.energyThreshold))
        if (silenceTrips(silenceMsRef.current, vad.silenceMs)) {
          stop()
          return
        }
      }
      if (elapsed >= (settingsRef.current?.record.maxSeconds ?? 60)) stop()
    }, 500)
  }

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
      speakText: (text, onDuration, released) =>
        speakWithBrowserVoice(text, settingsRef.current?.tts.browser, onDuration, released, browserEngine),
      playAudio: playAudioEntry,
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
      {mic.phase === 'error' && mic.error !== undefined
        ? <span data-dsh-talk-mic-error role="alert">{mic.error}</span>
        : null}
    </>
  )
}
