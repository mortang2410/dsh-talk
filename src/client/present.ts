/**
 * Pure presentation for the microphone button: recording state machines
 * down to one discriminated view model plus the last transcription. No
 * browser APIs, no React — unit-tested in isolation.
 *
 * @module dsh-talk/client/present
 */

/** What the mic button is doing right now. */
export type MicPhase = 'idle' | 'starting' | 'recording' | 'transcribing' | 'error'

/** One user-visible recording step. */
export interface MicViewModel {
  /** Current phase. */
  phase: MicPhase
  /** Seconds recorded so far (rounded up), during recording. */
  elapsedSec: number
  /** The last completed transcription, trimmed. */
  lastText: string
  /** Sanitized failure note when phase is error. */
  error: string | undefined
}

/** An event the recorder state can receive. */
export type MicEvent =
  | { kind: 'press'; maxSeconds?: number }
  | { kind: 'tick'; elapsedSec: number; maxSeconds?: number }
  | { kind: 'stopped' }
  | { kind: 'transcribed'; text: string }
  | { kind: 'failed'; message: string }
  | { kind: 'reset' }

/**
 * Fold one event into the view model. The cap is re-checked on every tick so
 * the caller only has to drive `tick` with the current elapsed seconds.
 *
 * @param state - prior view model.
 * @param event - the incoming event.
 * @returns the next view model.
 */
export function foldMic(state: MicViewModel, event: MicEvent): MicViewModel {
  switch (event.kind) {
    case 'press':
      return { phase: 'recording', elapsedSec: 0, lastText: state.lastText, error: undefined }
    case 'tick':
      return state.phase === 'recording'
        ? { ...state, elapsedSec: event.maxSeconds === undefined ? event.elapsedSec : Math.min(event.elapsedSec, event.maxSeconds) }
        : state
    case 'stopped':
      return { ...state, phase: 'transcribing' }
    case 'transcribed':
      return { phase: 'idle', elapsedSec: 0, lastText: event.text.trim(), error: undefined }
    case 'failed':
      return { phase: 'error', elapsedSec: 0, lastText: state.lastText, error: event.message }
    case 'reset':
      return { phase: 'idle', elapsedSec: 0, lastText: state.lastText, error: undefined }
    default:
      return state
  }
}

/** The idle start state. */
export function initMic(): MicViewModel {
  return { phase: 'idle', elapsedSec: 0, lastText: '', error: undefined }
}

/** What the stop control knows at the moment of a press. */
export interface StopInput {
  /** True while a reply is audible (audio element or browser voice). */
  playing: boolean
  /** True while the session has a running turn. */
  turnRunning: boolean
  /** True after a press that stopped nothing new, so the next press is the second one. */
  armed: boolean
}

/** What one stop press does. */
export type StopAction = 'stop-playback' | 'cancel-turn' | 'arm' | 'none'

/** The decision for one stop press, and the armed state to keep for the next press. */
export interface StopPlan {
  action: StopAction
  armed: boolean
}

/**
 * Decide one stop press. While audio is audible the press stops it and leaves
 * the turn running. Otherwise, with a turn running, the first press only arms
 * the control and the second press cancels the turn, so a stray click never
 * destroys a reply on its own.
 *
 * @param input - the playback, turn and armed state at the press.
 * @returns the action and the armed state for the next press.
 */
export function planStopPress(input: StopInput): StopPlan {
  if (input.playing) return { action: 'stop-playback', armed: true }
  if (input.turnRunning && input.armed) return { action: 'cancel-turn', armed: false }
  if (input.turnRunning) return { action: 'arm', armed: true }
  return { action: 'none', armed: false }
}

/**
 * The stop control is shown only when it can do something: audio is audible or
 * a turn is running.
 *
 * @param input - playback and turn state.
 * @returns whether the control is visible.
 */
export function stopControlVisible(input: { playing: boolean; turnRunning: boolean }): boolean {
  return input.playing || input.turnRunning
}

/** What speech detected during a capture does to playback and the turn. */
export interface BargeInPlan {
  stopPlayback: boolean
  cancelTurn: boolean
}

/**
 * Decide the effect of detected speech. Speech stops audible playback. It
 * cancels the turn only when audio was audible and a turn is running; speech
 * while nothing plays cancels nothing.
 *
 * @param input - playback and turn state at the detection.
 * @returns which effects apply.
 */
export function planBargeIn(input: { playing: boolean; turnRunning: boolean; capturing: boolean }): BargeInPlan {
  // While a capture runs, the listener hears the capture's own stream, so a reply's echo can
  // look like speech. Barge-in may stop that playback, but never cancels the turn there: an
  // accidental cancel destroys the turn the user was waiting for, and the stop control still
  // cancels deliberately.
  return { stopPlayback: input.playing, cancelTurn: input.playing && input.turnRunning && !input.capturing }
}

/** The facts the built-in Escape stop reads before it cancels a session's turn. */
export interface TurnGuardInput {
  /** The session binding reports the session as running. */
  running: boolean
  /** The session was removed. */
  removed: boolean
  /** The subagent mode, or null when the session is not a subagent. */
  subagentMode: string | null
  /** The session has a pending interaction (an approval or question awaiting an answer). */
  pendingInteraction: boolean
  /** The conversation publishes an open turn for the session. */
  openTurn: boolean
}

/**
 * Whether a turn may be cancelled now. This mirrors the built-in Escape guard in
 * dsh-client-ui-conversation `lib/client.js` (the `currentTurn` check in
 * `installStopShortcut`): the session must be running, not removed, not a
 * non-continuable subagent, free of a pending interaction, and have an open turn.
 * Barge-in and the stop button both go through this check.
 *
 * @param input - the session facts at the moment of the check.
 * @returns true when a cancel may proceed.
 */
export function turnCancellable(input: TurnGuardInput): boolean {
  return (
    input.running &&
    !input.removed &&
    (input.subagentMode === null || input.subagentMode === 'continuable') &&
    !input.pendingInteraction &&
    input.openTurn
  )
}

/** Recording-cap seconds, clamped to a safe range for display. */
export function clampRecordSeconds(value: number, fallback: number): number {
  if (!Number.isFinite(value) || value <= 0) return fallback
  return Math.min(Math.floor(value), 600)
}
