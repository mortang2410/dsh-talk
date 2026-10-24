/**
 * Pure voice-activity detection (VAD) helpers: root-mean-square energy, the
 * speech/silence frame verdict, and the accumulated-silence state machine.
 * No browser APIs, no React — the microphone button drives these with an
 * `AnalyserNode` and they are unit-tested in isolation. The browser wiring is
 * feature-detected and degrades to no auto-end when `AudioContext` is absent.
 *
 * @module dsh-talk/client/vad
 */

/** Root-mean-square energy of one audio frame (0..1 for normalized float samples). */
export function rmsOf(samples: Float32Array): number {
  if (samples.length === 0) return 0
  let sum = 0
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index]!
    sum += sample * sample
  }
  return Math.sqrt(sum / samples.length)
}

/** A frame's activity verdict at a given energy floor. */
export function isSpeechFrame(rms: number, threshold: number): boolean {
  return rms >= threshold
}

/**
 * Advance the consecutive-silence accumulator by one frame. A speech frame
 * resets the accumulator to zero; a silent frame adds the frame duration.
 * @param previousSilenceMs - accumulated silence so far (ms).
 * @param frameMs - this frame's duration (ms).
 * @param speech - whether this frame was speech.
 * @returns the new accumulated silence (ms).
 */
export function advanceSilence(previousSilenceMs: number, frameMs: number, speech: boolean): number {
  return speech ? 0 : previousSilenceMs + frameMs
}

/** Whether the accumulated silence trips the auto-end threshold. */
export function silenceTrips(silenceMs: number, thresholdMs: number): boolean {
  return silenceMs >= thresholdMs
}

/**
 * Barge-in listener: its energy floor is this multiple of the configured capture threshold.
 * Residual echo from the reply's own speaker output sits below normal speech, so the listener
 * needs a higher floor than the capture. 5x gives 0.05 at the default 0.01. Not measured on
 * real speakers: only a real-speaker run can confirm the value.
 */
export const LISTENER_ENERGY_MULTIPLIER = 5

/** Listener tick, in milliseconds. Shorter than the capture tick, so the frame count stays precise. */
export const LISTENER_TICK_MS = 100

/**
 * Consecutive listener frames above the listener floor before barge-in. 4 x 100 ms is 400 ms of
 * sustained speech, so a single frame (echo residue, a door, a chair) cannot stop a reply.
 */
export const LISTENER_CONSECUTIVE_FRAMES = 4

/**
 * Playback onset, in milliseconds: the listener ignores frames this long after a reply starts,
 * so the start-of-playback transient cannot trip it.
 */
export const LISTENER_ONSET_MS = 500

/**
 * Advance the consecutive-speech run by one frame. A speech frame extends the run; a silent frame
 * resets it to zero.
 * @param previousRun - consecutive speech frames so far.
 * @param speech - whether this frame was above the floor.
 * @returns the new run length.
 */
export function advanceSpeechRun(previousRun: number, speech: boolean): number {
  return speech ? previousRun + 1 : 0
}

/**
 * Silence, in milliseconds, after which a segmented capture ends without a trigger. It is
 * measured from the last speech frame, only once the capture has at least one segment.
 */
export const CAPTURE_SILENCE_TIMEOUT_MS = 120_000
