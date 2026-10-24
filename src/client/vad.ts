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
 * Barge-in listener energy floor. A reply through speakers is picked up by the microphone, so
 * the listener needs a higher floor than the capture path. 0.05 is a judgement that residual
 * echo sits below normal speech at conversational distance, and only a real-speaker run can
 * confirm it. It is a fixed value, not a multiple of the capture threshold: coupling them
 * flattened every configured threshold from 0.04 upward onto the same number, which carried
 * no information.
 */
export const LISTENER_FLOOR = 0.05

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
