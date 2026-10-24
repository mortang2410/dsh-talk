/**
 * Voice-turn bookkeeping for the spoken reply loop. The client arms a turn just
 * before it submits a microphone prompt. The host then (1) tells the model once,
 * in that turn, that the user is listening and must be answered through the
 * `speak` tool, and (2) when the turn ends with no `speak` call, speaks a short
 * version of the final reply itself. Pure logic only: the service and the plugin
 * wiring own the clock-free inputs and the speech output.
 *
 * @module dsh-talk/voice-turn
 */

import type { SessionEvent } from '@deepseek-ai/dsh-session/types'
import { speakableText } from './tool.ts'

/** How long an armed turn waits for its prompt to start a turn. */
export const VOICE_ARM_TTL_MS = 60_000

/** Most sentences of the reply the fallback speaks. */
export const VOICE_REPLY_MAX_SENTENCES = 4

/** Longest fallback utterance, in characters. */
export const VOICE_REPLY_MAX_CHARS = 600

/** Spoken when the reply has nothing speakable (code only, or empty). */
export const VOICE_REPLY_FALLBACK = 'The reply is in the chat.'

/** The model-facing instruction added once at the start of a voice turn. */
export const VOICE_TURN_INSTRUCTION = [
  'The user spoke this prompt into the voice loop, so the user is listening, not reading.',
  'Call the speak tool once with a short plain paraphrase of your answer: at most four sentences, no code, tables, file paths or URLs.',
  'Then write your full answer in the chat as usual.',
].join(' ')

/**
 * Per-session record of which turns came from the microphone and whether the
 * speak tool ran during them. Keyed by the session id string.
 */
export class VoiceTurns {
  private readonly armed = new Map<string, number>()
  private readonly active = new Map<string, { spoke: boolean }>()

  /** @param now - clock in milliseconds; injectable for tests. */
  constructor(private readonly now: () => number = Date.now) {}

  /** Mark the session's next submitted prompt as a voice prompt. */
  arm(sessionId: string): void {
    this.armed.set(sessionId, this.now())
  }

  /**
   * Called at the first step of a turn. Consumes any arm: a fresh arm makes this
   * turn a voice turn; no arm, or a stale one, makes it a typed turn.
   * @returns true when this turn came from the microphone.
   */
  startTurn(sessionId: string): boolean {
    const armedAt = this.armed.get(sessionId)
    this.armed.delete(sessionId)
    const voice = armedAt !== undefined && this.now() - armedAt <= VOICE_ARM_TTL_MS
    if (voice) this.active.set(sessionId, { spoke: false })
    else this.active.delete(sessionId)
    return voice
  }

  /** Record that the speak tool produced audio during the session's current turn. */
  noteToolSpeech(sessionId: string): void {
    const active = this.active.get(sessionId)
    if (active !== undefined) active.spoke = true
  }

  /**
   * Called when the session goes idle. Closes the turn.
   * @returns true when a voice turn ended without the speak tool, so the reply must be spoken by the host.
   */
  finishTurn(sessionId: string): boolean {
    const active = this.active.get(sessionId)
    this.active.delete(sessionId)
    return active !== undefined && !active.spoke
  }
}

/**
 * The text the host speaks when a voice turn ends without a speak call: the
 * first few sentences of the reply after the speakable filter, capped in length.
 * @param reply - the final assistant text of the turn (may be empty).
 * @returns a non-empty utterance.
 */
export function voiceReplyText(reply: string): string {
  const plain = speakableText(reply).replace(/\s+/g, ' ').trim()
  const sentences = plain.match(/[^.!?]+(?:[.!?]+|$)/g) ?? []
  const spoken = sentences
    .slice(0, VOICE_REPLY_MAX_SENTENCES)
    .map(sentence => sentence.trim())
    .join(' ')
    .trim()
    .slice(0, VOICE_REPLY_MAX_CHARS)
    .trim()
  return spoken === '' ? VOICE_REPLY_FALLBACK : spoken
}

/**
 * The final assistant text of the newest turn in a session log: the last
 * `assistant/message` after the last `turn/start` that carries text parts.
 * Tool-call parts and reasoning are ignored.
 * @param events - the session's own events, oldest first.
 * @returns the text, or an empty string when the turn produced none.
 */
export function lastReplyText(events: readonly SessionEvent[]): string {
  let start = 0
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index]?.type === 'turn/start') {
      start = index + 1
      break
    }
  }
  for (let index = events.length - 1; index >= start; index -= 1) {
    const event = events[index]
    if (event?.type !== 'assistant/message') continue
    const message = (event.data as { message?: { content?: unknown } } | undefined)?.message
    const parts: unknown[] = Array.isArray(message?.content) ? message.content : []
    const text = parts
      .filter((part): part is { type: 'text'; text: string } => {
        if (typeof part !== 'object' || part === null) return false
        const candidate = part as { type?: unknown; text?: unknown }
        return candidate.type === 'text' && typeof candidate.text === 'string'
      })
      .map(part => part.text)
      .join('\n')
      .trim()
    if (text !== '') return text
  }
  return ''
}
