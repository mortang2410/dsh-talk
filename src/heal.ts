/**
 * One local model call that heals a raw speech transcript and decides whether the
 * speaker ended it with the trigger phrase. When that call fails, a literal-phrase
 * check on the raw transcript decides instead.
 *
 * `HEAL_SYSTEM_PROMPT` is copied byte-for-byte from the fenced block under
 * "## System prompt, verbatim" in the project's `HEAL-PROMPT.md`. The byte-identity
 * test in `tests/heal.spec.ts` asserts that identity when `DSH_TALK_HEAL_PROMPT`
 * names that file. Edit the file first, re-measure, then copy the change here.
 *
 * @module dsh-talk/heal
 */

import { sanitizeText } from './sanitize.ts'

/** Loopback endpoint of the resident `bonsai2-27b` model (`HEAL-PROMPT.md`, Request). */
export const HEAL_ENDPOINT = 'http://127.0.0.1:8082/v1/chat/completions'

/** Model name sent in the request body. */
export const HEAL_MODEL = 'bonsai2-27b'

/**
 * Abort timeout for the heal call, in milliseconds. Warm calls measured 0.15 to 0.34 s.
 * One call after an idle gap took 8.58 s (`research/1cds-heal-recheck.md`). 15 s clears
 * that outlier with about 1.7x headroom. A value under 8.58 s would abort a call the
 * model was still answering, and the fallback would then miss any non-literal form.
 */
export const HEAL_TIMEOUT_MS = 15_000

/** The system prompt, verbatim from `HEAL-PROMPT.md`. Edit that file, not this constant. */
export const HEAL_SYSTEM_PROMPT = `You turn a raw speech transcript into a clean prompt. Reply with JSON only: {"trigger": true|false, "prompt": "<cleaned text without the trigger phrase>"}. The trigger phrase is "that is my prompt" or "that's my prompt". It ends the utterance. Fire on close variants of the word "prompt" caused by recognition error, such as "promt", "promp", "prompted". Do NOT fire when the recogniser returned a different word with a different meaning, such as "problem", "pocket" or "profit", even if it sounds similar: those are content, not the phrase. Set trigger true only when the phrase ends the transcript. If the phrase appears earlier and speech continues, trigger is false and the prompt keeps those words.`

/** What a caller receives: the trigger decision and the cleaned prompt. */
export interface HealDecision {
  /** True when the speaker ended the transcript with the trigger phrase. */
  trigger: boolean
  /** The cleaned prompt. When `trigger` is true, the trigger phrase is removed. */
  prompt: string
  /** `model` when the model call answered; `fallback` when the literal-phrase check decided. */
  source: 'model' | 'fallback'
}

/** Optional overrides. Production callers pass none; tests point `endpoint` at a stub. */
export interface HealOptions {
  /** Endpoint URL. Defaults to `HEAL_ENDPOINT`. */
  endpoint?: string
  /** Abort timeout in milliseconds. Defaults to `HEAL_TIMEOUT_MS`. */
  timeoutMs?: number
  /** Receives the raw transcript before the request is sent. Defaults to `console.log`. */
  logRawTranscript?: (rawTranscript: string) => void
}

/** Thrown when the model call failed and the literal-phrase fallback did not match. */
export class HealFailure extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'HealFailure'
  }
}

/**
 * The two literal trigger forms the fallback recognises, ignoring case. `\b` at the end
 * keeps "that is my prompted" from matching; the optional `[.!?]` removes the sentence
 * mark that recognisers attach to the phrase.
 */
const LITERAL_PHRASE = /\bthat(?: is |'s )my prompt\b[.!?]?/giu

/**
 * Heal a raw transcript with exactly one model call and report the trigger decision.
 *
 * The raw transcript is logged before the request is sent. The model call carries
 * `reasoning_effort: "none"`: with reasoning on, the same call spent its budget on
 * reasoning tokens and returned empty content. If the call fails for any reason (HTTP
 * error, timeout, non-JSON body, empty content, non-boolean `trigger`, non-string
 * `prompt`), the literal-phrase fallback runs on the raw transcript. A fallback hit
 * returns `trigger: true` with the phrase stripped. A fallback miss throws.
 *
 * @param rawTranscript - the accumulated raw transcript for the current capture.
 * @param options - optional endpoint, timeout and logger overrides.
 * @returns the decision, with `source` naming which path decided.
 * @throws {HealFailure} when the model call failed and the fallback found no literal phrase.
 */
export async function healTranscript(rawTranscript: string, options: HealOptions = {}): Promise<HealDecision> {
  const logRaw = options.logRawTranscript ?? logToConsole
  logRaw(rawTranscript)
  try {
    const { trigger, prompt } = await callModel(rawTranscript, options.endpoint ?? HEAL_ENDPOINT, options.timeoutMs ?? HEAL_TIMEOUT_MS)
    return { trigger, prompt, source: 'model' }
  } catch (error) {
    const reason = sanitizeText(error, 300)
    const stripped = stripLiteralPhrase(rawTranscript)
    if (stripped === null) throw new HealFailure(`heal call failed (${reason}) and the transcript has no literal trigger phrase`)
    return { trigger: true, prompt: stripped, source: 'fallback' }
  }
}

/** Remove the literal trigger forms and return what is left, or `null` when neither appears. */
function stripLiteralPhrase(rawTranscript: string): string | null {
  if (rawTranscript.search(LITERAL_PHRASE) === -1) return null
  return rawTranscript.replace(LITERAL_PHRASE, ' ').replace(/\s+/gu, ' ').trim()
}

/** One POST to the model endpoint. Throws `HealFailure` on any failure; never retries. */
async function callModel(rawTranscript: string, endpoint: string, timeoutMs: number): Promise<{ trigger: boolean; prompt: string }> {
  const requestBody = {
    model: HEAL_MODEL,
    temperature: 0,
    max_tokens: 300,
    reasoning_effort: 'none',
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: HEAL_SYSTEM_PROMPT },
      { role: 'user', content: `raw transcript: ${rawTranscript}` },
    ],
  }
  let response: Response
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (error) {
    throw new HealFailure(`request failed: ${sanitizeText(error, 300)}`)
  }
  if (!response.ok) throw new HealFailure(`endpoint responded ${String(response.status)}`)
  let text: string
  try {
    text = await response.text()
  } catch (error) {
    throw new HealFailure(`response body unreadable: ${sanitizeText(error, 300)}`)
  }
  const content = messageContent(parseJson(text, 'response body'))
  if (typeof content !== 'string' || content === '') throw new HealFailure('response has no message content')
  const decision = parseJson(content, 'message content')
  if (!isRecord(decision)) throw new HealFailure('message content is not a JSON object')
  const { trigger, prompt } = decision
  if (typeof trigger !== 'boolean') throw new HealFailure('message content has no boolean trigger')
  if (typeof prompt !== 'string') throw new HealFailure('message content has no string prompt')
  return { trigger, prompt }
}

/** The `choices[0].message.content` field of a chat-completions body, or `undefined`. */
function messageContent(envelope: unknown): unknown {
  if (!isRecord(envelope) || !Array.isArray(envelope['choices'])) return undefined
  const first: unknown = envelope['choices'][0]
  if (!isRecord(first) || !isRecord(first['message'])) return undefined
  return first['message']['content']
}

function parseJson(text: string, what: string): unknown {
  try {
    return JSON.parse(text) as unknown
  } catch {
    throw new HealFailure(`${what} is not JSON`)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function logToConsole(rawTranscript: string): void {
  console.log('dsh-talk heal raw transcript:', rawTranscript)
}
