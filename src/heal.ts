/**
 * One local model call that heals a raw speech transcript and decides whether the
 * speaker ended it with the trigger phrase. When that call fails, a literal check decides
 * instead: the phrase must end the raw transcript, or the call throws.
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
 * Abort timeout for the heal call, in milliseconds. It is a guard against a wedged
 * server, not a latency budget. Warm sequential calls measured 0.14 to 0.34 s; the first
 * call after an idle gap measured 4.6 s and 8.6 s, and five of twenty calls timed out at
 * 15 s while three agents shared the endpoint. The GPU is shared: `ninfer-serve.service`
 * holds 15.9 of 16.3 GiB and read 99 to 100 percent while other sessions were generating
 * (`WORKSTREAMS.md`, "Endpoint contention"). Raising the value only makes a failure
 * slower, because the fallback is what saves the utterance; lowering it below the 8.6 s
 * outlier would abort a call the model is still answering.
 */
// TODO(k6fr): promote HEAL_TIMEOUT_MS to a Config field when the capture loop is wired.
export const HEAL_TIMEOUT_MS = 15_000

/** The system prompt, verbatim from `HEAL-PROMPT.md`. Edit that file, not this constant. */
export const HEAL_SYSTEM_PROMPT = `You turn a raw speech transcript into a clean prompt. Reply with JSON only: {"trigger": true|false, "prompt": "<cleaned text without the trigger phrase>"}. The trigger phrase is "that is my prompt" or "that's my prompt". It ends the utterance. Fire on close variants of the word "prompt" caused by recognition error, such as "promt", "promp", "prompted". Do NOT fire when the recogniser returned a different word with a different meaning, such as "problem", "pocket" or "profit", even if it sounds similar: those are content, not the phrase. Set trigger true only when the phrase ends the transcript. If the phrase appears earlier and speech continues, trigger is false and the prompt keeps those words. Remove filler words such as "um", "uh" and "er", and false starts, and collapse a word repeated immediately such as "the the" to one. Drop a leading discourse marker such as "so", "okay" or "well" when it opens the transcript. Never change any other wording.`

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
  /** Receives a message each time the host overrides a model `trigger: true`. Defaults to `console.warn`. */
  logOverride?: (message: string) => void
}

/** Thrown when the model call failed and the fallback did not find a phrase that ends the transcript. */
export class HealFailure extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'HealFailure'
  }
}

/** Punctuation a recogniser may append after a phrase. */
const TRAILING_PUNCTUATION = /[.!?,;:]+\s*$/u

/**
 * Filler a transcript may carry after the phrase. The system prompt tells the model to drop
 * these, so refusing a phrase that carries one would contradict the model's own instruction.
 */
const TRAILING_FILLER = /\b(?:u+m+|u+h+|e+r+|erm|hmm+|mm+|uhm)\b\s*$/iu

/** The text without trailing punctuation and filler words, removed until neither is left. */
function withoutTrailingNoise(text: string): string {
  // Only the end is trimmed: `finalPhraseIndex` returns an index into this string, and the
  // caller slices the original with it, so a leading space must not shift the offsets.
  let current = text.trimEnd()
  for (;;) {
    const next = current.replace(TRAILING_PUNCTUATION, '').replace(TRAILING_FILLER, '').trimEnd()
    if (next === current) return current
    current = next
  }
}

/**
 * Index where the literal phrase that ends `text` begins, or `null`. Trailing punctuation and
 * filler are ignored, because a recogniser adds them and the prompt already drops them: a
 * phrase followed by a comma or by "uh" still ends the utterance.
 */
function finalPhraseIndex(text: string, pattern: RegExp): number | null {
  const match = pattern.exec(withoutTrailingNoise(text))
  return match === null ? null : match.index
}

/** The two literal trigger forms of the fallback, at the end of the transcript. */
const FALLBACK_FINAL_PHRASE = /\bthat(?: is |'s )my prompt$/iu

/**
 * The three literal forms the host check uses on a model answer: the two above plus
 * "thats my prompt". A variant such as "promt" matches none of them, so the host never
 * vetoes or strips it: the model decides.
 *
 * This set is deliberately wider than the fallback's, because the two checks guard opposite
 * mistakes. The fallback decides whether to SEND, so it is narrow: a wrong send is worse than
 * a missed one, and the apostrophe-less spelling is not accepted there. The host check
 * decides whether to send EARLY, so it is wide: any literal form found away from the end is
 * evidence that the model read the transcript wrong, and the wider set catches more of those
 * without touching the variants only the model can recognise.
 */
const HOST_LITERAL_FORM = /\bthat(?: is |'s |s )my prompt\b/iu
const HOST_FINAL_PHRASE = /\bthat(?: is |'s |s )my prompt$/iu
/** Every literal form anywhere, for removing a phrase the model left inside its own prompt. */
const HOST_ANY_PHRASE = /\bthat(?: is |'s |s )my prompt\b[.!?,;:]*/giu

/**
 * Heal a raw transcript with exactly one model call and report the trigger decision.
 *
 * The raw transcript is logged before the request is sent. It is user speech, and it is
 * written to the host's own console (journal) on this machine by design. The model call
 * carries `reasoning_effort: "none"`: with reasoning on, the same call spent its budget
 * on reasoning tokens and returned empty content. If the call fails for any reason (HTTP
 * error, timeout, non-JSON body, empty content, non-boolean `trigger`, non-string
 * `prompt`), the literal-phrase fallback runs on the raw transcript. The fallback decides
 * only when a literal form ends the transcript, ignoring trailing punctuation and filler words:
 * it returns `trigger: true` with every literal form stripped from what precedes it, which is
 * empty when the phrase was all that was said. A transcript with speech after the phrase makes
 * it throw; an empty prompt does not, because the caller must not send one and does not.
 *
 * When the model call answers `trigger: true`, the host checks the answer against the text
 * (defence in depth; the model still decides every other case):
 * - Veto: a literal form appears in the raw transcript but does not end it. The model has
 *   contradicted the text, so the decision becomes `trigger: false` with the transcript unchanged.
 * - Strip: every literal form in the returned prompt is removed, wherever the model left it.
 * Trailing punctuation and filler words (um, uh, er) never make a phrase count as mid-text, so
 * "... that is my prompt," and "... that is my prompt uh" are triggers, not vetoes.
 * Each veto is reported through `logOverride`, so a model that keeps contradicting the text is visible.
 *
 * @param rawTranscript - the accumulated raw transcript for the current capture.
 * @param options - optional endpoint, timeout and logger overrides.
 * @returns the decision, with `source` naming which path decided.
 * @throws {HealFailure} when the model call failed and the transcript does not end with a literal phrase.
 */
export async function healTranscript(rawTranscript: string, options: HealOptions = {}): Promise<HealDecision> {
  const logRaw = options.logRawTranscript ?? logToConsole
  logRaw(rawTranscript)
  try {
    const model = await callModel(rawTranscript, options.endpoint ?? HEAL_ENDPOINT, options.timeoutMs ?? HEAL_TIMEOUT_MS)
    return checkModelDecision(rawTranscript, model, options.logOverride ?? warnToConsole)
  } catch (error) {
    const reason = sanitizeText(error, 300)
    const prompt = promptBeforeFinalPhrase(rawTranscript)
    if (prompt === null) throw new HealFailure(`heal call failed (${reason}) and the transcript does not end with the trigger phrase`)
    return { trigger: true, prompt, source: 'fallback' }
  }
}

/**
 * Check a model answer against the raw transcript. Only `trigger: true` is checked; a
 * `false` answer passes unchanged. The veto fires only when a literal form appears and the
 * transcript does not end with one. A variant such as "promt" never fires it.
 */
function checkModelDecision(
  rawTranscript: string,
  model: { trigger: boolean; prompt: string },
  logOverride: (message: string) => void,
): HealDecision {
  if (!model.trigger) return { trigger: false, prompt: model.prompt, source: 'model' }
  if (finalPhraseIndex(rawTranscript, HOST_FINAL_PHRASE) === null && HOST_LITERAL_FORM.test(rawTranscript)) {
    logOverride('dsh-talk heal: model answered trigger true, but the literal trigger phrase is not at the end of the transcript; overridden to trigger false')
    return { trigger: false, prompt: rawTranscript, source: 'model' }
  }
  const prompt = stripPhrases(model.prompt)
  if (prompt !== '') return { trigger: true, prompt, source: 'model' }
  // The model fired the trigger but returned nothing to send. If the speaker said something
  // before the phrase, that speech must not be dropped: use the text before the phrase. The
  // host form set is used here, not the fallback's, because the model has already recognised
  // the phrase and a spelling such as "thats my prompt" must not lose the words before it.
  const before = promptBeforeFinalPhrase(rawTranscript, HOST_FINAL_PHRASE)
  if (before !== null && before !== '') {
    logOverride('dsh-talk heal: model answered trigger true with an empty prompt; using the text before the trigger phrase')
    return { trigger: true, prompt: before, source: 'model' }
  }
  return { trigger: true, prompt: '', source: 'model' }
}

/**
 * The prompt with every literal form removed, wherever the model left it. The ticket's
 * criterion is that the phrase never appears in what is sent, so a form in the middle goes
 * too, not only one at the end.
 */
function stripPhrases(prompt: string): string {
  return prompt.replace(HOST_ANY_PHRASE, ' ').replace(/\s+/gu, ' ').trim()
}

/**
 * The fallback's only decision. When a literal form ends the transcript, return the text
 * before it, whitespace collapsed. Otherwise return `null`, and the caller throws.
 */
function promptBeforeFinalPhrase(rawTranscript: string, pattern: RegExp = FALLBACK_FINAL_PHRASE): string | null {
  const index = finalPhraseIndex(rawTranscript, pattern)
  if (index === null) return null
  // A phrase the speaker also said earlier is still the phrase: it must not reach the prompt.
  // The phrase alone leaves an empty prompt. That is not a miss and not an error: the phrase
  // fired and there is nothing before it to send, so the caller's empty-prompt guard resets the
  // capture quietly (TalkMicButton.tsx, `if (prompt === '')`).
  return stripPhrases(rawTranscript.slice(0, index))
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

function warnToConsole(message: string): void {
  console.warn(message)
}
