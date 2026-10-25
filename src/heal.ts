/**
 * One local model call that cleans one segment of a raw speech transcript, with the earlier
 * raw words as context. The model never decides the trigger phrase: the client's matcher
 * (`src/trigger.ts`) finds it in the glued cleaned text, so the phrase reaches the model as
 * ordinary words and a model mistake cannot move the cut.
 *
 * `HEAL_WINDOW_SYSTEM_PROMPT` is copied byte-for-byte from the fenced block under
 * "## System prompt, verbatim" in the project's `HEAL-PROMPT.md`. The byte-identity
 * test in `tests/heal.spec.ts` asserts that identity when `DSH_TALK_HEAL_PROMPT`
 * names that file. Edit the file first, re-measure, then copy the change here.
 *
 * @module dsh-talk/heal
 */

/** Endpoint of the resident 27B (`HEAL-PROMPT.md`, Request), reached through Bifrost on
 * 8090 so every heal call leaves a gateway log row with its alias and token counts. A
 * direct call to 8082 works and is never used: Bifrost is the log of record. */
export const HEAL_ENDPOINT = 'http://127.0.0.1:8090/v1/chat/completions'

/** Model id sent in the request body: the Bifrost alias, never the raw served id. The
 * alias `qwen-strata` resolves to `scuba-27b`, the Swift 1.5 Qwen3.8-27B IQ3_XXS build. */
export const HEAL_MODEL = 'strata/qwen-strata'

/**
 * Abort timeout for the heal call, in milliseconds. It is a guard against a wedged
 * server, not a latency budget. Warm sequential calls measured 0.14 to 0.34 s; the first
 * call after an idle gap measured 4.6 s and 8.6 s, and five of twenty calls timed out at
 * 15 s while three agents shared the endpoint. The GPU is shared: `ninfer-serve.service`
 * holds 15.9 of 16.3 GiB and read 99 to 100 percent while other sessions were generating
 * (`WORKSTREAMS.md`, "Endpoint contention"). A timeout is not an error for the capture: the
 * segment keeps its raw newest text. Raising the value only makes a stalled segment slower
 * to fall back, and lowering it below the 8.6 s outlier would abort a call the model is
 * still answering.
 */
// TODO(k6fr): promote HEAL_TIMEOUT_MS to a Config field when the capture loop is wired.
export const HEAL_TIMEOUT_MS = 15_000

/**
 * The windowed cleanup prompt. The request carries two labeled blocks: CONTEXT is earlier
 * speech the model must not clean or return, and TEXT TO CLEAN is the only text it cleans.
 * The reply has no `trigger` field. The trigger phrase stays in the prompt as ordinary words,
 * and the client's matcher decides where the cut falls, so the model never decides the trigger.
 */
export const HEAL_WINDOW_SYSTEM_PROMPT = `You clean one piece of a raw speech transcript. The user message has a labeled TEXT TO CLEAN block and, sometimes, a labeled CONTEXT block before it. Reply with JSON only: {"prompt": "<the cleaned TEXT TO CLEAN>"}. Clean only TEXT TO CLEAN and return only that. CONTEXT is earlier speech that is already handled: do not clean it and do not return it. Use CONTEXT only to settle a filler, a repeat or a false start that crosses the boundary between the two blocks. Keep the trigger phrase ("that is my prompt", "that's my prompt" or a close variant such as "thats my prompt", "that is my problem" or "promt") exactly as spoken wherever it occurs, and keep an unfinished trigger prefix at the end of TEXT TO CLEAN, such as "that is my", exactly as spoken. Never remove the trigger phrase or such a prefix as filler or as a false start. Treat the trigger phrase as ordinary speech, never a reason to stop, to cut the text short or to end early. Remove filler words such as "um", "uh" and "er", remove false starts, and collapse a word or phrase of up to three words repeated immediately, such as "the the", to one copy. Drop a leading discourse marker such as "so", "okay" or "well" from the start of TEXT TO CLEAN only when there is no CONTEXT block. Never change any other wording.`

/** What a windowed cleanup call returns. */
export interface HealWindowResult {
  /** The cleaned TEXT TO CLEAN when `source` is `model`; the raw newest text unchanged when `raw`. */
  prompt: string
  /** `model` when the model answered with a usable prompt; `raw` for every failure. */
  source: 'model' | 'raw'
}

/** Thrown inside the windowed call for any answer the caller must treat as raw. `healWindow` catches it. */
class HealFailure extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'HealFailure'
  }
}

/**
 * Clean the newest speech with one labeled block of earlier speech as context.
 *
 * `context` is the client's fixed window of earlier raw words, empty when there are none.
 * `text` is the newest raw speech, the only text the model cleans and returns. Nothing
 * else from the transcript reaches the request, so the request size is bounded by the window
 * and the newest segment, however long the capture runs.
 *
 * Never throws. Any failure returns `text` with source `raw`, so the caller glues a segment
 * the same way whether or not the model answered. A truncated answer (`finish_reason`
 * `length`) counts as a failure: the output cap cut it off, so it may hold half a sentence,
 * and half a cleaned sentence is not a cleaned one. A valid empty prompt is a success: every
 * word was filler, and the caller must append nothing instead of putting the raw words back.
 */
export async function healWindow(context: string, text: string, options?: { endpoint?: string; timeoutMs?: number }): Promise<HealWindowResult> {
  try {
    const prompt = await callWindowModel(windowUserMessage(context, text), options?.endpoint ?? HEAL_ENDPOINT, options?.timeoutMs ?? HEAL_TIMEOUT_MS)
    return { prompt, source: 'model' }
  } catch {
    return { prompt: text, source: 'raw' }
  }
}

/**
 * The one user message of a windowed call. The labels are what the system prompt refers to:
 * CONTEXT is named as not to be cleaned or returned, so the model cannot mistake earlier
 * speech for the draft. An empty context omits its block, because a labeled CONTEXT block
 * with nothing in it would be an instruction with nothing to settle.
 */
function windowUserMessage(context: string, text: string): string {
  const toClean = `TEXT TO CLEAN (return the cleaned version of exactly this): ${text}`
  if (context === '') return toClean
  return `CONTEXT (earlier speech, do not clean, do not return): ${context}\n${toClean}`
}

/**
 * One POST for a windowed call, and the prompt string from its answer. The request fixes the
 * route, model, temperature, token cap, reasoning effort and the JSON-object response mode;
 * only the messages vary by call. Throws for every answer the caller must treat as raw;
 * `healWindow` turns the throw into the raw result.
 */
async function callWindowModel(userMessage: string, endpoint: string, timeoutMs: number): Promise<string> {
  const requestBody = {
    model: HEAL_MODEL,
    temperature: 0,
    max_tokens: 300,
    reasoning_effort: 'none',
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: HEAL_WINDOW_SYSTEM_PROMPT },
      { role: 'user', content: userMessage },
    ],
  }
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(requestBody),
    signal: AbortSignal.timeout(timeoutMs),
  })
  if (!response.ok) throw new HealFailure(`endpoint responded ${String(response.status)}`)
  const envelope = parseJson(await response.text(), 'response body')
  // Checked before the content is parsed: a cut answer can still be valid JSON with a
  // `prompt` string, and that string would be a sentence the model never finished.
  if (finishReason(envelope) === 'length') throw new HealFailure('output cap reached (finish_reason length)')
  const content = messageContent(envelope)
  if (typeof content !== 'string' || content === '') throw new HealFailure('response has no message content')
  const answer = parseJson(content, 'message content')
  if (!isRecord(answer)) throw new HealFailure('message content is not a JSON object')
  const { prompt } = answer
  if (typeof prompt !== 'string') throw new HealFailure('message content has no string prompt')
  return prompt
}

/** The `finish_reason` of the first choice of a chat-completions body, or `undefined`. */
function finishReason(envelope: unknown): unknown {
  if (!isRecord(envelope) || !Array.isArray(envelope['choices'])) return undefined
  const first: unknown = envelope['choices'][0]
  return isRecord(first) ? first['finish_reason'] : undefined
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
