/**
 * Heal contract: one model call per transcript, the raw transcript logged before the
 * call, the literal-phrase fallback decides only on the two literal forms, and the
 * prompt constant is byte-identical to HEAL-PROMPT.md when DSH_TALK_HEAL_PROMPT names it.
 *
 * Stubs are real loopback HTTP servers, so each request crosses the process boundary
 * the way production does. A stub replays a recorded answer; the model's own phrase
 * judgement is checked only by the live run recorded in research/1cds-build.md.
 *
 * Each test states, in a `Breaks if:` line, the defect that would make it fail.
 *
 * @module dsh-talk/test/heal.spec
 */

import { readFile } from 'node:fs/promises'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { HEAL_ENDPOINT, HEAL_MODEL, HEAL_SYSTEM_PROMPT, HEAL_TIMEOUT_MS, HealFailure, healTranscript } from '../src/heal.ts'

const RAW_EXACT = 'read the config for the ssh socket that is my prompt'
const CLEAN = 'read the config for the ssh socket'

type Reply = (res: ServerResponse) => void

interface Stub {
  endpoint: string
  requests: Array<Record<string, unknown>>
  events: string[]
  reply: Reply
}

const servers: Server[] = []

afterEach(async () => {
  vi.restoreAllMocks()
  await Promise.all(
    servers.splice(0).map(
      (server) =>
        new Promise<void>((resolve) => {
          server.closeAllConnections()
          server.close(() => resolve())
        }),
    ),
  )
})

/** Build a chat-completions body whose assistant message content is `content`. */
function chatBody(content: string): string {
  return JSON.stringify({ choices: [{ message: { role: 'assistant', content } }] })
}

function sendJson(res: ServerResponse, status: number, body: string): void {
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(body)
}

/** A reply that answers 200 with the given assistant message content. */
const answer =
  (content: string): Reply =>
  (res) =>
    sendJson(res, 200, chatBody(content))

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => chunks.push(chunk))
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

/** Start a loopback stub. It records each request body and an ordering event. */
async function startStub(): Promise<Stub> {
  const stub: Stub = {
    endpoint: '',
    requests: [],
    events: [],
    reply: answer('{"trigger": false, "prompt": ""}'),
  }
  const server = createServer((req, res) => {
    void readBody(req).then((raw) => {
      stub.events.push('request')
      stub.requests.push(JSON.parse(raw) as Record<string, unknown>)
      stub.reply(res)
    })
  })
  servers.push(server)
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('stub has no TCP port')
  stub.endpoint = `http://127.0.0.1:${String(address.port)}/v1/chat/completions`
  return stub
}

/** An endpoint on a loopback port with nothing listening, so the connection is refused. */
async function closedEndpoint(): Promise<string> {
  const server = createServer()
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('no TCP port')
  const port = address.port
  await new Promise<void>((resolve) => server.close(() => resolve()))
  return `http://127.0.0.1:${String(port)}/v1/chat/completions`
}

/** Heal against the stub while muting the default console logger. */
function heal(raw: string, endpoint: string, timeoutMs?: number) {
  return healTranscript(raw, {
    endpoint,
    logRawTranscript: () => undefined,
    ...(timeoutMs === undefined ? {} : { timeoutMs }),
  })
}

describe('contract constants', () => {
  it('points at the resident bonsai2-27b endpoint named in SPEC.md', () => {
    // Breaks if: the endpoint, model, or port changes without the spec changing with it.
    expect(HEAL_ENDPOINT).toBe('http://127.0.0.1:8082/v1/chat/completions')
    expect(HEAL_MODEL).toBe('bonsai2-27b')
  })

  it('keeps the timeout above the 8.58 s outlier in research/1cds-heal-recheck.md', () => {
    // Breaks if: someone lowers the timeout below the one measured idle-gap call.
    expect(HEAL_TIMEOUT_MS).toBeGreaterThan(8_580)
  })
})

describe('model path: one call returns trigger and prompt from one response', () => {
  it('returns the model decision with source model and makes exactly one request', async () => {
    // Breaks if: the code drops the model's prompt, re-derives trigger from the raw text,
    // or retries after a good answer.
    const stub = await startStub()
    stub.reply = answer(JSON.stringify({ trigger: true, prompt: CLEAN }))
    const result = await heal(RAW_EXACT, stub.endpoint)
    expect(result).toEqual({ trigger: true, prompt: CLEAN, source: 'model' })
    expect(stub.requests).toHaveLength(1)
  })

  // The six recorded outputs in HEAL-PROMPT.md, replayed as stub answers. This checks that
  // the caller passes each recorded decision through unchanged.
  // Breaks if: the caller overrides the model's trigger, or strips text on the model path.
  it.each([
    ['exact phrase', `${CLEAN} that is my prompt`, true, CLEAN],
    ['contracted, no apostrophe', `${CLEAN} thats my prompt`, true, CLEAN],
    ['misheard promt', `${CLEAN} that is my promt`, true, CLEAN],
    ['different word problem', `${CLEAN} that is my problem`, false, `${CLEAN} that is my problem`],
    ['different word proper', `${CLEAN} that is my proper`, false, `${CLEAN} that is my proper`],
    ['phrase followed by speech', 'that is my prompt now explain the vad', false, 'that is my prompt now explain the vad'],
  ])('%s: the recorded model answer reaches the caller as-is', async (_label, raw, trigger, prompt) => {
    const stub = await startStub()
    stub.reply = answer(JSON.stringify({ trigger, prompt }))
    const result = await heal(raw, stub.endpoint)
    expect(result).toEqual({ trigger, prompt, source: 'model' })
    expect(stub.requests).toHaveLength(1)
  })
})

describe('request shape and ordering', () => {
  it('sends the contracted request: reasoning off, json_object, temperature 0, prompt and transcript as the two messages', async () => {
    // Breaks if: reasoning_effort is dropped (reasoning tokens then consume the budget and
    // content comes back empty), response_format changes, or the prompt is not the constant.
    const stub = await startStub()
    stub.reply = answer('{"trigger": false, "prompt": "x"}')
    await heal(RAW_EXACT, stub.endpoint)
    const body = stub.requests[0] ?? {}
    expect(body).toMatchObject({
      model: 'bonsai2-27b',
      temperature: 0,
      max_tokens: 300,
      reasoning_effort: 'none',
      response_format: { type: 'json_object' },
    })
    expect(body['messages']).toEqual([
      { role: 'system', content: HEAL_SYSTEM_PROMPT },
      { role: 'user', content: `raw transcript: ${RAW_EXACT}` },
    ])
  })

  it('logs the raw transcript before the request is sent', async () => {
    // Breaks if: the log call moves after fetch, or logs a different string than the input.
    const stub = await startStub()
    stub.reply = answer('{"trigger": false, "prompt": "x"}')
    await healTranscript(RAW_EXACT, {
      endpoint: stub.endpoint,
      logRawTranscript: (raw) => stub.events.push(`log:${raw}`),
    })
    expect(stub.events).toEqual([`log:${RAW_EXACT}`, 'request'])
  })

  it('logs the raw transcript verbatim through the default console logger', async () => {
    // Breaks if: the default logger is removed or the transcript is altered before logging.
    const spy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    const stub = await startStub()
    stub.reply = answer('{"trigger": false, "prompt": "x"}')
    await healTranscript(RAW_EXACT, { endpoint: stub.endpoint })
    expect(spy).toHaveBeenCalledWith('dsh-talk heal raw transcript:', RAW_EXACT)
  })
})

describe('fallback path: the call failed, so the two literal forms decide', () => {
  // Each reply below is a failure shape. Every one must fall back, make exactly one request,
  // and never let the bad model answer reach the caller.
  // Breaks if: the failure throws instead of falling back, a second request is sent, or the
  // malformed `trigger` string is accepted as a decision.
  it.each([
    ['HTTP 500', (res: ServerResponse) => sendJson(res, 500, 'boom')],
    ['malformed JSON in content', answer('{"trigger": tru')],
    ['empty content (the reasoning-on failure shape)', answer('')],
    ['non-JSON body', (res: ServerResponse) => res.end('<html>not json</html>')],
    ['trigger as the string "true"', answer('{"trigger": "true", "prompt": "WRONG"}')],
    ['prompt not a string', answer('{"trigger": true, "prompt": 3}')],
    ['content not an object', answer('[1, 2]')],
  ])('%s falls back to the literal form and strips it', async (_label, reply) => {
    const stub = await startStub()
    stub.reply = reply
    const result = await heal(`read the config that is my prompt`, stub.endpoint)
    expect(result).toEqual({ trigger: true, prompt: 'read the config', source: 'fallback' })
    expect(stub.requests).toHaveLength(1)
  })

  it('times out at the configured limit and falls back', async () => {
    // Breaks if: the abort signal is missing (the call would wait the full 2 s), or a timeout
    // is not treated as a failure that falls back.
    const stub = await startStub()
    stub.reply = (res) => {
      const timer = setTimeout(() => {
        if (!res.destroyed) sendJson(res, 200, chatBody('{"trigger": false, "prompt": "late"}'))
      }, 2_000)
      res.on('close', () => clearTimeout(timer))
    }
    const started = Date.now()
    const result = await heal('read the config that is my prompt', stub.endpoint, 150)
    expect(result).toEqual({ trigger: true, prompt: 'read the config', source: 'fallback' })
    expect(Date.now() - started).toBeLessThan(1_500)
  })

  it('connection refused falls back to the literal form', async () => {
    // Breaks if: a network error escapes healTranscript instead of falling back.
    const result = await heal('read the config that is my prompt', await closedEndpoint())
    expect(result).toEqual({ trigger: true, prompt: 'read the config', source: 'fallback' })
  })

  it.each([
    ['apostrophe form', "read the config that's my prompt", 'read the config'],
    ['capitalised, with a sentence mark', 'Read the config. That is my prompt.', 'Read the config.'],
  ])('the literal form (%s) is stripped and what is left is sent', async (_label, raw, prompt) => {
    // Breaks if: the case-insensitive match or the sentence-mark removal is lost.
    const result = await heal(raw, await closedEndpoint())
    expect(result).toEqual({ trigger: true, prompt, source: 'fallback' })
  })

  it.each([
    ['misheard promt', 'read the config that is my promt'],
    ['different word problem', 'read the config that is my problem'],
    ['different word proper', 'read the config that is my proper'],
    ['no apostrophe (not one of the two literal forms)', 'read the config thats my prompt'],
    ['prompted (word boundary)', 'read the config that is my prompted'],
    ['no phrase at all', 'read the config'],
  ])('a fallback miss (%s) keeps the capture open by throwing HealFailure', async (_label, raw) => {
    // Breaks if: the fallback matches a substring such as "prompt" or "promt", or a miss
    // returns a prompt instead of throwing.
    await expect(heal(raw, await closedEndpoint())).rejects.toBeInstanceOf(HealFailure)
  })

  it('KNOWN CONFLICT, pinned: the fallback fires when the phrase is followed by more speech', async () => {
    // SPEC.md says the fallback is a plain substring check. The model path refuses a phrase
    // followed by speech. This test pins the current fallback behaviour so the conflict is
    // visible; it is an overseer decision, not a settled rule.
    // Breaks if: the fallback is changed to end-anchoring without a decision on this conflict.
    const result = await heal('that is my prompt now explain the vad', await closedEndpoint())
    expect(result).toEqual({ trigger: true, prompt: 'now explain the vad', source: 'fallback' })
  })
})

describe('byte identity with HEAL-PROMPT.md', () => {
  // Runs only when DSH_TALK_HEAL_PROMPT names the file. Unset, the test is reported as skipped.
  const healPromptFile = process.env.DSH_TALK_HEAL_PROMPT ?? ''

  it.skipIf(healPromptFile === '')(
    'HEAL_SYSTEM_PROMPT equals the fenced block under "## System prompt, verbatim", byte for byte',
    async () => {
      // Breaks if: one character of the constant differs from the file, the heading or fence
      // moves, or the constant is edited without HEAL-PROMPT.md.
      const markdown = await readFile(healPromptFile, 'utf8')
      const headingAt = markdown.indexOf('## System prompt, verbatim')
      expect(headingAt, 'heading present').toBeGreaterThanOrEqual(0)
      const opening = markdown.indexOf('\n```\n', headingAt)
      expect(opening, 'opening fence after the heading').toBeGreaterThanOrEqual(0)
      const contentStart = opening + '\n```\n'.length
      const closing = markdown.indexOf('\n```', contentStart)
      expect(closing, 'closing fence').toBeGreaterThan(contentStart)
      const block = markdown.slice(contentStart, closing)
      expect(Buffer.from(block, 'utf8').equals(Buffer.from(HEAL_SYSTEM_PROMPT, 'utf8'))).toBe(true)
    },
  )
})
