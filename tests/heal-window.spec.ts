/**
 * Windowed cleanup contract (ticket 02a): `healWindow` sends one user message with a labeled
 * CONTEXT block (only when context is non-empty) and a labeled TEXT TO CLEAN block, returns the
 * model's prompt with source `model`, and returns the raw newest text with source `raw` for
 * every failure. This file is the request and answer contract of the one heal call; the byte
 * identity of the prompt constant is in `tests/heal.spec.ts`.
 *
 * Stubs are real loopback HTTP servers, the same pattern as `tests/heal.spec.ts`, so each
 * request crosses the process boundary the way production does. No test reaches the Bifrost
 * gateway or a model.
 *
 * Each test states, in a `Breaks if:` line, the defect that would make it fail.
 *
 * @module dsh-talk/test/heal-window.spec
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { afterEach, describe, expect, it } from 'vitest'
import { HEAL_MODEL, HEAL_WINDOW_SYSTEM_PROMPT, healWindow } from '../src/heal.ts'

const CONTEXT = 'so I opened the ssh socket for the agent'
const TEXT = 'read the config that is my prompt'

type Reply = (res: ServerResponse) => void

interface Stub {
  endpoint: string
  requests: Array<Record<string, unknown>>
  reply: Reply
}

const servers: Server[] = []

afterEach(async () => {
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

/** Build a chat-completions body whose first choice has `content` and `finishReason`. */
function chatBody(content: string, finishReason = 'stop'): string {
  return JSON.stringify({ choices: [{ finish_reason: finishReason, message: { role: 'assistant', content } }] })
}

function sendJson(res: ServerResponse, status: number, body: string): void {
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(body)
}

/** A reply that answers 200 with the given assistant message content and finish reason. */
const answer =
  (content: string, finishReason = 'stop'): Reply =>
  (res) =>
    sendJson(res, 200, chatBody(content, finishReason))

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => chunks.push(chunk))
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

/** Start a loopback stub. It records each request body and then sends its reply. */
async function startStub(): Promise<Stub> {
  const stub: Stub = {
    endpoint: '',
    requests: [],
    reply: answer('{"prompt": ""}'),
  }
  const server = createServer((req, res) => {
    void readBody(req).then((raw) => {
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

/** Clean against an endpoint, passing a timeout only when the test sets one. */
function clean(context: string, text: string, endpoint: string, timeoutMs?: number) {
  return healWindow(context, text, timeoutMs === undefined ? { endpoint } : { endpoint, timeoutMs })
}

describe('request: one user message with labeled blocks', () => {
  it('context present: one user message carries the CONTEXT block, one newline, then the TEXT TO CLEAN block', async () => {
    // Breaks if: a block loses its label, the two blocks swap order, the separator is not a single
    // newline, or the context words are altered before they reach the model.
    const stub = await startStub()
    stub.reply = answer('{"prompt": "read the config"}')
    await clean(CONTEXT, TEXT, stub.endpoint)
    expect(stub.requests).toHaveLength(1)
    expect(stub.requests[0]?.['messages']).toEqual([
      { role: 'system', content: HEAL_WINDOW_SYSTEM_PROMPT },
      {
        role: 'user',
        content: `CONTEXT (earlier speech, do not clean, do not return): ${CONTEXT}\nTEXT TO CLEAN (return the cleaned version of exactly this): ${TEXT}`,
      },
    ])
  })

  it('context absent: the user message carries only the TEXT TO CLEAN block', async () => {
    // Breaks if: an empty CONTEXT block is sent (a labeled instruction with nothing to settle),
    // or the TEXT TO CLEAN label or the text it carries changes.
    const stub = await startStub()
    stub.reply = answer('{"prompt": "read the config"}')
    await clean('', TEXT, stub.endpoint)
    expect(stub.requests).toHaveLength(1)
    expect(stub.requests[0]?.['messages']).toEqual([
      { role: 'system', content: HEAL_WINDOW_SYSTEM_PROMPT },
      { role: 'user', content: `TEXT TO CLEAN (return the cleaned version of exactly this): ${TEXT}` },
    ])
  })

  it('sends the route-facing fields the cleanup is measured with', async () => {
    // Breaks if: the model alias, temperature, the 300-token cap, reasoning effort, or the
    // JSON-object mode changes, because the measured timing and the answer shape depend on each.
    const stub = await startStub()
    await clean(CONTEXT, TEXT, stub.endpoint)
    expect(stub.requests[0]).toMatchObject({
      model: HEAL_MODEL,
      temperature: 0,
      max_tokens: 300,
      reasoning_effort: 'none',
      response_format: { type: 'json_object' },
    })
  })
})

describe('a normal answer', () => {
  it('returns the model prompt with source model after exactly one request', async () => {
    // Breaks if: the answer is replaced by the raw text, the source is wrong, or a second request is sent.
    const stub = await startStub()
    stub.reply = answer(JSON.stringify({ prompt: 'read the config for the ssh socket' }))
    expect(await clean(CONTEXT, TEXT, stub.endpoint)).toEqual({ prompt: 'read the config for the ssh socket', source: 'model' })
    expect(stub.requests).toHaveLength(1)
  })

  it('returns the prompt as-is: a trigger phrase left in it is not removed by the host', async () => {
    // Breaks if: the host strips or cuts the phrase. The cut belongs to the client's matcher, so the
    // windowed call must hand back what the model wrote.
    const stub = await startStub()
    stub.reply = answer(JSON.stringify({ prompt: 'read the config that is my prompt' }))
    expect(await clean('', TEXT, stub.endpoint)).toEqual({ prompt: 'read the config that is my prompt', source: 'model' })
  })

  it('ignores a trigger field in the answer, so the result has no trigger key', async () => {
    // Breaks if: the windowed path reads or returns a trigger decision. The result type has no trigger.
    const stub = await startStub()
    stub.reply = answer(JSON.stringify({ trigger: true, prompt: 'read the config' }))
    const result = await clean(CONTEXT, TEXT, stub.endpoint)
    expect(result).toEqual({ prompt: 'read the config', source: 'model' })
    expect(result).not.toHaveProperty('trigger')
  })

  it('a valid empty prompt is returned as-is with source model, not replaced by the raw text', async () => {
    // Breaks if: an empty prompt is treated as a failure and the raw words are substituted, which
    // would put back filler the model removed on purpose. The empty prompt is also not retried.
    const stub = await startStub()
    stub.reply = answer(JSON.stringify({ prompt: '' }))
    expect(await clean(CONTEXT, 'um uh', stub.endpoint)).toEqual({ prompt: '', source: 'model' })
    expect(stub.requests).toHaveLength(1)
  })
})

describe('failures return the raw text with source raw, without throwing', () => {
  // Each reply is a failure shape. Every one must return the raw newest text, send exactly one request,
  // and never let the bad answer reach the caller.
  it.each([
    ['malformed JSON in content', answer('{"prompt": "read the')],
    ['prompt is a number', answer('{"prompt": 3}')],
    ['prompt is null', answer('{"prompt": null}')],
    ['prompt is an array', answer('{"prompt": ["read the config"]}')],
    ['content is a JSON array, not an object', answer('["read the config"]')],
    ['empty message content', answer('')],
    ['HTTP 500', (res: ServerResponse) => sendJson(res, 500, 'boom')],
    ['response body is not JSON', (res: ServerResponse) => res.end('<html>not json</html>')],
  ])('%s returns the raw text', async (_label, reply) => {
    // Breaks if: the failure throws instead of returning the raw text, a second request is sent,
    // or a bad answer's field reaches the caller in place of the raw newest text.
    const stub = await startStub()
    stub.reply = reply
    expect(await clean(CONTEXT, TEXT, stub.endpoint)).toEqual({ prompt: TEXT, source: 'raw' })
    expect(stub.requests).toHaveLength(1)
  })

  it('a timeout returns the raw text', async () => {
    // Breaks if: the abort signal is missing (the call would wait the full 2 s stub delay), or a
    // timeout is not treated as a failure that returns the raw text.
    const stub = await startStub()
    stub.reply = (res) => {
      const timer = setTimeout(() => {
        if (!res.destroyed) sendJson(res, 200, chatBody('{"prompt": "late"}'))
      }, 2_000)
      res.on('close', () => clearTimeout(timer))
    }
    const started = Date.now()
    expect(await clean(CONTEXT, TEXT, stub.endpoint, 150)).toEqual({ prompt: TEXT, source: 'raw' })
    expect(Date.now() - started).toBeLessThan(1_500)
  })

  it('a refused connection returns the raw text', async () => {
    // Breaks if: a network error escapes healWindow instead of returning the raw text.
    expect(await clean(CONTEXT, TEXT, await closedEndpoint())).toEqual({ prompt: TEXT, source: 'raw' })
  })

  it.each([
    ['complete JSON with a string prompt', '{"prompt": "read the config for the"}'],
    ['cut mid-JSON', '{"prompt": "read the'],
  ])('a truncated answer (finish_reason length, %s) returns the raw text', async (_label, content) => {
    // Breaks if: finish_reason is ignored, so a sentence cut by the 300-token cap is taken as cleaned.
    // The first row is valid JSON with a string prompt, so only the finish-reason check can refuse it.
    const stub = await startStub()
    stub.reply = answer(content, 'length')
    expect(await clean(CONTEXT, TEXT, stub.endpoint)).toEqual({ prompt: TEXT, source: 'raw' })
    expect(stub.requests).toHaveLength(1)
  })
})

describe('the windowed prompt constant states its rules', () => {
  it.each([
    ['the reply shape', '{"prompt": "<the cleaned TEXT TO CLEAN>"}'],
    ['JSON only', 'Reply with JSON only'],
    ['clean and return only TEXT TO CLEAN', 'Clean only TEXT TO CLEAN and return only that.'],
    ['CONTEXT is never cleaned or returned', 'do not clean it and do not return it.'],
    ['CONTEXT settles only the boundary cases', 'Use CONTEXT only to settle a filler, a repeat or a false start that crosses the boundary'],
    ['an unfinished trigger prefix stays at the end', 'keep an unfinished trigger prefix at the end of TEXT TO CLEAN'],
    ['the trigger phrase is kept as spoken', 'exactly as spoken'],
    ['the trigger phrase is never removed as filler or a false start', 'Never remove the trigger phrase or such a prefix as filler or as a false start.'],
    ['the trigger phrase is ordinary speech, never a reason to stop', 'Treat the trigger phrase as ordinary speech, never a reason to stop'],
    ['the leading discourse-marker rule applies only without CONTEXT', 'only when there is no CONTEXT block'],
    ['the filler rule', 'Remove filler words such as "um", "uh" and "er"'],
  ])('states %s', (_label, rule) => {
    // Breaks if: the rule is dropped or reworded so the model is no longer told that rule, which
    // would let it clean CONTEXT, remove the trigger phrase, or drop the wrong marker.
    expect(HEAL_WINDOW_SYSTEM_PROMPT).toContain(rule)
  })

  it('ends with the never-change clause', () => {
    // Breaks if: the prompt lets the model change wording beyond the listed edits.
    expect(HEAL_WINDOW_SYSTEM_PROMPT.endsWith('Never change any other wording.')).toBe(true)
  })

  it('asks for no trigger field', () => {
    // Breaks if: the windowed prompt reintroduces the model-decided trigger. The client's matcher
    // decides where the prompt ends, so a trigger field would be a second, unread decision.
    expect(HEAL_WINDOW_SYSTEM_PROMPT).not.toContain('"trigger"')
  })
})
