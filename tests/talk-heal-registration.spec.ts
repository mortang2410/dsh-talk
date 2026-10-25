/**
 * The heal invocation across the client registration and the host service. The descriptor
 * declares `context` then `text`; the real client registration (`apply` in src/client/index.ts)
 * passes both to the real TalkService, and the one model request that results carries them as
 * the two labeled blocks of the windowed prompt.
 *
 * The model is a loopback stub. `src/heal.ts` is wrapped so the service's windowed call goes to
 * that stub instead of the Bifrost gateway, so no test reaches port 8090.
 *
 * Each test states, in a `Breaks if:` line, the defect that would make it fail.
 *
 * @module dsh-talk/test/talk-heal-registration.spec
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import type { RemoteResult } from '@deepseek-ai/dsh-typert-protocol'
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { apply } from '../src/client/index.ts'
import type { TalkMicInjected } from '../src/client/TalkMicButton.tsx'
import { HEAL_WINDOW_SYSTEM_PROMPT } from '../src/heal.ts'
import { TALK_HEAL_DESCRIPTOR, TALK_INVOCATIONS, type TalkHeal } from '../src/wire.ts'
import { mountHarness } from './harness.ts'

const stub = vi.hoisted(() => ({ endpoint: '' }))

vi.mock('../src/heal.ts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/heal.ts')>()
  return {
    ...actual,
    healWindow: (context: string, text: string) => actual.healWindow(context, text, { endpoint: stub.endpoint }),
  }
})

/** Every request body the loopback model stub received, in order. */
const seen: Array<Record<string, unknown>> = []
/** The assistant message content the stub answers with. */
let reply = ''
const servers: Server[] = []

beforeAll(async () => {
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => chunks.push(chunk))
    req.on('end', () => {
      seen.push(JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>)
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: reply } }] }))
    })
  })
  servers.push(server)
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('stub has no TCP port')
  stub.endpoint = `http://127.0.0.1:${String(address.port)}/v1/chat/completions`
})

afterAll(async () => {
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

beforeEach(() => {
  seen.splice(0)
  reply = '{"prompt": "delta epsilon"}'
})

interface Pending {
  keys: string[]
  run: (scope: { get(name: string): unknown; remote: unknown; locale: unknown }) => void
}

/**
 * A host context whose `inject` waits for its keys, as cordis does. It records the object the
 * `talk-mic` registration builds. The same stub shape as tests/talk-inject-deps.spec.ts.
 */
function registrationHost() {
  const services = new Map<string, unknown>()
  const pending: Pending[] = []
  let registered: TalkMicInjected | undefined
  const flush = () => {
    for (let i = pending.length - 1; i >= 0; i -= 1) {
      const entry = pending[i]!
      if (entry.keys.every((key) => services.has(key))) {
        pending.splice(i, 1)
        entry.run({
          get: (name: string) => services.get(name),
          remote: { talk: services.get('remote.talk') },
          locale: { bind: () => (key: string) => key },
        })
      }
    }
  }
  const slots = {
    inject: (_slot: string, factory: () => unknown) => factory(),
    register: (options: { id: string; inject?: () => unknown }) => {
      if (options.id === 'talk-mic') registered = options.inject?.() as TalkMicInjected
      return options
    },
  }
  const ctx = {
    effect: () => undefined,
    locale: { register: () => undefined },
    remote: { $mount: async () => undefined },
    get: (key: string) => (key === 'slots' ? slots : undefined),
    inject: (keys: string[], run: Pending['run']) => {
      pending.push({ keys, run })
      flush()
    },
  }
  return {
    ctx,
    provide(name: string, value: unknown) {
      services.set(name, value)
      flush()
    },
    registered: () => registered,
  }
}

/**
 * One `talk/heal` invocation, from the client's positional call to the host service, driven by the
 * real descriptor. This package does not depend on the DSH client Remote proxy or on the host gateway
 * (`@deepseek-ai/dsh-api-gateway`), so this function stands in for both with the parts the descriptor
 * fixes: (1) the positional arguments are named by their descriptor wire fields, as the wire carries
 * them; (2) each field passes its strict codec from the descriptor (`codec.create().parse`); (3) the
 * decoded values reach the service positionally, in descriptor order; (4) the answer passes the
 * descriptor's result codec before it returns as a RemoteResult. Not run here: the gateway's exact-field
 * check, the RPC carrier and the typert-loader registration.
 */
async function dispatchHeal(
  service: { heal: (context: string, text: string) => Promise<TalkHeal> },
  positional: unknown[],
): Promise<RemoteResult<TalkHeal>> {
  const wireArgs: Record<string, unknown> = {}
  TALK_HEAL_DESCRIPTOR.parameters.forEach((parameter, index) => {
    wireArgs[parameter.wire] = positional[index]
  })
  const decoded = TALK_HEAL_DESCRIPTOR.parameters.map((parameter) => parameter.codec.create().parse(wireArgs[parameter.wire]))
  const answer = await service.heal(...(decoded as [string, string]))
  return { ok: true, value: TALK_HEAL_DESCRIPTOR.result.create().parse(answer) }
}

/** Mount the real host and register the real client, with the talk remote routed to the host service. */
async function registeredMic() {
  const harness = await mountHarness()
  const host = registrationHost()
  // The remote call crosses the RPC boundary in production. Here it goes through dispatchHeal, which
  // runs the descriptor's codecs and then calls the real service, so the client's unwrap reads a RemoteResult.
  host.provide('remote.talk', {
    heal: (...positional: unknown[]) => dispatchHeal(harness.service, positional),
  })
  host.provide('conversation', {})
  host.provide('uiConversation', {
    binding: () => ({ openTurn: { getSnapshot: () => ({}), subscribe: () => () => undefined } }),
  })
  host.provide('uiSession', { sessionStatus: { getSnapshot: () => new Map(), subscribe: () => () => undefined } })
  await apply(host.ctx as unknown as Parameters<typeof apply>[0])
  const injected = host.registered()
  if (injected === undefined) throw new Error('talk-mic was not registered')
  return { harness, injected }
}

describe('talk/heal descriptor', () => {
  it('declares context then text as the two parameters, and is a served invocation', () => {
    // Breaks if: a parameter is renamed or its wire name changes (the client sends these names),
    // the order is swapped (the codec then binds the words to the wrong slot), or the descriptor
    // is missing from TALK_INVOCATIONS, so the host never serves it.
    expect(TALK_HEAL_DESCRIPTOR.parameters.map((parameter) => parameter.name)).toEqual(['context', 'text'])
    expect(TALK_HEAL_DESCRIPTOR.parameters.map((parameter) => parameter.wire)).toEqual(['context', 'text'])
    expect(TALK_INVOCATIONS).toContain(TALK_HEAL_DESCRIPTOR)
  })
})

describe('host service validation', () => {
  it('rejects a non-string context before any model request', async () => {
    // Breaks if: a malformed context reaches the model, or the rejection does not name the cause.
    const harness = await mountHarness()
    await expect(harness.service.heal(42 as unknown as string, 'delta epsilon')).rejects.toThrow(
      'dsh-talk: heal context and text must be strings',
    )
    expect(seen).toHaveLength(0)
  })

  it('rejects a non-string text before any model request', async () => {
    // Breaks if: a malformed text reaches the model, or the rejection does not name the cause.
    const harness = await mountHarness()
    await expect(harness.service.heal('alpha beta gamma', undefined as unknown as string)).rejects.toThrow(
      'dsh-talk: heal context and text must be strings',
    )
    expect(seen).toHaveLength(0)
  })
})

describe('client registration: the mic heal reaches the model with both arguments', () => {
  it('passes the context and the text, and the one model request carries both as labeled blocks', async () => {
    // Breaks if: the registration swaps or drops either argument, the service reads only one of
    // them, or the labels and the window words change on the way to the model.
    const { injected } = await registeredMic()
    const result = await injected.heal('alpha beta gamma', 'delta epsilon')
    expect(result).toEqual({ prompt: 'delta epsilon', source: 'model' })
    expect(seen).toHaveLength(1)
    expect(seen[0]).toMatchObject({
      model: 'strata/qwen-strata',
      temperature: 0,
      max_tokens: 300,
      reasoning_effort: 'none',
      response_format: { type: 'json_object' },
    })
    expect(seen[0]?.['messages']).toEqual([
      { role: 'system', content: HEAL_WINDOW_SYSTEM_PROMPT },
      {
        role: 'user',
        content: 'CONTEXT (earlier speech, do not clean, do not return): alpha beta gamma\nTEXT TO CLEAN (return the cleaned version of exactly this): delta epsilon',
      },
    ])
  })

  it('a non-string context is refused by the descriptor codec before any model request', async () => {
    // Breaks if: the dispatch skips the descriptor codec for talk/heal, so a malformed argument reaches the service unchecked. The message is the codec's, not the service's own check.
    const { injected } = await registeredMic()
    await expect(injected.heal(42 as unknown as string, 'delta epsilon')).rejects.toThrow(/expected string/iu)
    expect(seen).toHaveLength(0)
  })

  it('an empty context sends only the TEXT TO CLEAN block', async () => {
    // Breaks if: the registration turns an empty context into a labeled block with nothing in it,
    // which would give the model an instruction with nothing to settle.
    const { injected } = await registeredMic()
    await injected.heal('', 'delta epsilon')
    expect(seen).toHaveLength(1)
    expect(seen[0]?.['messages']).toEqual([
      { role: 'system', content: HEAL_WINDOW_SYSTEM_PROMPT },
      { role: 'user', content: 'TEXT TO CLEAN (return the cleaned version of exactly this): delta epsilon' },
    ])
  })
})
