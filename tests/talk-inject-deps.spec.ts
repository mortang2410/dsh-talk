/**
 * The browser plugin's inject list. Cordis runs the `ctx.inject` callback once
 * every listed service exists, so a service that arrives late must not leave
 * the stop control's turn guard with undefined constants for the whole mount.
 * The stub below models that wait (the callback runs only when every key is
 * present, and again when the last one is provided). It checks the declared
 * list and the late-arrival path. It does not run cordis itself, so the real
 * wait is still unverified here; see the report for the live check.
 *
 * @module dsh-talk/test/talk-inject-deps.spec
 */

import { describe, expect, it } from 'vitest'
import { apply } from '../src/client/index.ts'
import type { TalkMicInjected } from '../src/client/TalkMicButton.tsx'

interface Pending {
  keys: string[]
  run: (scope: { get(name: string): unknown; remote: unknown; locale: unknown }) => void
}

/** A host context whose `inject` waits for its keys, like cordis. */
function waitingHost() {
  const services = new Map<string, unknown>()
  const pending: Pending[] = []
  let registered: TalkMicInjected | undefined
  const scopeFor = (keys: string[]) => ({
    get: (name: string) => services.get(name),
    remote: { talk: services.get('remote.talk') },
    locale: { bind: () => (key: string) => key },
    keys,
  })
  const flush = () => {
    for (let i = pending.length - 1; i >= 0; i -= 1) {
      const entry = pending[i]!
      if (entry.keys.every((key) => services.has(key))) {
        pending.splice(i, 1)
        entry.run(scopeFor(entry.keys))
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
    declared: () => pending[0]?.keys ?? [],
    provide(name: string, value: unknown) {
      services.set(name, value)
      flush()
    },
    registered: () => registered,
  }
}

/** A cordis-like turn fixture: one running session with an open turn. */
function turnServices() {
  const noop = () => () => undefined
  const session = { running: true, removed: false, subagent: null }
  return {
    'remote.talk': { latest: async () => ({ ok: true, value: null }) },
    conversation: {},
    sessions: {
      binding: () => ({ session: { getSnapshot: () => session, subscribe: noop } }),
      scope: () => null,
    },
    uiConversation: {
      binding: () => ({ openTurn: { getSnapshot: () => ({}), subscribe: noop } }),
    },
  }
}

describe('talk plugin inject list', () => {
  it('declares the turn guard services as hard dependencies', async () => {
    const host = waitingHost()
    await apply(host.ctx as unknown as Parameters<typeof apply>[0])
    expect(host.declared()).toEqual(['remote.talk', 'conversation', 'uiSession', 'uiConversation'])
  })

  it('registers no mic button while uiSession is absent, then registers it and reports a running turn once it arrives', async () => {
    const host = waitingHost()
    for (const [name, value] of Object.entries(turnServices())) {
      if (name !== 'uiConversation') host.provide(name, value)
    }
    await apply(host.ctx as unknown as Parameters<typeof apply>[0])
    expect(host.registered()).toBeUndefined()

    host.provide('uiConversation', turnServices().uiConversation)
    expect(host.registered()).toBeUndefined()

    host.provide('uiSession', {
      sessionStatus: { getSnapshot: () => new Map(), subscribe: () => () => undefined },
    })
    const injected = host.registered()
    expect(injected).toBeDefined()
    const seen: boolean[] = []
    const dispose = injected!.watchTurn!('session-late', (cancellable) => seen.push(cancellable))
    expect(seen[seen.length - 1]).toBe(true)
    dispose()
  })
})
