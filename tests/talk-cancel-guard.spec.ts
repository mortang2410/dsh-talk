/**
 * The stop control's cancel refuses at call time, not only at watch time.
 *
 * The watch reports a turn as cancellable through `turnCancellable`, which mirrors the built-in
 * Escape guard. That is a snapshot: between the watch saying "running" and the user pressing,
 * an approval can arrive. `cancelTurn` therefore re-reads the guard before it calls the
 * conversation service. Removing that re-check leaves the whole suite green without this file,
 * which is why it exists.
 *
 * @module dsh-talk/test/talk-cancel-guard.spec
 */

import { describe, expect, it } from 'vitest'
import { apply } from '../src/client/index.ts'
import type { TalkMicInjected } from '../src/client/TalkMicButton.tsx'

const SESSION = 'session-guard'

/** Mount the plugin over a session whose turn is running, and return the injected cancel. */
async function mountCancel(state: { pending: boolean; openTurn: boolean }) {
  const cancels: string[] = []
  let injected: TalkMicInjected | undefined

  const binding = {
    session: {
      getSnapshot: () => ({ running: true, removed: false, subagent: null }),
      subscribe: () => () => undefined,
    },
  }
  const openTurn = {
    getSnapshot: () => (state.openTurn ? {} : undefined),
    subscribe: () => () => undefined,
  }
  const scopedConversation = { cancel: async () => { cancels.push('cancel') } }
  const sessions = {
    binding: () => binding,
    scope: () => ({ get: (name: string) => (name === 'conversation' ? scopedConversation : undefined) }),
  }
  const uiConversation = { binding: () => ({ openTurn }) }
  const uiSession = {
    sessionStatus: {
      getSnapshot: () => new Map([[SESSION, state.pending ? { pendingInteraction: {} } : {}]]),
      subscribe: () => () => undefined,
    },
  }
  const slots = {
    inject: (_slot: string, factory: () => unknown) => factory(),
    register: (options: { id: string; inject?: () => unknown }) => {
      if (options.id === 'talk-mic') injected = options.inject?.() as TalkMicInjected
      return options
    },
  }
  const ctx = {
    effect: () => undefined,
    locale: { register: () => undefined, bind: () => (key: string) => key },
    remote: { $mount: async () => undefined },
    get: (key: string) => {
      if (key === 'slots') return slots
      if (key === 'uiConversation') return uiConversation
      if (key === 'uiSession') return uiSession
      if (key === 'sessions') return sessions
      return undefined
    },
    inject: (_keys: string[], factory: (scope: unknown) => void) => factory({
      remote: { talk: {} },
      locale: { bind: () => (key: string) => key },
      get: (key: string) => (key === 'uiConversation' ? uiConversation : key === 'uiSession' ? uiSession : key === 'sessions' ? sessions : undefined),
    }),
  }
  await apply(ctx as unknown as Parameters<typeof apply>[0])
  if (injected === undefined) throw new Error('talk-mic slot was not registered')
  return { cancelTurn: injected.cancelTurn, cancels }
}

describe('cancelTurn re-reads the guard at call time', () => {
  it('cancels a running turn with no approval pending', async () => {
    const { cancelTurn, cancels } = await mountCancel({ pending: false, openTurn: true })
    await cancelTurn?.(SESSION)
    expect(cancels).toEqual(['cancel'])
  })

  it('refuses when an approval arrived after the watch reported running', async () => {
    // Breaks if: the call-time re-check is removed, so a press or a barge-in cancels the turn the
    // approval belongs to.
    const { cancelTurn, cancels } = await mountCancel({ pending: true, openTurn: true })
    await cancelTurn?.(SESSION)
    expect(cancels).toEqual([])
  })

  it('refuses when the turn closed between the watch and the press', async () => {
    const { cancelTurn, cancels } = await mountCancel({ pending: false, openTurn: false })
    await cancelTurn?.(SESSION)
    expect(cancels).toEqual([])
  })
})
