/**
 * `dsh-talk`, browser half: mounts the `talk` Remote contribution, registers
 * the composer microphone button into `conversation.input.left`, and the
 * settings tab (`settings.plugins.tab`, id `talk`). All data arrives through
 * the `remote.talk` namespace; the draft/send bindings come from the
 * conversation input face resolved for the session scope.
 *
 * @module dsh-talk/client
 */

import type { Context } from '@deepseek-ai/cordis'
// Type-only: declares the client `remote` service (with `$mount`) on the
// cordis Context — the published assembly package that owns the merge on both
// the 0.1.1-rc.2 line and the 0.1.2-alpha checkout.
import type {} from '@deepseek-ai/dsh-api-remotes/client'
import type {} from '@deepseek-ai/dsh-client-locale/client'
// Type-only: pulls the 'settings.plugins.tab' SlotMap declaration into this
// program so the tab registration typechecks against the real declaration.
import type {} from '@deepseek-ai/dsh-client-ui-settings/client'
// Type-only: pulls the 'conversation.input.left' SlotMap declaration in.
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type { RemoteResult } from '@deepseek-ai/dsh-typert-protocol'
import type { IConversation } from '@deepseek-ai/dsh-client-ui-conversation/client'
import type { TalkSettingsTabInjected } from './TalkSettingsTab.tsx'
import { TalkSettingsTab } from './TalkSettingsTab.tsx'
import type { TalkMicInjected } from './TalkMicButton.tsx'
import { TalkMicButton } from './TalkMicButton.tsx'
import { en, zh, type TalkLocaleKey } from './locales.ts'
import { TALK_REMOTE } from './remote.ts'
import { installTalkStyles } from './styles.ts'
import type {
  TalkAudio,
  TalkHeal,
  TalkInterruptResult,
  TalkLatest,
  TalkSettingsInput,
  TalkSettingsResult,
  TalkStatus,
  TalkTranscript,
} from '../wire.ts'

export type { TalkMicInjected, TalkMicProps } from './TalkMicButton.tsx'
export type { TalkSettingsTabInjected, TalkSettingsTabProps } from './TalkSettingsTab.tsx'
export type { TalkLocaleKey } from './locales.ts'
export { foldMic, initMic, clampRecordSeconds, type MicEvent, type MicPhase, type MicViewModel } from './present.ts'

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface LocaleNamespaceMap {
    /** Voice-loop settings copy. */
    'settings.talk': TalkLocaleKey
  }
}

/** Structural face of one observable snapshot (the host's SnapshotStore shape). */
interface SnapshotFace<T> {
  getSnapshot(): T
  subscribe(listener: () => void): () => void
}

/** The session binding fields the stop control reads: the running flag of the session. */
interface SessionBindingFace {
  session: SnapshotFace<{ running?: boolean }>
}

/** The session service faces the stop control reads, declared locally like the other faces here. */
interface TurnSessionsFace {
  scope?: (id: unknown) => unknown
  binding?: (id: unknown) => unknown
}

/** The conversation assembly face that publishes a session's open turn. */
interface UiConversationFace {
  binding(source: SessionBindingFace): { openTurn: SnapshotFace<number | undefined> }
}

/** Dictionary namespace owned by this plugin. */
export const NS = 'settings.talk'

/** Plugin name: matches the package name, the graph row id, and the bundle id. */
export const name = 'dsh-talk'

/** Services the browser half reads; `remote.talk` appears once the contribution mounts. */
export const inject = ['slots', 'locale', 'remote']

/**
 * Minimal structural contract of the slot registry this client uses. Declared
 * locally because the registry's owning package differs across host lines
 * (the removed client runtime on 0.1.1-rc.2, the UI renderer on 0.1.2-alpha);
 * the runtime contract is structural.
 */
interface SlotsFace {
  inject(slot: string, factory: () => unknown): unknown
  register(options: Record<string, unknown>, component: unknown): unknown
}

/**
 * Browser plugin body: dictionaries, the scoped stylesheet, the Remote
 * contribution mount, the mic-button slot, and the settings tab.
 *
 * @param ctx - client root context.
 */
export async function apply(ctx: Context): Promise<void> {
  ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'dsh-talk: dictionaries')
  ctx.effect(() => installTalkStyles(), 'dsh-talk: stylesheet')

  // $mount registers the 'remote.talk' namespace service and owns its removal.
  await ctx.remote.$mount(TALK_REMOTE)

  const slots = ctx.get('slots') as SlotsFace
  ctx.inject(['remote.talk', 'conversation'], (scope) => {
    const talk = scope.remote.talk
    const t = scope.locale.bind(NS)
    const unwrap = <T>(result: RemoteResult<T>, method: string): T => {
      if (!result.ok) {
        throw new Error(`talk.${method} failed: ${result.error.code}: ${result.error.message}`)
      }
      return result.value
    }
    const status: TalkMicInjected['status'] = async () =>
      unwrap<TalkStatus>(await talk.status(), 'status')
    const interrupt: TalkMicInjected['interrupt'] = async () =>
      unwrap<TalkInterruptResult>(await talk.interrupt(), 'interrupt')
    const transcribe: TalkMicInjected['transcribe'] = async (audioData) =>
      unwrap<TalkTranscript>(await talk.transcribe(audioData), 'transcribe')
    const heal: TalkMicInjected['heal'] = async (rawTranscript) =>
      unwrap<TalkHeal>(await talk.heal(rawTranscript), 'heal')
    const audio: TalkMicInjected['audio'] = async (utteranceId) =>
      unwrap<TalkAudio | null>(await talk.audio(utteranceId), 'audio')
    // The poll's signal reaches the RPC request, so a timed-out read is cancelled.
    const latest: TalkMicInjected['latest'] = async (sessionId, signal?: AbortSignal) =>
      unwrap<TalkLatest | null>(await talk.latest(sessionId, signal), 'latest')
    const conversation = scope.get('conversation') as IConversation | undefined
    // Master's sanctioned id → session-scope-ctx exchange (same pattern
    // ui-commands uses); resolves to undefined when either service is
    // unavailable (rc.8 builds without the scope exchange).
    const sessions = scope.get('sessions') as unknown as TurnSessionsFace | undefined
    const forSession: NonNullable<TalkMicInjected['forSession']> = (id) => {
      if (typeof id !== 'string') return undefined
      if (conversation === undefined || sessions?.scope === undefined) return undefined
      const actx = sessions.scope(id)
      if (actx === undefined || actx === null) return undefined
      return conversation.input.for(actx as Parameters<typeof conversation.input.for>[0])
    }
    // The stop control's turn signal. A session has a running turn when its session
    // binding reports running and its conversation publishes an open turn: the same
    // two reads the built-in Escape stop makes. Absent services mean no turn is ever
    // reported running, so the turn stage stays off.
    const uiConversation = scope.get('uiConversation') as unknown as UiConversationFace | undefined
    const watchTurn: NonNullable<TalkMicInjected['watchTurn']> = (id, listener) => {
      if (typeof id !== 'string' || uiConversation === undefined || sessions?.binding === undefined) return () => {}
      try {
        const binding = sessions.binding(id) as SessionBindingFace | undefined
        if (binding === undefined) return () => {}
        const openTurn = uiConversation.binding(binding).openTurn
        const notify = (): void => listener(binding.session.getSnapshot().running === true && openTurn.getSnapshot() !== undefined)
        const disposers = [binding.session.subscribe(notify), openTurn.subscribe(notify)]
        notify()
        return () => {
          for (const dispose of disposers) dispose()
        }
      } catch {
        return () => {}
      }
    }
    // The built-in stop button's cancel: the session's scoped conversation service, which cancels the in-flight turn and keeps the queue.
    const cancelTurn: NonNullable<TalkMicInjected['cancelTurn']> = async (id) => {
      const scoped = sessions?.scope?.(id) as { get(name: string): unknown } | null | undefined
      const scopedConversation = scoped?.get('conversation') as IConversation | undefined
      if (scopedConversation === undefined) throw new Error('conversation service unavailable through the session scope')
      await scopedConversation.cancel()
    }
    const setDraft: TalkMicInjected['setDraft'] = (text) => {
      conversation?.input.for(scope).setDraft(text)
    }
    const send: TalkMicInjected['send'] = async (text) => {
      if (conversation === undefined) return
      conversation.input.for(scope).setDraft(text)
      conversation.input.for(scope).submit()
    }
    const applySettings: TalkSettingsTabInjected['applySettings'] = async (settings: TalkSettingsInput) =>
      unwrap<TalkSettingsResult>(await talk.applySettings(settings), 'applySettings')

    slots.inject('conversation.input.left', () => slots.register({
      name: 'conversation.input.left',
      id: 'talk-mic',
      order: 20,
      inject: (): TalkMicInjected => ({ status, interrupt, transcribe, heal, audio, latest, setDraft, send, forSession, watchTurn, cancelTurn }),
    }, TalkMicButton))

    slots.inject('settings.plugins.tab', () => slots.register({
      name: 'settings.plugins.tab',
      id: 'talk',
      order: 50,
      label: () => t('tab'),
      locale: NS,
      inject: (): TalkSettingsTabInjected => ({ status, applySettings }),
    }, TalkSettingsTab))
  })
}
