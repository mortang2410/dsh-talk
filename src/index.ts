/**
 * `dsh-talk` — the voice-first session loop for DeepSeek Harness.
 *
 * Host half: mounts the `talk` Remote service (speak pipeline + audio cache
 * + settings write-back), registers the `speak` tool, registers the
 * `talk:speech` session-projection unit the web client plays utterances
 * from, and speaks event announcements:
 * - `agent/status` → `idle` after a `running` observation: turn complete.
 * - `agent/error`: a turn errored.
 * - `approval/request` (waterfall): an approval is pending — the listener
 *   announces without blocking and always calls `next()`.
 *
 * Function plugin — no default export (the Loader unwraps
 * `exports.default ?? exports`).
 *
 * @module dsh-talk
 */

import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-tools'
import type {} from '@deepseek-ai/dsh-subprocess'
import type {} from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-session-projection'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { ContextFormed } from '@deepseek-ai/dsh-llm'
import { Config, resolveConfig } from './config.ts'
import {
  applyTalkSpeechProjection,
  initTalkSpeechProjection,
  TALK_SPEECH_PROJECTION_STATE_VERSION,
  talkSpeechProjectionSchema,
  viewTalkSpeechProjection,
} from './projection.ts'
import { TalkService } from './service.ts'
import { speakTool } from './tool.ts'
import { lastReplyText, VOICE_TURN_INSTRUCTION, voiceReplyText } from './voice-turn.ts'

// The voice-turn instruction is a user-role message with its own source kind, declared the way
// other producers declare theirs (dsh-user-approval). Its form is a snapshot with one section.
declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    'dsh-talk-voice-turn': { kind: 'dsh-talk-voice-turn' } & ContextFormed
  }
}

export const name = 'talk'

/** Hard services: the tool registry and the subprocess service. */
export const inject = ['tools', 'subprocess']

// Service Definition — public contract: the talk service face, the speak tool, and the wire vocabularies.
export { Config, resolveConfig, type Config as TalkConfig, type ResolvedConfig } from './config.ts'
export { TalkService, resolveSttEngine, resolveTtsEngine, type SpeakOptions, type SpeakOutcome } from './service.ts'
export { speakTool, renderSpeak } from './tool.ts'
export { sanitizeText, redactSecrets, displayPath } from './sanitize.ts'
export { edgeTtsArgv, piperSpec, whisperArgv, runCommand, synthesize, transcribeWhisper, transcribeFunasr, EngineFailure } from './engine.ts'
export { appendSettingsFragment, mergeTalkRowConfig, renderSettingsFragment, validateSettingsInput, yamlScalar } from './settings-patch.ts'
export { rmsOf, isSpeechFrame, advanceSilence, silenceTrips } from './client/vad.ts'
export type { SpeechProvider, SpeechSttProvider, SpeechTtsProvider } from './provider.ts'
export type { TalkAudio, TalkInterruptResult, TalkSettingsInput, TalkSettingsResult, TalkStatus, TalkTranscript } from './wire.ts'
export type { DshTalkSpeechEvent, SpeechReason, SpeechTtsEngine } from './speech.ts'
export type { TalkSpeechProjection, TalkSpeechProjectionState } from './projection.ts'

/**
 * Mount the plugin: the talk service, the speak tool, the projection unit,
 * and the three announcement listeners. Every registration is an effect on
 * this fiber, so unload/hot-reload removes the tool, the projection, and all
 * listeners together.
 *
 * @param ctx - context carrying tools + subprocess.
 * @param config - raw loader config; defaults applied through {@link resolveConfig}.
 */
export async function apply(ctx: Context, config: Config): Promise<void> {
  const resolved = resolveConfig(config)

  // Service Provider — registration: mount the talk service and register the speak tool.
  // Mount the talk service: the Service constructor self-registers; the
  // function plugin body must return nothing (a non-disposer return would
  // be rejected as an invalid effect).
  await ctx.plugin(function mountTalkService(ctx: Context): void {
    new TalkService(ctx, config, resolved, 5)
  })
  // A02: mounting the service crosses an await. If the fiber was disposed while
  // it resolved, every registration below would throw INACTIVE_EFFECT and leave
  // a half-mounted plugin behind; bail out instead (a reload runs the whole
  // path again). Scope-borrow check (card step 6, verified in source): the
  // service reads the loader's `baseUrl` at CALL time from the child context
  // `ctx.plugin` created (src/service.ts:450-459) — a plain inherited property,
  // never an effect — and `applySettings` is only reachable through the service
  // this same fiber owns, so the borrowed scope cannot dangle after
  // unload/reload.
  if (ctx.fiber.uid === null) return
  const service = ctx.get('talk') as TalkService

  ctx.effect(() => ctx.tools.register(speakTool(service, resolved.maxSpeakChars)), 'dsh-talk: speak tool')

  // The `talk:speech` projection unit: last-wins fold of spoken utterances.
  // It is no longer the browser's playback trigger (the client polls talk/latest).
  ctx.inject(['sessionProjections'], (projectionCtx) => {
    projectionCtx.effect(() => projectionCtx.sessionProjections.register<'talk:speech', ReturnType<typeof initTalkSpeechProjection>>({
      key: 'talk:speech',
      stateSchema: talkSpeechProjectionSchema,
      init: initTalkSpeechProjection,
      apply: applyTalkSpeechProjection,
      wire: {
        viewSchema: talkSpeechProjectionSchema,
        view: viewTalkSpeechProjection,
      },
      stateVersion: TALK_SPEECH_PROJECTION_STATE_VERSION,
    }), 'dsh-talk: talk:speech projection')
  })

  // Consumer — the announcement listeners drive the talk service from host events.
  // Turn-completion announcement: idle after an observed running transition.
  const runningAgents = new Set<Agent>()
  ctx.on('agent/status', ({ agent, status }) => {
    if (status === 'running') {
      runningAgents.add(agent)
      return
    }
    if (!runningAgents.delete(agent)) return
    // A turn the user spoke, which the model did not answer through the speak tool, is spoken here.
    if (service.voiceTurns.finishTurn(String(agent.session.id))) {
      const reply = voiceReplyText(lastReplyText(agent.session.ownEvents()))
      service.speak(reply, { reason: 'voice-reply', session: agent.session }).catch((error: unknown) => {
        console.warn('[dsh-talk] voice reply failed:', error)
      })
      return
    }
    if (!resolved.announceEnabled || !resolved.announceOnTurnEnd) return
    void service.speak(resolved.messageTurnEnd, { reason: 'turn-end', session: agent.session })
  })

  // Voice-turn instruction: the first step of a turn the user spoke tells the model to answer through
  // the speak tool. Other steps and typed turns pass through unchanged.
  ctx.on('agent/pre-step', async ({ agent, step, signal }, next) => {
    const decision = await next()
    if (decision.kind === 'reject' || signal.aborted || step !== 1) return decision
    if (!service.voiceTurns.startTurn(String(agent.session.id))) return decision
    return {
      ...decision,
      messages: [...decision.messages, createUserMessage({
        content: [{ type: 'text', text: VOICE_TURN_INSTRUCTION }],
        source: { kind: 'dsh-talk-voice-turn', form: 'snapshot', sections: [{ name: 'dsh-talk-voice-turn', text: VOICE_TURN_INSTRUCTION }] },
      })],
    }
  }, { prepend: true })

  // Error announcement: emit-mode notification, fire-and-forget speech.
  ctx.on('agent/error', ({ agent }) => {
    if (!resolved.announceEnabled || !resolved.announceOnError) return
    void service.speak(resolved.messageError, { reason: 'error', session: agent.session })
  })

  // Approval announcement: waterfall listener — always delegates with next()
  // and never blocks the gate on speech. The 0.1.2-alpha.3 waterfall carries
  // the ApprovalRequestEvent payload; the listener types it by inference.
  ctx.on('approval/request', (request, next) => {
    const outcome = next()
    if (resolved.announceEnabled && resolved.announceOnApproval) {
      const session = (request.agent as { session?: unknown }).session
      if (session !== undefined) {
        void service.speak(resolved.messageApproval, { reason: 'approval', session: session as never }).catch(() => {})
      }
    }
    return outcome
  })
}
