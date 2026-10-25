/**
 * The client-side Remote face of the `talk` namespace: the hand-written
 * `TypertRemoteContribution` mounted through `ctx.remote.$mount`, plus the
 * declaration merging that types `ctx.remote.talk`. The descriptor list is
 * shared with the host `./typert` manifest (`../wire.ts`), so the two faces
 * can never drift.
 *
 * @module dsh-talk/client/remote
 */

import type { RemoteResult, TypertRemoteContribution } from '@deepseek-ai/dsh-typert-protocol'
import { TALK_INVOCATIONS } from '../wire.ts'
import type {
  TalkAudio,
  TalkHeal,
  TalkInterruptResult,
  TalkLatest,
  TalkSettingsInput,
  TalkSettingsResult,
  TalkStatus,
  TalkTranscript,
  TalkTranscriptLogEntry,
  TalkTranscriptLogResult,
  TalkVoiceTurn,
} from '../wire.ts'

declare module '@deepseek-ai/dsh-typert-protocol' {
  interface TypertRemoteNamespace$talk {
    /** Read-only settings snapshot. */
    status: () => Promise<RemoteResult<TalkStatus>>
    /** Serve one cached utterance's audio; null when evicted. */
    audio: (utteranceId: string) => Promise<RemoteResult<TalkAudio | null>>
    /** Transcribe one base64 WAV with the configured local STT engine. */
    transcribe: (audioData: string) => Promise<RemoteResult<TalkTranscript>>
    /** Clean one segment's newest text on the host, with the earlier raw words as context. */
    heal: (context: string, text: string) => Promise<RemoteResult<TalkHeal>>
    /** Append one stop's raw and cleaned text to the host's transcript trail. */
    logTranscript: (entry: TalkTranscriptLogEntry) => Promise<RemoteResult<TalkTranscriptLogResult>>
    /** Apply the settings panel's edits as an append-only profile patch operation. */
    applySettings: (settings: TalkSettingsInput) => Promise<RemoteResult<TalkSettingsResult>>
    /** Stop in-flight host synthesis (the user started talking). */
    interrupt: () => Promise<RemoteResult<TalkInterruptResult>>
    /** The newest utterance of ONE session (null when it never spoke); session-keyed, so multi-session surfaces never read another session's utterance. */
    latest: (sessionId: string, signal?: AbortSignal) => Promise<RemoteResult<TalkLatest | null>>
    /** Arm the session's next prompt as a microphone turn, so the host tells the model it is spoken. */
    voiceTurn: (sessionId: string) => Promise<RemoteResult<TalkVoiceTurn>>
  }
  interface TypertRemoteMap {
    'talk/voiceTurn': (sessionId: string) => Promise<RemoteResult<TalkVoiceTurn>>
    'talk/status': () => Promise<RemoteResult<TalkStatus>>
    'talk/audio': (utteranceId: string) => Promise<RemoteResult<TalkAudio | null>>
    'talk/transcribe': (audioData: string) => Promise<RemoteResult<TalkTranscript>>
    'talk/heal': (context: string, text: string) => Promise<RemoteResult<TalkHeal>>
    'talk/logTranscript': (entry: TalkTranscriptLogEntry) => Promise<RemoteResult<TalkTranscriptLogResult>>
    'talk/applySettings': (settings: TalkSettingsInput) => Promise<RemoteResult<TalkSettingsResult>>
    'talk/interrupt': () => Promise<RemoteResult<TalkInterruptResult>>
    'talk/latest': (sessionId: string, signal?: AbortSignal) => Promise<RemoteResult<TalkLatest | null>>
  }
  interface TypertRemoteNamespaceMap {
    talk: TypertRemoteNamespace$talk
  }
}

/** The client Remote contribution for the `talk` namespace. */
export const TALK_REMOTE = Object.freeze({
  package: 'dsh-talk',
  descriptors: TALK_INVOCATIONS,
} satisfies TypertRemoteContribution)
