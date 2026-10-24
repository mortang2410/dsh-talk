/**
 * Voice-turn bookkeeping: the arm a microphone prompt leaves behind, the
 * first-step decision, the speak-back text the host uses when the model does
 * not call the speak tool, and the wire descriptor. Pure logic; the clock is
 * injected, so no host or DOM is needed.
 *
 * @module dsh-talk/test/voice-turn.spec
 */

import { describe, expect, it } from 'vitest'
import type { SessionEvent } from '@deepseek-ai/dsh-session/types'
import { TALK_INVOCATIONS, TALK_VOICE_TURN_DESCRIPTOR } from '../src/wire.ts'
import {
  lastReplyText,
  VOICE_ARM_TTL_MS,
  VOICE_REPLY_FALLBACK,
  VOICE_REPLY_MAX_CHARS,
  VoiceTurns,
  voiceReplyText,
} from '../src/voice-turn.ts'

const SESSION = 'session-voice'

/** A clock the test moves by hand. */
function fakeClock(start = 1_000) {
  let now = start
  return {
    now: () => now,
    advance: (ms: number) => {
      now += ms
    },
  }
}

describe('VoiceTurns', () => {
  it('starts a typed turn when nothing was armed', () => {
    const turns = new VoiceTurns(fakeClock().now)
    expect(turns.startTurn(SESSION)).toBe(false)
  })

  it('starts a voice turn after an arm, and the arm is consumed', () => {
    const turns = new VoiceTurns(fakeClock().now)
    turns.arm(SESSION)
    expect(turns.startTurn(SESSION)).toBe(true)
    expect(turns.startTurn(SESSION)).toBe(false)
  })

  it('treats an arm older than the TTL as a typed turn', () => {
    const time = fakeClock()
    const turns = new VoiceTurns(time.now)
    turns.arm(SESSION)
    time.advance(VOICE_ARM_TTL_MS + 1)
    expect(turns.startTurn(SESSION)).toBe(false)
  })

  it('still accepts an arm at exactly the TTL', () => {
    const time = fakeClock()
    const turns = new VoiceTurns(time.now)
    turns.arm(SESSION)
    time.advance(VOICE_ARM_TTL_MS)
    expect(turns.startTurn(SESSION)).toBe(true)
  })

  it('reports no fallback when the speak tool ran during the voice turn', () => {
    const turns = new VoiceTurns(fakeClock().now)
    turns.arm(SESSION)
    turns.startTurn(SESSION)
    turns.noteToolSpeech(SESSION)
    expect(turns.finishTurn(SESSION)).toBe(false)
  })

  it('reports a fallback once when a voice turn ends without the speak tool', () => {
    const turns = new VoiceTurns(fakeClock().now)
    turns.arm(SESSION)
    turns.startTurn(SESSION)
    expect(turns.finishTurn(SESSION)).toBe(true)
    expect(turns.finishTurn(SESSION)).toBe(false)
  })

  it('never reports a fallback for a typed turn', () => {
    const turns = new VoiceTurns(fakeClock().now)
    turns.startTurn(SESSION)
    expect(turns.finishTurn(SESSION)).toBe(false)
  })

  it('clears an unfinished voice turn when a typed turn starts after it', () => {
    const turns = new VoiceTurns(fakeClock().now)
    turns.arm(SESSION)
    turns.startTurn(SESSION)
    turns.startTurn(SESSION)
    expect(turns.finishTurn(SESSION)).toBe(false)
  })

  it('keeps sessions apart', () => {
    const turns = new VoiceTurns(fakeClock().now)
    turns.arm('session-a')
    expect(turns.startTurn('session-b')).toBe(false)
    expect(turns.startTurn('session-a')).toBe(true)
  })
})

describe('voiceReplyText', () => {
  it('speaks the first four sentences only, separated by single spaces', () => {
    const reply = 'One is here. Two is here. Three is here. Four is here. Five is here.'
    expect(voiceReplyText(reply)).toBe('One is here. Two is here. Three is here. Four is here.')
  })

  it('keeps a trailing sentence that has no final punctuation', () => {
    expect(voiceReplyText('Fix is in. The tests are still running')).toBe(
      'Fix is in. The tests are still running',
    )
  })

  it('collapses line breaks and runs of spaces', () => {
    expect(voiceReplyText('First line.\n\nSecond   line.')).toBe('First line. Second line.')
  })

  it('speaks the fallback line when the reply is code only', () => {
    expect(voiceReplyText('```ts\nconst x = 1\n```')).toBe(VOICE_REPLY_FALLBACK)
  })

  it('speaks the fallback line for an empty or blank reply', () => {
    expect(voiceReplyText('')).toBe(VOICE_REPLY_FALLBACK)
    expect(voiceReplyText('   \n  ')).toBe(VOICE_REPLY_FALLBACK)
  })

  it('caps the utterance length', () => {
    const spoken = voiceReplyText(`${'word '.repeat(200)}end.`)
    expect(spoken.length).toBeLessThanOrEqual(VOICE_REPLY_MAX_CHARS)
    expect(spoken.startsWith('word word')).toBe(true)
  })
})

describe('lastReplyText', () => {
  const turnStart = { type: 'turn/start', data: {} } as unknown as SessionEvent
  const assistant = (...parts: unknown[]) =>
    ({ type: 'assistant/message', data: { message: { role: 'assistant', content: parts } } }) as unknown as SessionEvent
  const text = (value: string) => ({ type: 'text', text: value })
  const toolCall = { type: 'tool-call', toolCallId: 'call-1', toolName: 'speak', args: {} }

  it('returns the text of the last assistant message of the newest turn', () => {
    const events = [turnStart, assistant(text('Old turn.')), turnStart, assistant(text('New turn.'))]
    expect(lastReplyText(events)).toBe('New turn.')
  })

  it('ignores tool-call parts', () => {
    expect(lastReplyText([turnStart, assistant(toolCall, text('Here is the answer.'))])).toBe(
      'Here is the answer.',
    )
  })

  it('falls back to an earlier message of the same turn when the last has no text', () => {
    const events = [turnStart, assistant(text('Checking now.')), assistant(toolCall)]
    expect(lastReplyText(events)).toBe('Checking now.')
  })

  it('does not read the turn before the newest one', () => {
    const events = [turnStart, assistant(text('Old turn.')), turnStart, assistant(toolCall)]
    expect(lastReplyText(events)).toBe('')
  })

  it('returns an empty string when there is no turn', () => {
    expect(lastReplyText([])).toBe('')
  })
})

describe('wire', () => {
  it('registers the voiceTurn invocation', () => {
    expect(TALK_INVOCATIONS).toContain(TALK_VOICE_TURN_DESCRIPTOR)
    expect(TALK_VOICE_TURN_DESCRIPTOR.method).toBe('voiceTurn')
  })
})
