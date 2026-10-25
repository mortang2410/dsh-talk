/**
 * The mic-button state machine: press → recording → transcribing → done, and
 * failures.
 *
 * @module dsh-talk/test/present.spec
 */

import { describe, expect, it } from 'vitest'
import { foldMic, initMic } from '../src/client/present.ts'

describe('foldMic', () => {
  it('walks the happy path', () => {
    let state = initMic()
    state = foldMic(state, { kind: 'press' })
    expect(state.phase).toBe('recording')
    state = foldMic(state, { kind: 'tick', elapsedSec: 3 })
    expect(state.elapsedSec).toBe(3)
    state = foldMic(state, { kind: 'stopped' })
    expect(state.phase).toBe('transcribing')
    state = foldMic(state, { kind: 'transcribed', text: '  hello world  ' })
    expect(state.phase).toBe('idle')
    expect(state.lastText).toBe('hello world')
  })

  it('records a failure and recovers on reset', () => {
    let state = initMic()
    state = foldMic(state, { kind: 'failed', message: 'mic denied' })
    expect(state.phase).toBe('error')
    expect(state.error).toBe('mic denied')
    state = foldMic(state, { kind: 'reset' })
    expect(state.phase).toBe('idle')
    expect(state.error).toBeUndefined()
  })

  it('shows a notice while the phase stays recording, and clears it on a clear event or a new press', () => {
    let state = initMic()
    state = foldMic(state, { kind: 'press' })
    state = foldMic(state, { kind: 'notice', message: 'no words recognised in that segment' })
    expect(state.phase).toBe('recording')
    expect(state.notice).toBe('no words recognised in that segment')
    state = foldMic(state, { kind: 'tick', elapsedSec: 2 })
    expect(state.notice).toBe('no words recognised in that segment')
    state = foldMic(state, { kind: 'notice', message: undefined })
    expect(state.notice).toBeUndefined()
    state = foldMic(state, { kind: 'notice', message: 'no words recognised in that segment' })
    state = foldMic(state, { kind: 'press' })
    expect(state.notice).toBeUndefined()
  })

})
