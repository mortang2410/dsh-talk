// @vitest-environment jsdom
/**
 * The record button's playback wiring, rendered under jsdom with the remote
 * surface stubbed: the button polls `latest` for the session id it was given,
 * requests `audio` once per new utterance id, does not replay after a remount
 * (a page reload), and stops polling on unmount. The poll rules themselves are
 * covered by `speech-poll.spec.ts`.
 *
 * @module dsh-talk/test/talk-mic-playback.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { TalkMicButton } from '../src/client/TalkMicButton.tsx'
import type { TalkLatest, TalkStatus } from '../src/wire.ts'

const SESSION = 'session-mic-under-test'
const SETTINGS: TalkStatus = {
  stt: { engine: 'web', resolved: 'web', language: 'en-US', silenceFinaliseMs: 1500, funasrUrl: null, whisperModel: null },
  tts: {
    engine: 'edge-tts',
    resolved: 'edge-tts',
    voice: null,
    fallbackToBrowser: false,
    piperModel: null,
    edgeTtsVoice: 'en-US-AriaNeural',
    browser: { voiceName: null, rate: 1, pitch: 1 },
  },
  announce: { enabled: false, onTurnEnd: false, onApproval: false, onError: false },
  record: { enabled: true, hotkey: null, maxSeconds: 60, autoSubmit: false, vad: { enabled: false, silenceMs: 1000, energyThreshold: 0.01 } },
  interrupt: false,
  patchFile: null,
}

/** What the stubbed remote surface has seen and returned. */
interface Remote {
  latestValue: TalkLatest | null
  latestCalls: string[]
  audioCalls: string[]
}

/** Audio elements the button constructed (one per played utterance). */
const constructedAudio: string[] = []

class FakeAudio {
  private readonly handlers = new Map<string, Array<() => void>>()
  constructor(src: string) {
    constructedAudio.push(src)
  }
  addEventListener(type: string, handler: () => void): void {
    this.handlers.set(type, [...(this.handlers.get(type) ?? []), handler])
  }
  play(): Promise<void> {
    // Playback "ends" on the next microtask, so the queue advances.
    return Promise.resolve().then(() => {
      for (const handler of this.handlers.get('ended') ?? []) handler()
    })
  }
}

function utterance(utteranceId: string): TalkLatest {
  return {
    sessionId: SESSION,
    utteranceId,
    engine: 'edge-tts',
    text: `spoken ${utteranceId}`,
    audioBytes: 3,
    reason: 'speak-tool',
    logged: true,
    appended: 1,
    skipped: 0,
  }
}

let container: HTMLElement
let root: Root | undefined
let remote: Remote
const originalCreate = (URL as unknown as { createObjectURL?: unknown }).createObjectURL
const originalRevoke = (URL as unknown as { revokeObjectURL?: unknown }).revokeObjectURL

function mount(): void {
  const props = {
    status: async () => SETTINGS,
    interrupt: async () => ({ stopped: false }),
    transcribe: async () => ({ text: '', engine: 'whisper', language: 'en' }),
    audio: async (utteranceId: string) => {
      remote.audioCalls.push(utteranceId)
      return { utteranceId, mime: 'audio/mpeg', data: 'AAAA', text: '', engine: 'edge-tts' }
    },
    latest: async (sessionId: string) => {
      remote.latestCalls.push(sessionId)
      return remote.latestValue
    },
    setDraft: () => {},
    send: async () => {},
    sessionId: SESSION,
  }
  root = createRoot(container)
  act(() => {
    root?.render(createElement(TalkMicButton as never, props as never))
  })
}

function unmount(): void {
  if (root !== undefined) act(() => root?.unmount())
  root = undefined
}

async function advance(ms: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

describe('TalkMicButton playback wiring (stubbed talk remote)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => false })
    vi.stubGlobal('Audio', FakeAudio)
    Object.assign(URL, { createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} })
    constructedAudio.length = 0
    remote = { latestValue: null, latestCalls: [], audioCalls: [] }
    container = document.createElement('div')
    document.body.append(container)
  })

  afterEach(() => {
    unmount()
    container.remove()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    Reflect.deleteProperty(document, 'hidden')
    Object.assign(URL, { createObjectURL: originalCreate, revokeObjectURL: originalRevoke })
  })

  it('polls for its own session and requests audio once per new utterance id', async () => {
    mount()
    await advance(0)
    remote.latestValue = utterance('u1')
    await advance(700)
    expect(remote.audioCalls).toEqual(['u1'])
    expect(constructedAudio).toHaveLength(1)
    await advance(700 * 3)
    expect(remote.audioCalls).toEqual(['u1'])
    expect(constructedAudio).toHaveLength(1)
    expect(new Set(remote.latestCalls)).toEqual(new Set([SESSION]))
  })

  it('a remount (page reload) does not replay the utterance cached before it', async () => {
    remote.latestValue = utterance('before-reload')
    mount()
    await advance(0)
    unmount()
    mount()
    await advance(700 * 2)
    expect(remote.audioCalls).toEqual([])
    expect(constructedAudio).toHaveLength(0)
  })

  it('unmounting stops the poll', async () => {
    mount()
    await advance(0)
    const callsBeforeUnmount = remote.latestCalls.length
    expect(callsBeforeUnmount).toBeGreaterThan(0)
    unmount()
    await advance(700 * 5)
    expect(remote.latestCalls.length).toBe(callsBeforeUnmount)
  })
})
