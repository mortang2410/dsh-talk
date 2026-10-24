/**
 * The speak tool through the REAL tool registry: registration, schema
 * validation, the browser-engine fast path, the local-engine failure →
 * browser fallback, and the interrupt outcome. The subprocess provider is
 * scripted; every other piece of the pipeline runs for real.
 *
 * @module dsh-talk/test/tool.spec
 */

import { describe, expect, it } from 'vitest'
import type { ToolExecutionResult } from '@deepseek-ai/dsh-tools'
import { mountHarness, withKnownSpeechVocabulary, type Harness } from './harness.ts'
import { speakableText } from '../src/tool.ts'

let callCounter = 0

/** Execute-parameter brand of the call id (CallId on rc.2, ToolCallId on 0.1.2-alpha). */
type ExecuteArgs = Parameters<Harness['ctx']['tools']['execute']>[0]

/** Build a call id typed to whichever brand the resolved tool registry expects. */
function callIdOf(id: string): ExecuteArgs['callId'] {
  return id as ExecuteArgs['callId']
}

async function callTool(harness: Harness, args: unknown): Promise<ToolExecutionResult> {
  callCounter += 1
  return harness.ctx.tools.execute({
    callId: callIdOf(`talk-spec-${callCounter}`),
    name: 'speak',
    arguments: args,
    agent: harness.agent,
    signal: new AbortController().signal,
  })
}

describe('speak tool', () => {
  it('registers in the real registry', async () => {
    const harness = await mountHarness()
    expect(harness.ctx.tools.get('speak')).toBeDefined()
  })

  it('rejects empty text', async () => {
    const harness = await mountHarness()
    const result = await callTool(harness, { text: '   ' })
    expect(result.isError).toBe(true)
  })

  it('speaks through the browser engine without subprocess I/O and records the log degradation', async () => {
    const harness = await mountHarness()
    const result = await callTool(harness, { text: 'hello', engine: 'browser' })
    expect(result.isError).toBe(false)
    if (result.isError) return
    const value = result.value as { spoken: boolean; engine: string; audioBytes: number }
    expect(value.spoken).toBe(true)
    expect(value.engine).toBe('browser')
    expect(value.audioBytes).toBe(0)
    // This host's vocabulary excludes the out-of-repo speech type: the gate
    // skips the append, and the skip is recorded per session instead of being
    // dropped silently. The `speak` tool result stays the model-visible record.
    const speech = harness.session.snapshotEvents().filter(event => event.type === 'dsh-talk/speech')
    expect(speech).toHaveLength(0)
    expect(harness.service.speechLogOutcome(String(harness.session.id))).toMatchObject({ appended: 0, skipped: 1 })
  })

  it('appends the speech event and reports it when the host vocabulary covers it (explicit opt-in)', async () => {
    await withKnownSpeechVocabulary(async () => {
      const harness = await mountHarness()
      await callTool(harness, { text: 'hello', engine: 'browser' })
      const speech = harness.session.snapshotEvents().filter(event => event.type === 'dsh-talk/speech')
      expect(speech).toHaveLength(1)
      expect(harness.service.speechLogOutcome(String(harness.session.id))).toMatchObject({ appended: 1, skipped: 0 })
    })
  })

  it('carries browser delivery settings and the per-call voice override in the speech event', async () => {
    await withKnownSpeechVocabulary(async () => {
      const harness = await mountHarness({ tts: { browser: { voiceName: 'Configured Voice', rate: 1.5, pitch: 0.8 } } })
      await callTool(harness, { text: 'hello', engine: 'browser' })
      let data = (harness.session.snapshotEvents().filter(event => event.type === 'dsh-talk/speech').at(-1)?.data ?? {}) as {
        voice?: string
        rate?: number
        pitch?: number
      }
      expect(data.voice).toBe('Configured Voice')
      expect(data.rate).toBe(1.5)
      expect(data.pitch).toBe(0.8)
      await callTool(harness, { text: 'hello again', engine: 'browser', voice: 'Override Voice' })
      data = (harness.session.snapshotEvents().filter(event => event.type === 'dsh-talk/speech').at(-1)?.data ?? {}) as typeof data
      expect(data.voice).toBe('Override Voice')
      expect(data.rate).toBe(1.5)
      expect(data.pitch).toBe(0.8)
    })
  })

  it('omits browser delivery fields on local-engine utterances', async () => {
    await withKnownSpeechVocabulary(async () => {
      const harness = await mountHarness()
      harness.subprocess.nextSynthBytes = 'ID3-fake-audio'
      const result = await callTool(harness, { text: 'hello', engine: 'edge-tts', interrupt: false })
      expect(result.isError).toBe(false)
      if (result.isError) throw new Error('expected successful local synthesis')
      expect(result.value).toMatchObject({ spoken: true, engine: 'edge-tts' })
      const data = (harness.session.snapshotEvents().filter(event => event.type === 'dsh-talk/speech').at(-1)?.data ?? {}) as Record<string, unknown>
      expect(data.engine).toBe('edge-tts')
      expect('voice' in data).toBe(false)
      expect('rate' in data).toBe(false)
      expect('pitch' in data).toBe(false)
    })
  })

  it('falls back to the browser voice when edge-tts fails', async () => {
    const harness = await mountHarness()
    harness.subprocess.nextExitCode = 1
    harness.subprocess.nextStderr = 'edge-tts exploded'
    const result = await callTool(harness, { text: 'hello', engine: 'edge-tts', interrupt: false })
    expect(result.isError).toBe(false)
    if (result.isError) return
    const value = result.value as { engine: string; error?: string }
    expect(value.engine).toBe('browser')
    expect(value.error).toContain('edge-tts')
    expect(harness.subprocess.spawns.length).toBeGreaterThan(0)
  })

  it('refuses a hostile engine value at the schema boundary', async () => {
    const harness = await mountHarness()
    const result = await callTool(harness, { text: 'hi', engine: 'rm-rf' })
    expect(result.isError).toBe(true)
  })

  it('lists the paraphrase rule in the composed description the model reads', async () => {
    const harness = await mountHarness()
    const listed = harness.ctx.tools.get('speak') as unknown as { description?: string } | undefined
    const description = listed?.description ?? ''
    expect(description).toContain('at most four sentences')
    expect(description).toContain('colloquial')
    expect(description).toContain('no code')
    expect(description).toContain('no tables')
    expect(description).toContain('no file paths')
    expect(description).toContain('no URLs')
  })

  it('refuses an oversized text, names the size and the cap, and synthesizes nothing', async () => {
    const harness = await mountHarness({ maxSpeakChars: 20 })
    const result = await callTool(harness, { text: 'x'.repeat(30), engine: 'browser' })
    expect(result.isError).toBe(true)
    expect(JSON.stringify(result)).toContain('30 characters')
    expect(JSON.stringify(result)).toContain('20 character cap')
    expect(harness.service.speechLogOutcome(String(harness.session.id))?.skipped ?? 0).toBe(0)
  })

  it('refuses a text that is only a fenced code block and speaks nothing', async () => {
    const harness = await mountHarness()
    const result = await callTool(harness, { text: '```ts\nconst x = 1\n```', engine: 'browser' })
    expect(result.isError).toBe(true)
    expect(JSON.stringify(result)).toContain('nothing speakable')
    expect(harness.service.speechLogOutcome(String(harness.session.id))?.skipped ?? 0).toBe(0)
  })

  it('speaks the prose of a text that mixes a code block with a sentence', async () => {
    await withKnownSpeechVocabulary(async () => {
      const harness = await mountHarness()
      const result = await callTool(harness, { text: 'Here is the fix.\n```ts\nconst x = 1\n```\nRun it after the reload.', engine: 'browser' })
      expect(result.isError).toBe(false)
      const data = (harness.session.snapshotEvents().filter(event => event.type === 'dsh-talk/speech').at(-1)?.data ?? {}) as { text?: string }
      expect(data.text).toBe('Here is the fix. Run it after the reload.')
    })
  })
})

describe('speakableText', () => {
  it('removes fenced code blocks, tilde fences and an unclosed fence to the end', () => {
    expect(speakableText('Before.\n```\ncode\n```\nAfter.')).toBe('Before. After.')
    expect(speakableText('Before.\n~~~py\nprint(1)\n~~~\nAfter.')).toBe('Before. After.')
    expect(speakableText('Before.\n```\nnever closed')).toBe('Before.')
  })

  it('removes indented code lines and table rows', () => {
    expect(speakableText('Look here.\n    const y = 2\nDone.')).toBe('Look here. Done.')
    expect(speakableText('Values.\n| a | b |\n|---|---|\n| 1 | 2 |\nThat is all.')).toBe('Values. That is all.')
  })

  it('removes absolute, home-relative, dot-relative and multi-segment file paths and URLs', () => {
    expect(speakableText('Open /home/wilder/.dsh/x.json now.')).toBe('Open now.')
    expect(speakableText('Edit ~/notes/todo and ./run here.')).toBe('Edit and here.')
    expect(speakableText('See src/tool.ts for it.')).toBe('See for it.')
    expect(speakableText('Read https://example.com/a/b today.')).toBe('Read today.')
  })

  it('keeps ordinary prose and slashes between words, and drops an inline code span', () => {
    // An inline span is code, so the whole span goes rather than the backticks alone. The
    // surrounding prose and a slash inside a word stay.
    expect(speakableText('Use `pow2` and/or the 1/2 split.')).toBe('Use and/or the 1/2 split.')
  })

  it('returns an empty string when nothing speakable remains', () => {
    expect(speakableText('```\nonly code\n```')).toBe('')
    expect(speakableText('| a | b |\n|---|---|')).toBe('')
  })
})

describe('speakableText: the leaks the review probes found', () => {
  it('drops a table whose rows have a single pipe with spaces around it', () => {
    // Breaks if: a two-column table without edge pipes is spoken as prose.
    expect(speakableText('name | value\nfoo | 1\nbar | 2')).toBe('')
  })

  it('drops a URL with no scheme', () => {
    // Breaks if: "www.example.com" reaches the speaker.
    expect(speakableText('see www.example.com for details')).toBe('see for details')
    expect(speakableText('see example.com/docs for details')).toBe('see for details')
  })

  it('returns nothing when stripping leaves only punctuation', () => {
    // Breaks if: the caller is handed "..." and synthesizes it.
    expect(speakableText('```\ncode\n```\n...')).toBe('')
  })
})

describe('speakableText: the cases the second review probe found', () => {
  it('drops a fence opened on the same line as a list marker, and keeps the prose after it', () => {
    expect(speakableText('- ```js\nconst a = 1\n```\nand then prose')).toBe('and then prose')
    expect(speakableText('1. ```py\nprint(1)\n```\nafter')).toBe('after')
  })

  it('keeps a markdown link label and drops its target', () => {
    expect(speakableText('see [the docs](https://example.com/x) for more')).toBe('see the docs for more')
  })

  it('drops an inline code span whole rather than leaving the bare word', () => {
    expect(speakableText('call `speakableText` on it')).toBe('call on it')
  })

  it('drops a parenthesised absolute path with its parenthesis', () => {
    expect(speakableText('open it (in /etc/hosts) and look')).toBe('open it (in and look')
  })

  it('drops a bare filename and a bare TLD', () => {
    expect(speakableText('edit config.ts and rerun')).toBe('edit and rerun')
    expect(speakableText('go to example.xyz now')).toBe('go to now')
  })
})

describe('speakableText: tables with no pipes and a lead-in left dangling', () => {
  it('drops a whitespace-column table block, header row included', () => {
    expect(speakableText('Name  Count\n----- -----\nAlice  1\nBob  2')).toBe('')
  })

  it('stops the block at the first line that is not columnar', () => {
    expect(speakableText('Totals\nName  Count\n----- -----\nAlice  1\nand then prose')).toBe('Totals and then prose')
  })

  it('drops a horizontal rule', () => {
    expect(speakableText('Intro\n---\nBody text')).toBe('Intro Body text')
  })

  it('refuses a lead-in whose payload was removed', () => {
    expect(speakableText('The result is:\n```json\n{"ok":true}\n```')).toBe('')
    expect(speakableText('Here is the plan:')).toBe('')
  })

  it('keeps a colon inside a sentence', () => {
    expect(speakableText('There are two things: first this, then that.')).toBe('There are two things: first this, then that.')
  })
})

describe('speakableText: the third review probe found these', () => {
  it('drops a four-space-indented first line, which the tool path must not trim away first', () => {
    // The tool trims for the empty and length checks, but the filter sees the raw text: trimming
    // first would strip the indentation that marks the line as code.
    expect(speakableText('    const secret = 1\nand then prose')).toBe('and then prose')
  })

  it('drops a domain with any TLD, not a fixed list of common ones', () => {
    expect(speakableText('Visit example.edu/docs for reference.')).toBe('Visit for reference.')
    expect(speakableText('go to example.technology/docs now')).toBe('go to now')
    expect(speakableText('see example.com for more')).toBe('see for more')
  })

  it('still removes a path whole before its tail can look like a domain', () => {
    expect(speakableText('See src/tool.ts for it.')).toBe('See for it.')
    expect(speakableText('edit config.ts and rerun')).toBe('edit and rerun')
  })

  it('keeps an abbreviation and a decimal', () => {
    expect(speakableText('use it, e.g. this one')).toBe('use it, e.g. this one')
    expect(speakableText('it is 3.5 seconds')).toBe('it is 3.5 seconds')
  })

  it('drops a shell prompt line, a footnote definition and a one-line JSON object', () => {
    expect(speakableText('$ npm run test\nthen look at the output')).toBe('then look at the output')
    expect(speakableText('[^1]: a note here\nthe answer follows')).toBe('the answer follows')
    expect(speakableText('{"ok":true}\nthe result is ready')).toBe('the result is ready')
  })
})
