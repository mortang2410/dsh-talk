/**
 * The `speak` tool: the agent's audible reply channel. Execution resolves
 * the TTS engine through the service (config defaults + per-call overrides)
 * and returns a small canonical value the model can reason about — the audio
 * itself plays in the browser and never enters model context. The model
 * only sees the render text; the spoken utterance lands in the session log
 * as a `dsh-talk/speech` event.
 *
 * @module dsh-talk/tool
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { sanitizeText } from './sanitize.ts'
import type { TalkService } from './service.ts'

/**
 * Render one speak outcome as the model-facing content. The value shape is
 * the schema-inferred canonical value (every property optional).
 *
 * @param args - the validated tool arguments.
 * @param value - the canonical outcome.
 * @returns the model content blocks.
 */
export function renderSpeak(args: { text: string }, value: {
  spoken?: boolean
  engine?: string
  audioBytes?: number
  durationSec?: number
  error?: string
  utteranceId?: string
}) {
  if (value.spoken === true) {
    const note = value.error === undefined ? '' : ` (${value.error})`
    return [{
      type: 'text' as const,
      text: `Spoke ${sanitizeText(args.text, 200)} aloud with the ${value.engine ?? 'unknown'} engine.${note}`,
    }]
  }
  return [{
    type: 'text' as const,
    text: `Speech failed with the ${value.engine ?? 'unknown'} engine: ${value.error ?? 'unknown error'}.`,
  }]
}

/** A fenced code block opener or closer: a run of three backticks or tildes. */
const FENCE_LINE = /^\s*(```|~~~)/u

/** A markdown table row or separator: two or more pipes on one line. */
function isTableLine(line: string): boolean {
  return (line.match(/\|/gu) ?? []).length >= 2
}

/**
 * Reduce a written answer to the text that may be spoken. This is the second
 * guard behind the tool description: it does not rely on the model complying.
 * Removes fenced code blocks (through the closing fence, or to the end when
 * unclosed), indented code lines (four spaces or a tab), table rows, URLs,
 * absolute, home-relative and dot-relative paths, and multi-segment relative
 * file paths. Backticks are dropped so inline code words stay readable.
 *
 * @param raw - the text the model passed to the tool.
 * @returns the speakable text; empty when nothing else remains.
 */
export function speakableText(raw: string): string {
  const kept: string[] = []
  let fence: string | null = null
  for (const line of raw.split(/\r?\n/u)) {
    const fenceMatch = FENCE_LINE.exec(line)
    if (fence !== null) {
      if (fenceMatch !== null && fenceMatch[1] === fence) fence = null
      continue
    }
    if (fenceMatch !== null) {
      fence = fenceMatch[1] ?? null
      continue
    }
    if (/^( {4}|\t)/u.test(line) || isTableLine(line)) continue
    kept.push(line)
  }
  return kept.join(' ')
    .replace(/https?:\/\/\S+/giu, ' ')
    .replace(/(?<=^|\s)(?:~|\.{1,2})?\/\S+/gu, ' ')
    .replace(/(?<![\w/.-])[\w.-]+(?:\/[\w.-]+)+\.[A-Za-z0-9]{1,8}(?![\w/])/gu, ' ')
    .replace(/(?<![\w/.-])[\w.-]+(?:\/[\w.-]+){2,}(?![\w/])/gu, ' ')
    .replace(/`/gu, '')
    .replace(/\s+/gu, ' ')
    .trim()
}

/**
 * Assemble the `speak` tool definition. `execute` returns only the canonical
 * {@link SpeakOutcome} declared by the output schema; infrastructure failures
 * (empty text) throw so the pipeline marks the call errored.
 *
 * @param service - the talk service the execution delegates to.
 * @param maxChars - cap on one spoken text (resolved config).
 * @returns a registry-ready tool.
 */
export function speakTool(service: TalkService, maxChars: number) {
  return defineTool({
    name: 'speak',
    description: 'Speak a short paraphrase of your answer aloud to the user. Write the paraphrase yourself, in at most four sentences, colloquial, with no code, no tables, no file paths and no URLs. Do not paste your written answer.',
    parameters: {
      text: {
        type: 'string',
        required: true as const,
        description: 'Your spoken paraphrase: at most four sentences, colloquial, no code, tables, file paths or URLs.',
      },
      engine: {
        type: 'string',
        enum: ['auto', 'browser', 'edge-tts', 'piper'],
        description: 'TTS engine override; default is the configured engine.',
      },
      voice: {
        type: 'string',
        description: 'Voice name override for the resolved engine.',
      },
      interrupt: {
        type: 'boolean',
        description: 'Interrupt any speech currently playing before starting (default true).',
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          spoken: { type: 'boolean' },
          engine: { type: 'string' },
          audioBytes: { type: 'integer' },
          durationSec: { type: 'number' },
          error: { type: 'string' },
          utteranceId: { type: 'string' },
        },
        additionalProperties: false,
      },
      render: renderSpeak,
    },
    async execute(args, exec) {
      const text = args.text.trim()
      if (text === '') throw new Error('dsh-talk: speak text must not be empty')
      if (text.length > maxChars) {
        throw new Error(`dsh-talk: speak text is ${text.length} characters, over the ${maxChars} character cap; shorten the paraphrase and retry`)
      }
      const spoken = speakableText(text)
      if (spoken === '') {
        throw new Error('dsh-talk: nothing speakable is left after removing code, tables and paths; write a plain paraphrase and retry')
      }
      const session = exec.agent?.session
      return await service.speak(spoken, {
        ...(args.engine === undefined ? {} : { engine: args.engine }),
        ...(args.voice === undefined ? {} : { voice: args.voice }),
        ...(args.interrupt === undefined ? {} : { interrupt: args.interrupt }),
        reason: 'speak-tool',
        ...(session === undefined ? {} : { session }),
        signal: exec.signal,
      })
    },
  })
}
