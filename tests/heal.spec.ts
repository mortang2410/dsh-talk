/**
 * Heal contract: the endpoint and model pins, the timeout bound, and the byte identity of
 * the windowed prompt constant with HEAL-PROMPT.md when DSH_TALK_HEAL_PROMPT names it.
 * The request and answer behaviour of the windowed call is in heal-window.spec.ts.
 *
 * Each test states, in a `Breaks if:` line, the defect that would make it fail.
 *
 * @module dsh-talk/test/heal.spec
 */

import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'
import { HEAL_ENDPOINT, HEAL_MODEL, HEAL_TIMEOUT_MS, HEAL_WINDOW_SYSTEM_PROMPT } from '../src/heal.ts'

describe('contract constants', () => {
  it('points at the Bifrost alias for the resident 27B named in SPEC.md', () => {
    // Breaks if: the endpoint, model, or port changes without the spec changing with it.
    // The alias is used, not the served id: Bifrost logs every call it relays.
    expect(HEAL_ENDPOINT).toBe('http://127.0.0.1:8090/v1/chat/completions')
    expect(HEAL_MODEL).toBe('strata/qwen-strata')
  })

  it('keeps the timeout above the 8.58 s outlier in research/1cds-heal-recheck.md', () => {
    // Breaks if: someone lowers the timeout below the one measured idle-gap call.
    expect(HEAL_TIMEOUT_MS).toBeGreaterThan(8_580)
  })
})

describe('byte identity with HEAL-PROMPT.md', () => {
  // Runs only when DSH_TALK_HEAL_PROMPT names the file. Unset, the test is reported as skipped.
  const healPromptFile = process.env.DSH_TALK_HEAL_PROMPT ?? ''

  it.skipIf(healPromptFile === '')(
    'HEAL_WINDOW_SYSTEM_PROMPT equals the fenced block under "## System prompt, verbatim", byte for byte',
    async () => {
      // Breaks if: one character of the constant differs from the file, the heading or fence
      // moves, or the constant is edited without HEAL-PROMPT.md.
      const markdown = await readFile(healPromptFile, 'utf8')
      const headingAt = markdown.indexOf('## System prompt, verbatim')
      expect(headingAt, 'heading present').toBeGreaterThanOrEqual(0)
      const opening = markdown.indexOf('\n```\n', headingAt)
      expect(opening, 'opening fence after the heading').toBeGreaterThanOrEqual(0)
      const contentStart = opening + '\n```\n'.length
      const closing = markdown.indexOf('\n```', contentStart)
      expect(closing, 'closing fence').toBeGreaterThan(contentStart)
      const block = markdown.slice(contentStart, closing)
      expect(Buffer.from(block, 'utf8').equals(Buffer.from(HEAL_WINDOW_SYSTEM_PROMPT, 'utf8'))).toBe(true)
    },
  )
})
