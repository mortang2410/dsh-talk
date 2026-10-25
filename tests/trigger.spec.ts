/**
 * The trigger-phrase matcher: where "that is my prompt" and its recogniser
 * misspellings end an utterance, and the prompt that comes before them.
 *
 * @module dsh-talk/test/trigger.spec
 */

import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { triggerEnd, triggerPrompt } from '../src/trigger.ts'

/** Ordinary speech before the phrase, so the cut lands mid-text. */
const LEAD = 'read the config '

describe('triggerEnd accepts every trigger form', () => {
  // Breaks if: any spelling the ticket names stops matching, so the speaker's prompt never ends.
  it.each([
    'that is my prompt',
    "that's my prompt",
    'thats my prompt',
    'that is my problem',
    "that's my problem",
    'thats my problem',
    'that is my promt',
    "that's my promt",
    'thats my promt',
    'that is my promp',
    "that's my promp",
    'thats my promp',
    'that is my prompted',
    "that's my prompted",
    'thats my prompted',
  ])('matches "%s"', (phrase) => {
    expect(triggerEnd(LEAD + phrase)).toBe(LEAD.length)
  })

  // Breaks if: the match is case-sensitive, so a recogniser's capitalised output sends no prompt.
  it.each(['That Is My Prompt', 'THATS MY PROMPT', "That's My Prompted", 'That IS my PROBLEM'])(
    'matches "%s" in any case',
    (phrase) => {
      expect(triggerEnd(LEAD + phrase)).toBe(LEAD.length)
    },
  )

  // Breaks if: a phrase at the start of the text reports anything but 0, so an empty prompt reads as no trigger.
  it('returns 0 when the phrase is the whole text', () => {
    expect(triggerEnd('that is my prompt')).toBe(0)
  })

  // Breaks if: the matcher only looks at the end of the text, so a phrase followed by more speech is missed.
  it('finds the phrase in the middle of the text', () => {
    expect(triggerEnd(LEAD + 'that is my prompt and then more speech')).toBe(LEAD.length)
  })

  // Breaks if: the last occurrence wins instead of the first, so the cut lands after an earlier phrase.
  it('cuts at the first occurrence', () => {
    expect(triggerEnd(LEAD + 'that is my problem, then more, that is my prompt')).toBe(LEAD.length)
  })

  // Breaks if: the regex keeps state between calls (a g flag), so the same text matches once and then misses.
  it('gives the same answer on repeated calls', () => {
    const text = LEAD + 'that is my prompt'
    expect(triggerEnd(text)).toBe(LEAD.length)
    expect(triggerEnd(text)).toBe(LEAD.length)
  })
})

describe('triggerEnd returns null when there is no phrase', () => {
  // Breaks if: text without the phrase returns an index instead of null, so the caller cuts a prompt that was never triggered.
  it.each(['', 'read the config', 'that is', 'my prompt'])('returns null for "%s"', (text) => {
    expect(triggerEnd(text)).toBeNull()
  })
})

describe('triggerEnd requires whole words at both ends of the phrase', () => {
  // Breaks if: the end boundary is dropped, so a longer word that starts with a form ("promptedx", "problems") is taken as the phrase.
  it.each(['that is my promptedx', 'that is my problems', 'that is my prompts', 'that is my promptéd'])(
    'does not match inside "%s"',
    (phrase) => {
      expect(triggerEnd(LEAD + phrase)).toBeNull()
    },
  )

  // Breaks if: the start boundary is dropped, so a phrase glued onto the end of a longer word ("rethat") is cut.
  it('does not match a phrase glued onto a longer word', () => {
    expect(triggerEnd('rethat is my prompt')).toBeNull()
  })
})

describe('triggerEnd does not match words that only look like a form', () => {
  // Breaks if: a noun that shares a prefix with a form ("pocket", "profit", "proper") joins the set, so ordinary speech ends a prompt.
  it.each(['that is my pocket', 'that is my profit', 'that is my proper'])('does not match "%s"', (phrase) => {
    expect(triggerEnd(LEAD + phrase)).toBeNull()
  })

  // Breaks if: "is" becomes optional in the lead, so "that my prompt" is taken as the phrase.
  it('does not match "that my prompt" without "is"', () => {
    expect(triggerEnd(LEAD + 'that my prompt')).toBeNull()
  })
})

describe('triggerPrompt', () => {
  // Breaks if: the comma before the phrase stays, so the prompt is "read the config," instead of "read the config".
  it('removes the comma that precedes the phrase', () => {
    expect(triggerPrompt('read the config, that is my prompt')).toBe('read the config')
  })

  // Breaks if: a semicolon or colon before the phrase survives the cut.
  it.each(['read the config; that is my prompt', 'read the config: that is my prompt'])(
    'removes a semicolon or colon before the phrase in "%s"',
    (text) => {
      expect(triggerPrompt(text)).toBe('read the config')
    },
  )

  // Breaks if: the trim stops at the first separator of a mixed run, leaving a dangling mark behind.
  it('removes a mixed run of separators and spaces', () => {
    expect(triggerPrompt('read the config ,; : that is my prompt')).toBe('read the config')
  })

  // Breaks if: the trim also removes a full stop, so the sentence loses its own ending.
  it('keeps a full stop before the phrase', () => {
    expect(triggerPrompt('read the config. that is my prompt')).toBe('read the config.')
  })

  // Breaks if: a phrase at index 0 returns anything but the empty string, so an empty prompt is reported as something else.
  it('returns the empty string when the phrase is the whole text', () => {
    expect(triggerPrompt('that is my prompt')).toBe('')
  })

  // Breaks if: the words after the first phrase are kept, so speech said after the trigger is sent inside the prompt.
  it('drops the text after the first phrase', () => {
    expect(triggerPrompt(LEAD + 'that is my prompt and more speech')).toBe('read the config')
  })

  // Breaks if: the no-match path skips the trim, so padding stays on text that was never triggered.
  it('returns text with no phrase, trimmed', () => {
    expect(triggerPrompt('  read the config  ')).toBe('read the config')
    expect(triggerPrompt('')).toBe('')
  })
})

describe('trigger module purity', () => {
  // Breaks if: the module gains an import, such as a Node built-in, which the browser client cannot load.
  it('has no import statements and no node: references', () => {
    const source = readFileSync(new URL('../src/trigger.ts', import.meta.url), 'utf8')
    expect(source).not.toMatch(/^\s*import\b/mu)
    expect(source).not.toMatch(/\bnode:/u)
  })
})
