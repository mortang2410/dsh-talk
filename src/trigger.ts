/**
 * Trigger-phrase matcher for the voice loop. Finds where the spoken phrase
 * "that is my prompt", or a spelling a speech recogniser produces for it,
 * starts. That index is where the prompt ends, and the text before it is the
 * prompt. Pure: no I/O, no clock, no Node or DOM API, because the browser
 * client imports this module too.
 *
 * @module dsh-talk/trigger
 */

/**
 * The trigger phrase, case-insensitive. The lead is "that is my", "that's my"
 * or "thats my"; the noun is "prompt" or a spelling a recogniser produces for
 * it ("problem", "promt", "promp", "prompted").
 *
 * The lookarounds require a whole word at both ends. JavaScript's `\b` only
 * knows ASCII letters, so it would let "promptéd" match; these test Unicode
 * letters and digits instead.
 *
 * No `g` flag: a global regex keeps `lastIndex` between calls, so the same
 * text would match on one call and miss on the next.
 */
export const TRIGGER_PHRASE =
  /(?<![\p{L}\p{N}_])that(?: is |'s |s )my (?:prompted|prompt|promp|promt|problem)/iu

/**
 * Characters removed from the end of a prompt: whitespace, plus the comma,
 * semicolon and colon a speaker says just before the phrase ("read the config,
 * that is my prompt"). A full stop ends a sentence, so it is not here.
 */
const CUT_BREAK = /[\s,;:]/u

/**
 * Index in `text` where the first trigger phrase starts. That index is where
 * the prompt ends.
 *
 * @param text - one transcript.
 * @returns the index of the phrase, `0` when the text opens with it, or `null` when there is no phrase.
 */
export function triggerEnd(text: string): number | null {
  const match = TRIGGER_PHRASE.exec(text)
  return match === null ? null : match.index
}

/**
 * The prompt in `text`: the text before the first trigger phrase, with the
 * trailing whitespace and `,` `;` `:` run removed. With no phrase, the whole
 * text, trimmed.
 *
 * @param text - one transcript.
 * @returns the prompt; the empty string when the phrase opens the text.
 */
export function triggerPrompt(text: string): string {
  const end = triggerEnd(text)
  if (end === null) return text.trim()
  // A hand-written scan, not a pattern such as /[\s,;:]+$/: that pattern takes
  // quadratic time on a long run of spaces, and a transcript can carry one.
  let cut = end
  while (cut > 0 && CUT_BREAK.test(text.charAt(cut - 1))) cut -= 1
  return text.slice(0, cut)
}
