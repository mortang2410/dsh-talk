/**
 * The host-side transcript trail: one JSON line per stop, appended to a monthly file.
 *
 * The user's decision (`B1`, trust the model to clean) rests on being able to compare the
 * cleaned prompt with what the recogniser actually produced, so the raw transcript is kept
 * beside the cleaned text and the action the stop took with it. The files are small (about
 * 1 KB per 5 stops), so nothing is pruned; delete a month's file to drop it.
 *
 * @module dsh-talk/transcript-log
 */

import { appendFile, mkdir } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { TalkTranscriptLogEntry } from './wire.ts'

/** Default trail directory, under the host's own state dir. */
export function transcriptLogDir(): string {
  return join(homedir(), '.dsh', 'logs', 'dsh-talk')
}

/** The monthly trail file for `now`, named by its UTC month. */
export function transcriptLogPath(now: Date = new Date(), dir: string = transcriptLogDir()): string {
  return join(dir, `transcripts-${now.toISOString().slice(0, 7)}.jsonl`)
}

/**
 * Append one stop to the trail.
 *
 * @param entry - the raw and cleaned text, their source, and the action the stop took.
 * @param options - directory and clock overrides, used by tests.
 * @returns absolute path of the file the line was appended to.
 */
export async function appendTranscriptLog(
  entry: TalkTranscriptLogEntry,
  options: { dir?: string; now?: Date } = {},
): Promise<string> {
  const now = options.now ?? new Date()
  const dir = options.dir ?? transcriptLogDir()
  const path = transcriptLogPath(now, dir)
  const line = JSON.stringify({ ts: now.toISOString(), ...entry })
  await mkdir(dir, { recursive: true, mode: 0o700 })
  await appendFile(path, `${line}\n`, { mode: 0o600 })
  return path
}
