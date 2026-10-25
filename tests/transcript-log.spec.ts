/**
 * The host-side transcript trail: one JSON line per stop, appended to a monthly file so the
 * raw words stay beside the cleaned text. The file, its mode, and the monthly name are the
 * contract; the tests use their own directory and clock.
 *
 * @module dsh-talk/test/transcript-log
 */

import { chmod, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { appendTranscriptLog, transcriptLogDir, transcriptLogPath } from '../src/transcript-log.ts'
import { TALK_TRANSCRIPT_LOG_SCHEMA, type TalkTranscriptLogEntry } from '../src/wire.ts'

const dirs: string[] = []

afterEach(async () => {
  await Promise.all(dirs.splice(0).map(dir => rm(dir, { recursive: true, force: true })))
})

async function tempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'dsh-talk-trail-'))
  dirs.push(dir)
  return dir
}

const ENTRY = { raw: 'um first sentence', cleaned: 'first sentence', source: 'model', action: 'draft' } as const

describe('the transcript trail', () => {
  it('appends one JSON line with the timestamp and every field, and returns its path', async () => {
    // Breaks if: the line loses a field, the timestamp is missing, or the path is not the one written.
    const dir = await tempDir()
    const now = new Date('2026-10-25T18:04:05.000Z')
    const path = await appendTranscriptLog(ENTRY, { dir, now })
    expect(path).toBe(join(dir, 'transcripts-2026-10.jsonl'))
    const lines = (await readFile(path, 'utf8')).trimEnd().split('\n')
    expect(lines).toHaveLength(1)
    expect(JSON.parse(lines[0] ?? '')).toEqual({ ts: '2026-10-25T18:04:05.000Z', ...ENTRY })
  })

  it('appends a second stop to the same monthly file, in order', async () => {
    // Breaks if: each stop overwrites the file, or a later stop lands in a different file.
    const dir = await tempDir()
    const now = new Date('2026-10-25T18:04:05.000Z')
    await appendTranscriptLog(ENTRY, { dir, now })
    await appendTranscriptLog({ raw: 'second words', cleaned: 'second words', source: 'raw', action: 'submit' }, { dir, now })
    const lines = (await readFile(join(dir, 'transcripts-2026-10.jsonl'), 'utf8')).trimEnd().split('\n')
    expect(lines.map(line => (JSON.parse(line) as { raw: string }).raw)).toEqual(['um first sentence', 'second words'])
  })

  it('creates the directory 0700 and the file 0600', async () => {
    // Breaks if: the trail is world-readable on a shared host.
    const dir = join(await tempDir(), 'nested', 'dsh-talk')
    const path = await appendTranscriptLog(ENTRY, { dir, now: new Date('2026-10-25T00:00:00.000Z') })
    expect((await stat(dir)).mode & 0o777).toBe(0o700)
    expect((await stat(path)).mode & 0o777).toBe(0o600)
  })

  it('tightens the modes of an existing directory and file before appending', async () => {
    // Breaks if: a pre-created 0755 dir and 0644 file let another local user read the raw words.
    // The mode options only apply at creation, so an existing path keeps whatever it had.
    const dir = join(await tempDir(), 'dsh-talk')
    await mkdir(dir, { recursive: true })
    await chmod(dir, 0o755)
    const path = join(dir, 'transcripts-2026-10.jsonl')
    await writeFile(path, `${JSON.stringify({ ts: '2026-09-30T00:00:00.000Z', ...ENTRY })}\n`)
    await chmod(path, 0o644)
    await appendTranscriptLog(ENTRY, { dir, now: new Date('2026-10-25T18:04:05.000Z') })
    expect((await stat(dir)).mode & 0o777).toBe(0o700)
    expect((await stat(path)).mode & 0o777).toBe(0o600)
  })

  it('the trail source is model or raw only, and the schema refuses any other value', () => {
    // Breaks if: the schema accepts any source other than model or raw, or refuses either of those two.
    // @ts-expect-error: only model and raw are trail sources, so the type must refuse any other value at compile time.
    const other: TalkTranscriptLogEntry = { ...ENTRY, source: 'unknown' }
    expect(TALK_TRANSCRIPT_LOG_SCHEMA.safeParse(other).success).toBe(false)
    expect(TALK_TRANSCRIPT_LOG_SCHEMA.safeParse({ ...ENTRY, source: 'model' }).success).toBe(true)
    expect(TALK_TRANSCRIPT_LOG_SCHEMA.safeParse({ ...ENTRY, source: 'raw' }).success).toBe(true)
  })

  it('names the file by the UTC month and defaults to the host state dir', async () => {
    // Breaks if: the month comes from the local clock, or the default leaves the host's state dir.
    expect(transcriptLogPath(new Date('2026-10-31T23:30:00.000Z'), '/x')).toBe('/x/transcripts-2026-10.jsonl')
    expect(transcriptLogPath(new Date('2026-11-01T00:30:00.000Z'), '/x')).toBe('/x/transcripts-2026-11.jsonl')
    expect(transcriptLogDir().endsWith('/.dsh/logs/dsh-talk')).toBe(true)
  })
})
