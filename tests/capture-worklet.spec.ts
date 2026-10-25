/**
 * The capture worklet's processor, run in a Node VM with fake worklet globals. The stop handshake
 * is checked here: a flush message makes the processor post the partial block it still holds, so
 * the audio of the last 1023 samples or fewer is not lost when the user stops.
 */

import vm from 'node:vm'
import { describe, expect, it } from 'vitest'
import { CAPTURE_BLOCK_SAMPLES, CAPTURE_FLUSH_MESSAGE, CAPTURE_WORKLET_SOURCE } from '../src/client/capture-worklet.ts'

/** The processor as the worklet source defines it: fed render quanta, and driven through its port. */
interface Processor {
  port: { onmessage: ((event: { data: unknown }) => void) | null }
  process: (inputs: Float32Array[][]) => boolean
}

/** One processor instance, plus every message it posted to the main thread, in order. */
function loadProcessor(): { processor: Processor; posted: unknown[] } {
  const posted: unknown[] = []
  let registered: (new () => Processor) | undefined
  class FakeAudioWorkletProcessor {
    port: { onmessage: ((event: { data: unknown }) => void) | null; postMessage: (data: unknown) => void }

    constructor() {
      this.port = { onmessage: null, postMessage: (data: unknown) => { posted.push(data) } }
    }
  }
  vm.runInNewContext(CAPTURE_WORKLET_SOURCE, {
    AudioWorkletProcessor: FakeAudioWorkletProcessor,
    registerProcessor: (_name: string, processor: new () => Processor) => {
      registered = processor
    },
    Float32Array,
  })
  if (registered === undefined) throw new Error('the worklet source did not register its processor')
  return { processor: new registered(), posted }
}

/** Feed `values` to the processor in 128-frame render quanta, the size the audio thread uses. */
function feed(processor: Processor, values: Float32Array): void {
  for (let at = 0; at < values.length; at += 128) processor.process([[values.subarray(at, at + 128)]])
}

/** Send the stop message the main thread sends when the user stops. */
function flush(processor: Processor): void {
  processor.port.onmessage?.({ data: CAPTURE_FLUSH_MESSAGE })
}

describe('capture worklet stop handshake', () => {
  it('a flush posts the partial block the processor still holds, with every sample', () => {
    // Seven render quanta of 128 samples: 896 samples, under the 1024-sample block, so no full block is posted.
    const { processor, posted } = loadProcessor()
    feed(processor, new Float32Array(7 * 128).fill(0.2))
    expect(posted).toEqual([])
    flush(processor)
    expect(posted).toHaveLength(1)
    const reply = posted[0] as { kind: string; samples: Float32Array }
    expect(reply.kind).toBe('flushed')
    expect(reply.samples.length).toBe(7 * 128)
    expect(reply.samples.every(value => value === Math.fround(0.2))).toBe(true)
  })

  it('full blocks go out first, in order, and the flush carries the rest', () => {
    // A ramp, so a reordered or dropped sample shows as a wrong value. 1152 = one full block plus 128.
    const { processor, posted } = loadProcessor()
    const ramp = Float32Array.from({ length: CAPTURE_BLOCK_SAMPLES + 128 }, (_, index) => index)
    feed(processor, ramp)
    expect(posted).toHaveLength(1)
    const full = posted[0] as Float32Array
    expect(full).toBeInstanceOf(Float32Array)
    expect(full.length).toBe(CAPTURE_BLOCK_SAMPLES)
    expect(full[0]).toBe(0)
    expect(full[CAPTURE_BLOCK_SAMPLES - 1]).toBe(CAPTURE_BLOCK_SAMPLES - 1)
    flush(processor)
    const reply = posted[1] as { kind: string; samples: Float32Array }
    expect(reply.kind).toBe('flushed')
    expect(Array.from(reply.samples)).toEqual(Array.from({ length: 128 }, (_, index) => CAPTURE_BLOCK_SAMPLES + index))
  })

  it('a flush right after a full block replies with an empty partial block', () => {
    const { processor, posted } = loadProcessor()
    feed(processor, new Float32Array(CAPTURE_BLOCK_SAMPLES).fill(0.5))
    expect(posted).toHaveLength(1)
    flush(processor)
    const reply = posted[1] as { kind: string; samples: Float32Array }
    expect(reply.kind).toBe('flushed')
    expect(reply.samples.length).toBe(0)
  })
})
