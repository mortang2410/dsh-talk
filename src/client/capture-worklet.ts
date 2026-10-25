/**
 * The microphone capture worklet. An AudioWorkletProcessor has to load as its own module, so
 * its source is kept here as a string and loaded through a Blob URL. The processor downmixes
 * each input frame to mono and posts every CAPTURE_BLOCK_SAMPLES block to the main thread as a
 * transferred Float32Array. The segmenter then judges every block, so no audio goes unjudged.
 *
 * @module dsh-talk/client/capture-worklet
 */

/** The name the AudioWorkletNode is created with. The processor registers under the same name. */
export const CAPTURE_WORKLET_NAME = 'dsh-talk-capture'

/**
 * Samples per posted block. At 48 kHz, 1024 samples is about 21 ms: a speech onset is judged
 * within one block, and the message rate stays low.
 */
export const CAPTURE_BLOCK_SAMPLES = 1024

/**
 * The processor source. It runs on the audio rendering thread, so it uses only the worklet
 * globals (AudioWorkletProcessor and registerProcessor) and imports nothing.
 */
export const CAPTURE_WORKLET_SOURCE = `
class DshTalkCapture extends AudioWorkletProcessor {
  constructor() {
    super()
    this.block = new Float32Array(${CAPTURE_BLOCK_SAMPLES})
    this.filled = 0
  }

  process(inputs) {
    const channels = inputs[0]
    if (!channels || channels.length === 0) return true
    const frames = channels[0].length
    for (let frame = 0; frame < frames; frame += 1) {
      let sum = 0
      for (let channel = 0; channel < channels.length; channel += 1) sum += channels[channel][frame]
      this.block[this.filled] = sum / channels.length
      this.filled += 1
      if (this.filled === this.block.length) {
        const full = this.block
        this.port.postMessage(full, [full.buffer])
        this.block = new Float32Array(${CAPTURE_BLOCK_SAMPLES})
        this.filled = 0
      }
    }
    return true
  }
}

registerProcessor('${CAPTURE_WORKLET_NAME}', DshTalkCapture)
`
