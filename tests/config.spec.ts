/**
 * Config contract: the Schemastery schema fills leaf defaults on an empty
 * input, rejects out-of-range values loudly, and `resolveConfig` re-judges
 * every default for programmatic mounts plus the cross-field engine checks.
 *
 * @module dsh-talk/test/config.spec
 */

import { describe, expect, it } from 'vitest'
import { Config, resolveConfig, MAX_CEILING_SECONDS, MAX_RECORD_SECONDS } from '../src/config.ts'
import { resolveSttEngine, resolveTtsEngine } from '../src/service.ts'

describe('Config schema', () => {
  it('fills every leaf default on an empty input', () => {
    const normalized = Config({}) as { record: { enabled: boolean }; stt: { engine: string; language: string; interim: boolean; silenceFinaliseMs: number }; tts: { engine: string; rate: number }; announce: { enabled: boolean; messages: { turnEnd: string } } }
    expect(normalized.record.enabled).toBe(true)
    expect(normalized.stt.engine).toBe('auto')
    expect(normalized.stt.language).toBe('auto')
    expect(normalized.stt.interim).toBe(true)
    expect(normalized.stt.silenceFinaliseMs).toBe(4000)
    expect(normalized.tts.engine).toBe('auto')
    expect(normalized.tts.rate).toBe(0)
    expect(normalized.announce.enabled).toBe(true)
    expect(normalized.announce.messages.turnEnd).toBe('Turn complete.')
  })

  it('keeps nested defaults when only a leaf is overridden', () => {
    const normalized = Config({ record: { autoSubmit: true } }) as { record: { autoSubmit: boolean; enabled: boolean }; stt: { interim: boolean } }
    expect(normalized.record.autoSubmit).toBe(true)
    expect(normalized.record.enabled).toBe(true)
    expect(normalized.stt.interim).toBe(true)
  })

  it('rejects a recording cap above the ceiling', () => {
    expect(() => Config({ record: { maxSeconds: MAX_RECORD_SECONDS + 1 } })).toThrow()
  })

  it('defaults the segmented-capture ceiling to 600 seconds and rejects a value above its own maximum', () => {
    // record.ceilingSeconds governs only the segmented capture; record.maxSeconds stays the legacy cap.
    const normalized = Config({}) as { record: { ceilingSeconds: number; maxSeconds: number } }
    expect(normalized.record.ceilingSeconds).toBe(600)
    expect(normalized.record.maxSeconds).toBe(60)
    expect(resolveConfig({}).recordCeilingSeconds).toBe(600)
    expect(() => Config({ record: { ceilingSeconds: MAX_CEILING_SECONDS + 1 } })).toThrow()
  })

  it('rejects an unknown STT engine', () => {
    expect(() => Config({ stt: { engine: 'watson' as 'web' } })).toThrow()
  })
})

describe('resolveConfig', () => {
  it('re-validates bounds for a programmatic mount', () => {
    expect(() => resolveConfig({ maxSpeakChars: 0 })).toThrow(/maxSpeakChars/u)
    expect(() => resolveConfig({ stt: { silenceFinaliseMs: 499 } })).toThrow(/silenceFinaliseMs/u)
    expect(() => resolveConfig({ stt: { silenceFinaliseMs: 15_001 } })).toThrow(/silenceFinaliseMs/u)
  })

  it('fails loud when funasr is selected without an endpoint', () => {
    expect(() => resolveConfig({ stt: { engine: 'funasr' } })).toThrow(/funasr/u)
  })

  it('fails loud when piper is selected without a model', () => {
    expect(() => resolveConfig({ tts: { engine: 'piper' } })).toThrow(/piper/u)
  })

  it('accepts the auto engine without any local model', () => {
    const resolved = resolveConfig({})
    expect(resolved.sttEngine).toBe('auto')
    expect(resolved.ttsEngine).toBe('auto')
  })

  it('keeps the default announce messages', () => {
    const resolved = resolveConfig({})
    expect(resolved.messageApproval).toBe('Approval required.')
  })

  it('resolves the VAD defaults and rejects an out-of-range silence window', () => {
    const resolved = resolveConfig({})
    expect(resolved.vadEnabled).toBe(true)
    expect(resolved.vadSilenceMs).toBe(1_500)
    expect(resolved.vadEnergyThreshold).toBe(0.01)
    expect(() => resolveConfig({ record: { vad: { silenceMs: 10 } } })).toThrow(/silenceMs/u)
  })
})

describe('STT engine with blank endpoints', () => {
  // cordis.patch.yml ships funasr.url, funasr.model and whisper.modelPath as ''.
  // A blank value means "not configured": auto must resolve to web, not funasr.
  it('resolves the shipped default (auto, blank endpoints) to web', () => {
    const resolved = resolveConfig({ stt: { engine: 'auto', funasr: { url: '', model: '' }, whisper: { modelPath: '' } } })
    expect(resolved.funasrUrl).toBeNull()
    expect(resolved.funasrModel).toBeNull()
    expect(resolved.whisperModelPath).toBeNull()
    expect(resolveSttEngine(resolved)).toBe('web')
  })

  it('resolves an empty config to web', () => {
    expect(resolveSttEngine(resolveConfig({}))).toBe('web')
  })

  it('still resolves auto to funasr once a url is set', () => {
    const resolved = resolveConfig({ stt: { engine: 'auto', funasr: { url: 'http://127.0.0.1:8002', model: 'nemotron-3.5' } } })
    expect(resolved.funasrUrl).toBe('http://127.0.0.1:8002')
    expect(resolveSttEngine(resolved)).toBe('funasr')
  })

  it('rejects funasr and whisper selected with blank endpoints', () => {
    expect(() => resolveConfig({ stt: { engine: 'funasr', funasr: { url: '' } } })).toThrow(/funasr.url is not set/u)
    expect(() => resolveConfig({ stt: { engine: 'whisper', whisper: { modelPath: '' } } })).toThrow(/whisper.modelPath is not set/u)
  })
})

describe('TTS engine with a blank piper model path', () => {
  // cordis.patch.yml ships tts.piper.modelPath as ''. A blank value means "not
  // configured": auto must resolve to edge-tts, not piper.
  it('resolves the shipped default (auto, blank modelPath) to edge-tts', () => {
    const resolved = resolveConfig({ tts: { engine: 'auto', piper: { modelPath: '' } } })
    expect(resolved.piperModelPath).toBeNull()
    expect(resolveTtsEngine(undefined, resolved)).toBe('edge-tts')
  })

  it('rejects piper selected with a blank modelPath', () => {
    expect(() => resolveConfig({ tts: { engine: 'piper', piper: { modelPath: '' } } })).toThrow(/piper.modelPath is not set/u)
  })

  it('still resolves auto to piper once a model is set', () => {
    const resolved = resolveConfig({ tts: { engine: 'auto', piper: { modelPath: 'en.onnx' } } })
    expect(resolved.piperModelPath).toBe('en.onnx')
    expect(resolveTtsEngine(undefined, resolved)).toBe('piper')
  })
})
