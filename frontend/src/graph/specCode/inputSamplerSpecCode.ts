import type { InputSamplerNodeData } from "../../components/nodes/inputSamplerDefaults";
import { defaultInputSamplerData } from "../../components/nodes/inputSamplerDefaults";

export const DEFAULT_INPUT_SAMPLER_SPEC_NAME = "InputSampler";

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

export function generateInputSamplerSpecCode(d: InputSamplerNodeData, specName: string): string {
  const name = specName.trim() || DEFAULT_INPUT_SAMPLER_SPEC_NAME;
  const merged = { ...defaultInputSamplerData(), ...d };
  const numSamples = Math.max(1, Number(firstScalar(merged.numSamples) ?? 800) || 800);
  return [
    `def ${name}(`,
    `    num_samples: int = ${numSamples},`,
    `):`,
    `    """`,
    `    Sample x from a wired random-input distribution config.`,
    `    Runtime uses comfy_research.engine.random_input_distribution_runtime.sample_x_from_sampler_dict.`,
    `    """`,
    `    from comfy_research.engine.random_input_distribution_runtime import sample_x_from_sampler_dict`,
    ``,
    `    sampler = {"numSamples": max(1, int(num_samples))}`,
    `    rid = {`,
    `        "inputDim": 10,`,
    `        "inputDistribution": "standard_normal",`,
    `        "noiseDistribution": "deterministic",`,
    `        "noiseLevel": 0.0,`,
    `        "seed": 0,`,
    `    }`,
    `    return sample_x_from_sampler_dict(sampler, rid)`,
  ].join("\n");
}
