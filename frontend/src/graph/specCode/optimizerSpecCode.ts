import type { AdamOptimizerNodeData } from "../../components/nodes/adamOptimizerDefaults";
import { defaultAdamOptimizerData } from "../../components/nodes/adamOptimizerDefaults";
import type { AdamWOptimizerNodeData } from "../../components/nodes/adamWOptimizerDefaults";
import { defaultAdamWOptimizerData } from "../../components/nodes/adamWOptimizerDefaults";
import type { MuonOptimizerNodeData } from "../../components/nodes/muonOptimizerDefaults";
import { defaultMuonOptimizerData } from "../../components/nodes/muonOptimizerDefaults";
import type { SgdOptimizerNodeData } from "../../components/nodes/sgdOptimizerDefaults";
import { defaultSgdOptimizerData } from "../../components/nodes/sgdOptimizerDefaults";
import type { SignSgdOptimizerNodeData } from "../../components/nodes/signSgdOptimizerDefaults";
import { defaultSignSgdOptimizerData } from "../../components/nodes/signSgdOptimizerDefaults";
import type { ShampooOptimizerNodeData } from "../../components/nodes/shampooOptimizerDefaults";
import { defaultShampooOptimizerData } from "../../components/nodes/shampooOptimizerDefaults";
import type { SoapOptimizerNodeData } from "../../components/nodes/soapOptimizerDefaults";
import { defaultSoapOptimizerData } from "../../components/nodes/soapOptimizerDefaults";

function firstNumber(v: unknown, fallback: number): number {
  const x = Array.isArray(v) ? v[0] : v;
  return typeof x === "number" && Number.isFinite(x) ? x : fallback;
}

export function generateSgdOptimizerSpecCode(d: SgdOptimizerNodeData): string {
  const merged = { ...defaultSgdOptimizerData(), ...d };
  const learningRate = firstNumber(merged.learningRate, 0.01);
  const momentum = firstNumber(merged.momentum, 0);
  const weightDecay = firstNumber(merged.weightDecay, 0);
  return [
    "def SGDOptimizer(",
    `    learning_rate: float = ${learningRate},`,
    `    momentum: float = ${momentum},`,
    `    weight_decay: float = ${weightDecay},`,
    "):",
    '    """Construct an SGD optimizer config."""',
    "    return {",
    '        "type": "sgd",',
    '        "learning_rate": float(learning_rate),',
    '        "momentum": float(momentum),',
    '        "weight_decay": float(weight_decay),',
    "    }",
  ].join("\n");
}

export function generateAdamOptimizerSpecCode(d: AdamOptimizerNodeData): string {
  const merged = { ...defaultAdamOptimizerData(), ...d };
  const learningRate = firstNumber(merged.learningRate, 0.001);
  const beta1 = firstNumber(merged.beta1, 0.9);
  const beta2 = firstNumber(merged.beta2, 0.999);
  const epsilon = firstNumber(merged.epsilon, 1e-8);
  const weightDecay = firstNumber(merged.weightDecay, 0);
  return [
    "def AdamOptimizer(",
    `    learning_rate: float = ${learningRate},`,
    `    beta1: float = ${beta1},`,
    `    beta2: float = ${beta2},`,
    `    epsilon: float = ${epsilon},`,
    `    weight_decay: float = ${weightDecay},`,
    "):",
    '    """Construct an Adam optimizer config."""',
    "    return {",
    '        "type": "adam",',
    '        "learning_rate": float(learning_rate),',
    '        "beta1": float(beta1),',
    '        "beta2": float(beta2),',
    '        "epsilon": float(epsilon),',
    '        "weight_decay": float(weight_decay),',
    "    }",
  ].join("\n");
}

export function generateAdamWOptimizerSpecCode(d: AdamWOptimizerNodeData): string {
  const merged = { ...defaultAdamWOptimizerData(), ...d };
  const learningRate = firstNumber(merged.learningRate, 0.001);
  const beta1 = firstNumber(merged.beta1, 0.9);
  const beta2 = firstNumber(merged.beta2, 0.999);
  const epsilon = firstNumber(merged.epsilon, 1e-8);
  const weightDecay = firstNumber(merged.weightDecay, 0.01);
  return [
    "def AdamWOptimizer(",
    `    learning_rate: float = ${learningRate},`,
    `    beta1: float = ${beta1},`,
    `    beta2: float = ${beta2},`,
    `    epsilon: float = ${epsilon},`,
    `    weight_decay: float = ${weightDecay},`,
    "):",
    '    """Construct an AdamW optimizer config."""',
    "    return {",
    '        "type": "adamw",',
    '        "learning_rate": float(learning_rate),',
    '        "beta1": float(beta1),',
    '        "beta2": float(beta2),',
    '        "epsilon": float(epsilon),',
    '        "weight_decay": float(weight_decay),',
    "    }",
  ].join("\n");
}

export function generateSignSgdOptimizerSpecCode(d: SignSgdOptimizerNodeData): string {
  const merged = { ...defaultSignSgdOptimizerData(), ...d };
  const learningRate = firstNumber(merged.learningRate, 0.001);
  const weightDecay = firstNumber(merged.weightDecay, 0);
  return [
    "def SignSGDOptimizer(",
    `    learning_rate: float = ${learningRate},`,
    `    weight_decay: float = ${weightDecay},`,
    "):",
    '    """Construct a SignSGD optimizer config."""',
    "    return {",
    '        "type": "signsgd",',
    '        "learning_rate": float(learning_rate),',
    '        "weight_decay": float(weight_decay),',
    "    }",
  ].join("\n");
}

export function generateMuonOptimizerSpecCode(d: MuonOptimizerNodeData): string {
  const merged = { ...defaultMuonOptimizerData(), ...d };
  const learningRate = firstNumber(merged.learningRate, 0.003);
  const momentum = firstNumber(merged.momentum, 0.95);
  return [
    "def MuonOptimizer(",
    `    learning_rate: float = ${learningRate},`,
    `    momentum: float = ${momentum},`,
    "):",
    '    """Construct a Muon optimizer config."""',
    "    return {",
    '        "type": "muon",',
    '        "learning_rate": float(learning_rate),',
    '        "momentum": float(momentum),',
    "    }",
  ].join("\n");
}

export function generateShampooOptimizerSpecCode(d: ShampooOptimizerNodeData): string {
  const merged = { ...defaultShampooOptimizerData(), ...d };
  const learningRate = firstNumber(merged.learningRate, 0.01);
  const momentum = firstNumber(merged.momentum, 0);
  const epsilon = firstNumber(merged.epsilon, 1e-8);
  const weightDecay = firstNumber(merged.weightDecay, 0);
  const preconditionFrequency = Math.max(1, Math.round(firstNumber(merged.preconditionFrequency, 10)));
  const maxPreconditionerDim = Math.max(1, Math.round(firstNumber(merged.maxPreconditionerDim, 1024)));
  return [
    "def ShampooOptimizer(",
    `    learning_rate: float = ${learningRate},`,
    `    momentum: float = ${momentum},`,
    `    epsilon: float = ${epsilon},`,
    `    weight_decay: float = ${weightDecay},`,
    `    precondition_frequency: int = ${preconditionFrequency},`,
    `    max_preconditioner_dim: int = ${maxPreconditionerDim},`,
    "):",
    '    """Construct a Shampoo optimizer config."""',
    "    return {",
    '        "type": "shampoo",',
    '        "learning_rate": float(learning_rate),',
    '        "momentum": float(momentum),',
    '        "epsilon": float(epsilon),',
    '        "weight_decay": float(weight_decay),',
    '        "precondition_frequency": int(precondition_frequency),',
    '        "max_preconditioner_dim": int(max_preconditioner_dim),',
    "    }",
  ].join("\n");
}

export function generateSoapOptimizerSpecCode(d: SoapOptimizerNodeData): string {
  const merged = { ...defaultSoapOptimizerData(), ...d };
  const learningRate = firstNumber(merged.learningRate, 0.0003);
  const beta1 = firstNumber(merged.beta1, 0.9);
  const beta2 = firstNumber(merged.beta2, 0.95);
  const epsilon = firstNumber(merged.epsilon, 1e-8);
  const weightDecay = firstNumber(merged.weightDecay, 0);
  const preconditionFrequency = Math.max(1, Math.round(firstNumber(merged.preconditionFrequency, 10)));
  const maxPreconditionerDim = Math.max(1, Math.round(firstNumber(merged.maxPreconditionerDim, 1024)));
  return [
    "def SOAPOptimizer(",
    `    learning_rate: float = ${learningRate},`,
    `    beta1: float = ${beta1},`,
    `    beta2: float = ${beta2},`,
    `    epsilon: float = ${epsilon},`,
    `    weight_decay: float = ${weightDecay},`,
    `    precondition_frequency: int = ${preconditionFrequency},`,
    `    max_preconditioner_dim: int = ${maxPreconditionerDim},`,
    "):",
    '    """Construct a SOAP optimizer config."""',
    "    return {",
    '        "type": "soap",',
    '        "learning_rate": float(learning_rate),',
    '        "beta1": float(beta1),',
    '        "beta2": float(beta2),',
    '        "epsilon": float(epsilon),',
    '        "weight_decay": float(weight_decay),',
    '        "precondition_frequency": int(precondition_frequency),',
    '        "max_preconditioner_dim": int(max_preconditioner_dim),',
    "    }",
  ].join("\n");
}

