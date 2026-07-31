export type BasicCalculatorNodeData = {
  /** Number of target handles `tensor_1` … `tensor_N`. */
  inputCount: number;
  /** LaTeX expression using scalars `x_1`, `x_2`, … (also `x_{10}` style). */
  equationLatex: string;
  outputTensor: { shape: number[]; values: number[] } | null;
  lastError: string | null;
};

export const BASIC_CALCULATOR_INPUT_MAX = 32;

export function clampBasicCalculatorInputCount(v: unknown): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return 2;
  return Math.min(BASIC_CALCULATOR_INPUT_MAX, Math.max(1, Math.floor(n)));
}

export function defaultBasicCalculatorData(partial?: Partial<BasicCalculatorNodeData>): BasicCalculatorNodeData {
  return {
    inputCount: clampBasicCalculatorInputCount(partial?.inputCount),
    equationLatex: typeof partial?.equationLatex === "string" ? partial.equationLatex : String.raw`x_1 + x_2`,
    outputTensor: partial?.outputTensor ?? null,
    lastError: partial?.lastError ?? null,
  };
}
