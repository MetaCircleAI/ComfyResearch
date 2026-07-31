/** Trainer compute-device UI ids ↔ PyTorch device strings stored on the node. */

export type ComputeModeUi = "local_cpu" | "local_mps" | "local_cuda" | "autodl_gpu";

export type LocalCudaDeviceInfo = {
  index: number;
  name: string;
  totalMemoryMb: number;
};

export const COMPUTE_MODE_OPTIONS: { id: ComputeModeUi; label: string }[] = [
  { id: "local_cpu", label: "CPU" },
  { id: "local_mps", label: "MPS" },
  { id: "local_cuda", label: "CUDA (local)" },
  { id: "autodl_gpu", label: "AutoDL GPU" },
];

const CUDA_INDEX_RE = /^cuda:(\d+)$/;

/** Map stored trainer ``computeDevice`` (+ optional ``remoteGpu``) to UI mode. */
export function computeModeUiFromDevice(spec: unknown, remoteGpu?: boolean): ComputeModeUi {
  const s = String(spec || "cpu").trim().toLowerCase();
  if (s === "mps") return "local_mps";
  if (s === "cuda" || CUDA_INDEX_RE.test(s)) {
    if (remoteGpu === false) return "local_cuda";
    return "autodl_gpu";
  }
  return "local_cpu";
}

/** CUDA device index from stored ``computeDevice`` (``cuda`` → 0). */
export function localCudaIndexFromDevice(spec: unknown): number {
  const s = String(spec || "cuda").trim().toLowerCase();
  if (s === "cuda") return 0;
  const m = CUDA_INDEX_RE.exec(s);
  return m ? Number.parseInt(m[1]!, 10) : 0;
}

export function localCudaDeviceFromIndex(index: number): string {
  const i = Number.isFinite(index) && index >= 0 ? Math.floor(index) : 0;
  return `cuda:${i}`;
}

/** Default local CUDA device when switching to CUDA (local) mode. */
export function defaultLocalCudaDevice(current: unknown): string {
  const s = String(current || "").trim().toLowerCase();
  if (s === "cuda" || CUDA_INDEX_RE.test(s)) return s === "cuda" ? "cuda:0" : s;
  return "cuda:0";
}

/** Map UI mode from DiscreteMultiSelect ``onCommit`` to stored ``computeDevice``. */
export function computeDeviceFromModeUi(mode: ComputeModeUi): string {
  if (mode === "local_mps") return "mps";
  if (mode === "local_cuda") return "cuda:0";
  if (mode === "autodl_gpu") return "cuda";
  return "cpu";
}

/** Whether this UI mode should route training to AutoDL remote SSH. */
export function remoteGpuFromModeUi(mode: ComputeModeUi): boolean {
  return mode === "autodl_gpu";
}

/** Normalize DiscreteMultiSelect value (single id or one-element array) to UI mode. */
export function normalizeComputeModeUi(next: ComputeModeUi | ComputeModeUi[]): ComputeModeUi {
  return (Array.isArray(next) ? next[0] : next) ?? "local_cpu";
}

export function formatLocalCudaDeviceLabel(d: LocalCudaDeviceInfo): string {
  const memGb = d.totalMemoryMb > 0 ? `${(d.totalMemoryMb / 1024).toFixed(1)} GB` : null;
  return memGb ? `GPU ${d.index}: ${d.name} (${memGb})` : `GPU ${d.index}: ${d.name}`;
}

export function localCudaGpuSelectOptions(
  devices: LocalCudaDeviceInfo[],
): { id: string; label: string }[] {
  return devices.map((d) => ({
    id: String(d.index),
    label: formatLocalCudaDeviceLabel(d),
  }));
}
