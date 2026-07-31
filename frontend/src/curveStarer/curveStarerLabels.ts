import type { Node } from "@xyflow/react";
import { getObservableNodeDisplayName } from "../components/nodes/observableVizTitle";
import { nodeDisplayLabel } from "./observableCurvePayload";

const GENERIC_OBSERVABLE_VIZ_TITLE = /^Observable viz(?: \d+)?$/i;
const GENERIC_TRAINING_VIZ_TITLE = /^Training viz(?: \d+)?$/i;
/** Payload / viz legend ids that are not layer-specific series names. */
const GENERIC_SERIES_LABEL = /^(train|test|global|series\s+\d+)$/i;

function isGenericSeriesLabel(series: string): boolean {
  return !series.trim() || GENERIC_SERIES_LABEL.test(series.trim());
}

/** Longest-prefix match so ``l2_norm_dim0`` is not parsed as op ``l2``. */
const REDUCTION_OPS_ORDERED = [
  "l2_norm",
  "l1_norm",
  "mean",
  "median",
  "max",
  "min",
  "std",
  "entropy",
] as const;

function reductionOpFromLabelPart(part: string): string | null {
  for (const op of REDUCTION_OPS_ORDERED) {
    if (part === op || part.startsWith(`${op}_`)) return op;
  }
  return null;
}

function splitAlgebraLabel(label: string): { tensorPart: string; reductionParts: string[] } {
  const parts = label.trim().split(".");
  if (parts.length < 2) return { tensorPart: label.trim(), reductionParts: [] };
  let firstReductionIdx = -1;
  for (let i = 1; i < parts.length; i++) {
    if (reductionOpFromLabelPart(parts[i]!)) {
      firstReductionIdx = i;
      break;
    }
  }
  if (firstReductionIdx < 0) return { tensorPart: label.trim(), reductionParts: [] };
  return {
    tensorPart: parts.slice(0, firstReductionIdx).join("."),
    reductionParts: parts.slice(firstReductionIdx),
  };
}

function formatReductionChain(reductionParts: string[]): string {
  return reductionParts
    .map((part) => {
      const sep = part.lastIndexOf("_");
      if (sep <= 0) return part;
      return `${part.slice(0, sep)}(${part.slice(sep + 1)})`;
    })
    .join(" → ");
}

/** Human-readable CurveStarer title: tensor + reduction chain, not generic “Observable viz N”. */
export function buildCurveStarerEntryLabel(
  observableLabel: string,
  seriesLabel: string,
  multiSeries: boolean,
): string {
  const obs = observableLabel.trim();
  const series = seriesLabel.trim();
  const { tensorPart, reductionParts } = splitAlgebraLabel(obs);
  const reductionText = reductionParts.length ? formatReductionChain(reductionParts) : "";

  const titleFromObservable = (): string => {
    if (reductionText) {
      const tensor =
        tensorPart.includes("*") && series && !isGenericSeriesLabel(series) ? series : tensorPart;
      const displayTensor = tensor.replace(/^\*\./, "");
      return `${displayTensor} · ${reductionText}`;
    }
    if (obs && !GENERIC_OBSERVABLE_VIZ_TITLE.test(obs)) return obs;
    return "";
  };

  if (multiSeries && series) {
    const base = titleFromObservable();
    // ``valueHistory`` + mirrored ``valueHistories`` both surface as train/global — keep observable name.
    if (isGenericSeriesLabel(series)) {
      if (base) {
        if (/^(train|test)$/i.test(series)) return `${base} · ${series}`;
        return base;
      }
      return series;
    }
    if (reductionText) return `${series} · ${reductionText}`;
    if (tensorPart.includes("*") || obs.includes("*")) return series;
    if (base) return `${base} · ${series}`;
    if (obs && !GENERIC_OBSERVABLE_VIZ_TITLE.test(obs)) return `${obs} · ${series}`;
    return series;
  }

  const single = titleFromObservable();
  if (single) return single;

  if (obs && !GENERIC_OBSERVABLE_VIZ_TITLE.test(obs)) return obs;
  return series || obs || "Observable";
}

function isTrainTestSeries(curve: { id: string; label: string }): "train" | "test" | null {
  const series = curve.label.trim().toLowerCase();
  const seriesId = curve.id.trim().toLowerCase();
  if (series === "test" || seriesId === "test") return "test";
  if (series === "train" || seriesId === "train") return "train";
  return null;
}

/** CurveStarer cell title for one plottable series (training loss, accuracy, algebra observables, …). */
export function buildCurveStarerEntryLabelForNode(
  node: Node,
  curve: { id: string; label: string },
  sourceLabel: string,
  multiSeries: boolean,
): string {
  const nodeType = String(node.type ?? "");
  const split = isTrainTestSeries(curve);

  if (nodeType === "training_visualization" && split) {
    return split === "test" ? "test loss" : "train loss";
  }

  const data = (node.data ?? {}) as { vizVariant?: string };
  const isAccuracyViz =
    nodeType === "observable_viz" &&
    (data.vizVariant === "accuracy" || sourceLabel.trim().toLowerCase() === "accuracy");
  if (isAccuracyViz && split) {
    return `${split} accuracy`;
  }

  return buildCurveStarerEntryLabel(sourceLabel, curve.label, multiSeries);
}

function familySuffixFromTensorName(tensorName: string): string | null {
  const trimmed = tensorName.trim();
  const bodyMatch = trimmed.match(/^(?:body\.)?\d+\.(.+)$/);
  if (bodyMatch?.[1]) return bodyMatch[1];
  const idxMatch = trimmed.match(/^\d+\.(.+)$/);
  if (idxMatch?.[1]) return idxMatch[1];
  return null;
}

function familyPatternFromTensorName(tensorName: string): string {
  const suf = familySuffixFromTensorName(tensorName);
  if (suf) return `*.${suf}`;
  return tensorName.trim() || "tensor";
}

/** Replace ``*.weight…`` (or ``h*\_…``) with a concrete member tensor name. */
export function expandWildcardMemberLabel(wildcardLabel: string, memberStorageKey: string): string {
  const memberTensor = memberStorageKey.replace(/_/g, ".");
  const splitIdx = wildcardLabel.indexOf(" · ");
  const base = splitIdx >= 0 ? wildcardLabel.slice(0, splitIdx) : wildcardLabel;
  const suffix = splitIdx >= 0 ? wildcardLabel.slice(splitIdx) : "";

  if (!base.includes("*")) return wildcardLabel;

  const familyPat = familyPatternFromTensorName(memberTensor);
  if (base.startsWith(familyPat)) {
    return `${memberTensor}${base.slice(familyPat.length)}${suffix}`;
  }

  const starDot = base.match(/^(\*\.)([^.]+)(.*)$/);
  if (starDot) {
    const kind = starDot[2]!;
    if (base.startsWith(`*.${kind}`)) {
      return `${memberTensor}${base.slice(2 + kind.length)}${suffix}`;
    }
  }

  return `${memberTensor}${suffix}`;
}

export function labelContainsWildcard(label: string): boolean {
  return label.includes("*");
}

export function resolveCurveStarerObservableLabel(node: Node, nodesById: Map<string, Node>): string {
  const nodeType = String(node.type ?? "");
  const data = (node.data ?? {}) as Record<string, unknown>;

  if (nodeType === "training_visualization") {
    const title = typeof data.instanceTitle === "string" ? data.instanceTitle.trim() : "";
    if (title && !GENERIC_TRAINING_VIZ_TITLE.test(title)) return title;
    return "Training viz";
  }

  const observableName = typeof data.observableName === "string" ? data.observableName.trim() : "";
  if (observableName) return observableName;

  const pairedId = typeof data.pairedObservableId === "string" ? data.pairedObservableId.trim() : "";
  if (pairedId) {
    const paired = nodesById.get(pairedId);
    if (paired) return getObservableNodeDisplayName(paired);
  }

  const instanceTitle = typeof data.instanceTitle === "string" ? data.instanceTitle.trim() : "";
  if (instanceTitle && !GENERIC_OBSERVABLE_VIZ_TITLE.test(instanceTitle)) return instanceTitle;

  return nodeDisplayLabel(node);
}
