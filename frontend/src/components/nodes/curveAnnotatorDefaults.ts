import type { DiscreteMultiSelectOptionGroup } from "./DiscreteMultiSelect";

/** Fixed labels grouped for the region-label picker (must match stored region labels). */
export const CURVE_ANNOTATOR_LABEL_GROUPS = {
  local: [
    "spike",
    "oscillation",
    "noise burst",
    "phase transition",
    "turning point",
  ],
  regional: [
    "plateau",
    "slight plateau (slowdown)",
    "exponential decay",
    "power law",
  ],
  global: [
    "decreasing",
    "increasing",
    "U shape",
    "inverted U",
    "double descent",
    "double ascent",
    "complex",
  ],
} as const;

export type CurveAnnotatorLabel =
  | (typeof CURVE_ANNOTATOR_LABEL_GROUPS.local)[number]
  | (typeof CURVE_ANNOTATOR_LABEL_GROUPS.regional)[number]
  | (typeof CURVE_ANNOTATOR_LABEL_GROUPS.global)[number];

/** `DiscreteMultiSelect` sections for curve annotator region labels. */
export const CURVE_ANNOTATOR_LABEL_OPTION_GROUPS: DiscreteMultiSelectOptionGroup<CurveAnnotatorLabel>[] = [
  {
    title: "Local",
    options: CURVE_ANNOTATOR_LABEL_GROUPS.local.map((id) => ({ id, label: id })),
  },
  {
    title: "Regional",
    options: CURVE_ANNOTATOR_LABEL_GROUPS.regional.map((id) => ({ id, label: id })),
  },
  {
    title: "Global",
    options: CURVE_ANNOTATOR_LABEL_GROUPS.global.map((id) => ({ id, label: id })),
  },
];

export type CurveAnnotatorRegion = {
  id: string;
  /** Raw step axis (training step / tick), inclusive bounds in data space. */
  stepMin: number;
  stepMax: number;
  label: string;
};

export type CurveAnnotatorNodeData = {
  regions?: CurveAnnotatorRegion[];
};

export function defaultCurveAnnotatorData(): CurveAnnotatorNodeData {
  return { regions: [] };
}

export function isCurveAnnotatorLabel(s: string): s is CurveAnnotatorLabel {
  const all = [
    ...CURVE_ANNOTATOR_LABEL_GROUPS.local,
    ...CURVE_ANNOTATOR_LABEL_GROUPS.regional,
    ...CURVE_ANNOTATOR_LABEL_GROUPS.global,
  ];
  return (all as readonly string[]).includes(s);
}
