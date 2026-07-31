import {
  TRAINER_AUTO_OBSERVABLE_VIZ_DY,
  TRAINER_AUTO_TRAINING_VIZ_DY,
  TRAINER_AUTO_VIZ_DX,
} from "./trainerAutoVizSpawn";

/**
 * Matrix placement order for graph-assist batch runs (same seed count as cells used).
 * Order: (0,0) → (0,1),(1,0),(1,1) → (0,2),(1,2),(2,0),(2,1),(2,2) → … expanding the square.
 */
export function graphAssistGridCellOrder(dim: number): Array<[number, number]> {
  if (dim < 1) return [];
  const cells: Array<[number, number]> = [[0, 0]];
  for (let L = 1; L < dim; L++) {
    for (let i = 0; i < L; i++) cells.push([i, L]);
    for (let j = 0; j < L; j++) cells.push([L, j]);
    cells.push([L, L]);
  }
  return cells;
}

/**
 * Keep in sync with ``planRandomTrainerSubgraph`` in ``selfDrivingGraph.ts`` (column/row origins).
 * Cell size is derived so auto-spawned viz nodes (``trainerAutoVizSpawn``) do not collide with the
 * next matrix slot.
 */
const COL2 = 720;
const ROW_MD = 0;
const ROW_TR = 260;
const ROW_OBS = 700;
const ROW_OPT = 420;
const TRAINER_SHELL_W = 420;
const VIZ_SHELL_W = 480;
/** Training / observable viz shells and tall token models often exceed 420px. */
const TALL_BLOCK = 680;

const SUBGRAPH_TOP_DY = ROW_TR + TRAINER_AUTO_TRAINING_VIZ_DY;
const SUBGRAPH_LEFT_DX = 0;
const SUBGRAPH_RIGHT_X = COL2 + TRAINER_SHELL_W + TRAINER_AUTO_VIZ_DX + VIZ_SHELL_W;
const SUBGRAPH_BOTTOM_Y = Math.max(
  ROW_OBS + TALL_BLOCK,
  ROW_TR + TRAINER_AUTO_OBSERVABLE_VIZ_DY + TALL_BLOCK,
  ROW_OPT + TALL_BLOCK,
  /** Token / transformer stacks in column 1 can grow far below the dataset row. */
  ROW_MD + 920,
);

/** Extra slack so stacked matrix rows/columns never clip after fitView / node resize. */
const CELL_GUTTER = 160;

/** Y offset (often negative) from the plan anchor to the top of the tallest auto-spawned viz. */
export const GRAPH_ASSIST_SUBGRAPH_TOP_DY = SUBGRAPH_TOP_DY;

/** Flow-space offset between subgraph anchors (see ``planRandomTrainerSubgraph``). */
export const GRAPH_ASSIST_MATRIX_CELL_W = Math.ceil(SUBGRAPH_RIGHT_X + CELL_GUTTER);
export const GRAPH_ASSIST_MATRIX_CELL_H = Math.ceil(SUBGRAPH_BOTTOM_Y - SUBGRAPH_TOP_DY + CELL_GUTTER);

/** Inset from a matrix cell top-left to place a subgraph centered in that cell. */
export const GRAPH_ASSIST_MATRIX_CELL_INSET_X =
  (GRAPH_ASSIST_MATRIX_CELL_W - (SUBGRAPH_RIGHT_X - SUBGRAPH_LEFT_DX)) / 2 - SUBGRAPH_LEFT_DX;
export const GRAPH_ASSIST_MATRIX_CELL_INSET_Y =
  (GRAPH_ASSIST_MATRIX_CELL_H - (SUBGRAPH_BOTTOM_Y - SUBGRAPH_TOP_DY)) / 2;
