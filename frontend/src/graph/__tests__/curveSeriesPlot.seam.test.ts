import { describe, expect, it } from "vitest";

import { appendRowsToTable } from "../curveSeriesParametricAppend";
import { buildCurveOverlayPlotSeries, curveSeriesPlotAxisLabel } from "../curveSeriesPlot";

describe("curve series plot x modes", () => {
  it("param mode plots raw alpha x values; progress normalizes to 0..100", () => {
    const rows = [{
      id: "r1", label: "path", metricId: "test_loss",
      x: [-1, 0.5, 2], y: [0.9, 0.2, 0.8], params: {}, rawSweep: "",
    }] as never;
    const param = buildCurveOverlayPlotSeries(rows, null, "param");
    expect(param[0]!.points.map((p) => p.x)).toEqual([-1, 0.5, 2]);
    const prog = buildCurveOverlayPlotSeries(rows, null, "progress");
    expect(Math.max(...prog[0]!.points.map((p) => p.x))).toBe(100);
  });

  it("epoch mode converts trainer steps with captured run metadata", () => {
    const rows = [{
      id: "sb", label: "SB", metricId: "test_acc",
      x: [0, 500, 1000], y: [0.1, 0.5, 0.8],
      params: { "trainer.trainingEpochs": "100", "trainer.trainingSteps": "1000" },
      paramsNumeric: { "trainer.trainingEpochs": 100, "trainer.trainingSteps": 1000 },
      rawSweep: "",
    }] as never;
    const epoch = buildCurveOverlayPlotSeries(rows, null, "epoch");
    expect(epoch[0]!.points.map((p) => p.x)).toEqual([0, 50, 100]);
  });

  it("epoch mode leaves raw x values when run metadata is unavailable", () => {
    const rows = [{
      id: "legacy", label: "legacy", metricId: "test_acc",
      x: [0, 10], y: [0.1, 0.2], params: {}, rawSweep: "",
    }] as never;
    const epoch = buildCurveOverlayPlotSeries(rows, null, "epoch");
    expect(epoch[0]!.points.map((p) => p.x)).toEqual([0, 10]);
  });

  it("recognizes legacy cyclic rows whose stored x values are already epochs", () => {
    const rows = [{
      id: "legacy-cyclic", label: "CLR", metricId: "test_acc",
      x: [0, 150, 300], y: [0.1, 0.7, 0.8],
      params: { "trainer.trainingEpochs": "300", "trainer.trainingSteps": "105600" },
      paramsNumeric: { "trainer.trainingEpochs": 300, "trainer.trainingSteps": 105600 },
      rawSweep: "model.seed=0",
    }] as never;

    const epoch = buildCurveOverlayPlotSeries(rows, null, "epoch");
    expect(epoch[0]!.points.map((p) => p.x)).toEqual([0, 150, 300]);
    const step = buildCurveOverlayPlotSeries(rows, null, "step");
    expect(step[0]!.points.map((p) => p.x)).toEqual([0, 52800, 105600]);
  });

  it("uses captured exact epoch ticks while keeping optimizer steps on step mode", () => {
    const rows = [{
      id: "new", label: "CBS", metricId: "test_acc",
      x: [0, 36, 44], epochX: [0, 1, 2], y: [0.1, 0.4, 0.5],
      params: { "trainer.trainingEpochs": "2", "trainer.trainingSteps": "44" },
      rawSweep: "model.seed=0",
    }] as never;

    expect(buildCurveOverlayPlotSeries(rows, null, "step")[0]!.points.map((p) => p.x)).toEqual([0, 36, 44]);
    expect(buildCurveOverlayPlotSeries(rows, null, "epoch")[0]!.points.map((p) => p.x)).toEqual([0, 1, 2]);
  });

  it("labels each built-in x mode independently of stale stored plotXKey", () => {
    expect(curveSeriesPlotAxisLabel("step", "epoch")).toBe("step");
    expect(curveSeriesPlotAxisLabel("epoch", "step")).toBe("epoch");
    expect(curveSeriesPlotAxisLabel("progress", "epoch")).toBe("progress %");
    expect(curveSeriesPlotAxisLabel("param", "step")).toBe("α");
  });

  it("averages seeds by trainer run and keeps train/test styling paired", () => {
    const row = (id: string, run: string, metricId: string, y: number[]) => ({
      id,
      label: `${run} ${metricId}`,
      metricId,
      x: [0, 10],
      y,
      params: {
        "trainer.run": run,
        "trainer.trainingEpochs": "10",
        "trainer.trainingSteps": "10",
      },
      rawSweep: "",
    });
    const rows = [
      row("cbs-train-0", "CBS", "train_acc", [0.2, 0.8]),
      row("cbs-train-1", "CBS", "train_acc", [0.4, 1.0]),
      row("cbs-test-0", "CBS", "test_acc", [0.1, 0.7]),
      row("cbs-test-1", "CBS", "test_acc", [0.3, 0.9]),
      row("clr-train-0", "CLR", "train_acc", [0.3, 0.9]),
      row("clr-train-1", "CLR", "train_acc", [0.5, 0.9]),
    ] as never;

    const series = buildCurveOverlayPlotSeries(rows, null, "epoch", true);
    expect(series).toHaveLength(3);
    expect(series[0]!.points[0]!.y).toBeCloseTo(0.3);
    expect(series[0]!.points[1]!.y).toBeCloseTo(0.9);
    expect(series[1]!.points.map((p) => p.y)).toEqual([0.2, 0.8]);
    expect(series[0]!.color).toBe(series[1]!.color);
    expect(series[0]!.strokeDasharray).toBeUndefined();
    expect(series[1]!.strokeDasharray).toBe("4 3");
    expect(series[2]!.color).not.toBe(series[0]!.color);
  });

  it("keeps different trainer runs with the same seed and metric as separate rows", () => {
    const empty = { rows: [], selectedSeriesIds: null, captureMetrics: ["train_acc", "test_acc"], paramKeyOrder: null };
    const curve = { metricId: "test_acc", label: "test acc", x: [0, 1], y: [0.2, 0.8] };
    const cbs = appendRowsToTable(
      empty,
      [curve],
      "model.seed=0",
      { "model.seed": "0", "trainer.run": "CBS", "stream.source": "CBS acc" },
      { "model.seed": 0 },
    );
    const both = appendRowsToTable(
      cbs,
      [curve],
      "model.seed=0",
      { "model.seed": "0", "trainer.run": "CLR", "stream.source": "CLR acc" },
      { "model.seed": 0 },
    );
    expect(both.rows).toHaveLength(2);

    const rerun = appendRowsToTable(
      both,
      [{ ...curve, y: [0.3, 0.9] }],
      "model.seed=0",
      { "model.seed": "0", "trainer.run": "CLR", "stream.source": "CLR acc" },
      { "model.seed": 0 },
    );
    expect(rerun.rows).toHaveLength(2);
    expect(rerun.rows.find((row) => row.params["trainer.run"] === "CLR")!.y).toEqual([0.3, 0.9]);
  });
});
