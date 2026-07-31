import { describe, expect, it } from "vitest";
import { readNdjsonTrainStream } from "../readNdjsonTrainStream";

describe("readNdjsonTrainStream metrics events", () => {
  it("delivers each metrics snapshot separately from progress events", async () => {
    const encoder = new TextEncoder();
    const chunks = [
      '{"type":"progress","step":1,"total":4}\n{"type":"metrics","step":1,"loss_history":[3],"test_loss_history":[4],"reg_loss_history":[0],"step_ticks":[1],"observable_metric_histories":{}}\n',
      '{"type":"progress","step":2,"total":4}\n{"type":"complete","loss_history":[3],"test_loss_history":[4],"reg_loss_history":[0],"step_ticks":[1],"plot_png_base64":"","visualization_node_ids":[]}\n',
    ];
    const reader = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }).getReader();
    const progress: number[] = [];
    const metrics: number[] = [];

    const result = await readNdjsonTrainStream(
      reader,
      (event) => progress.push(event.step),
      { onMetrics: (event) => metrics.push(event.step) },
    );

    expect(progress).toEqual([1, 2]);
    expect(metrics).toEqual([1]);
    expect(result.complete?.step_ticks).toEqual([1]);
    expect(result.error).toBeNull();
  });
});