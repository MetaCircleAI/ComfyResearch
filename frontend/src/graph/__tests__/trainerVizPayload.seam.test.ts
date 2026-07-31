import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { applyTrainerVizPayload } from "../trainerVizPayload";

describe("local trainer result routing", () => {
  it("updates local loss and observable visualizations omitted from the TrainRequest", () => {
    let nodes = [
      { id: "trainer", type: "trainer", position: { x: 0, y: 0 }, data: {} },
      { id: "obs", type: "observable_weight_l2", position: { x: 0, y: 0 }, data: {} },
      { id: "loss-viz", type: "training_visualization", position: { x: 0, y: 0 }, data: {} },
      {
        id: "obs-viz",
        type: "observable_viz",
        position: { x: 0, y: 0 },
        data: { pairedObservableId: "obs" },
      },
    ] as Node[];
    const edges = [
      {
        id: "obs-trainer",
        source: "obs",
        target: "trainer",
        sourceHandle: "observable",
        targetHandle: "observables",
      },
      {
        id: "trainer-loss-viz",
        source: "trainer",
        target: "loss-viz",
        sourceHandle: "loss_results",
        targetHandle: "tensor_list",
      },
      {
        id: "trainer-obs-viz",
        source: "trainer",
        target: "obs-viz",
        sourceHandle: "observable_results",
        targetHandle: "tensor",
      },
    ] as Edge[];

    applyTrainerVizPayload(
      (update) => {
        nodes = update(nodes);
      },
      {
        loss_history: [3, 2, 1],
        step_ticks: [1, 2, 3],
        epoch_ticks: [0, 0.5, 1],
        plot_png_base64: "plot",
        visualization_node_ids: [],
        observable_viz_updates: [
          {
            node_id: "trainer::__observable_result__obs",
            paired_observable_id: "obs",
            value_history: [4, 5, 6],
          },
        ],
        observable_metric_histories: { obs: [4, 5, 6] },
      },
      "trainer",
      undefined,
      edges,
    );

    expect(nodes.find((node) => node.id === "loss-viz")?.data).toMatchObject({
      lossHistory: [3, 2, 1],
      stepTicks: [1, 2, 3],
      epochTicks: [0, 0.5, 1],
    });
    expect(nodes.find((node) => node.id === "obs-viz")?.data).toMatchObject({
      valueHistory: [4, 5, 6],
      stepTicks: [1, 2, 3],
      epochTicks: [0, 0.5, 1],
    });
  });

  it("routes fallback attention-map frames to the paired canvas visualization", () => {
    let nodes = [
      { id: "trainer", type: "trainer", position: { x: 0, y: 0 }, data: {} },
      { id: "attention", type: "observable_attention_map", position: { x: 0, y: 0 }, data: {} },
      {
        id: "attention-viz",
        type: "observable_viz",
        position: { x: 0, y: 0 },
        data: { pairedObservableId: "attention", vizVariant: "attention_map" },
      },
    ] as Node[];

    applyTrainerVizPayload(
      (update) => {
        nodes = update(nodes);
      },
      {
        loss_history: [],
        step_ticks: [3],
        plot_png_base64: "",
        visualization_node_ids: [],
        observable_viz_updates: [
          {
            node_id: "trainer::__observable_result__attention",
            paired_observable_id: "attention",
            attention_map_frames: [
              {
                step: 3,
                slices: [
                  {
                    layer: 0, batch: 0, head: 0, map: [[1]], token_ids: [7],
                    source_shape: [1, 1], row_start: 0, col_start: 0,
                  },
                ],
              },
            ],
          },
        ],
      },
      "trainer",
    );

    expect(nodes.find((node) => node.id === "attention-viz")?.data).toMatchObject({
      attentionMapFrames: [{ step: 3 }],
      stepTicks: [3],
    });
  });

  it("routes scalar observable histories to RankMe and alpha-ReQ user charts", () => {
    let nodes = [
      { id: "trainer", type: "trainer", position: { x: 0, y: 0 }, data: {} },
      {
        id: "rankme",
        type: "observable_representation_rankme",
        position: { x: 0, y: 0 },
        data: {},
      },
      {
        id: "alpha",
        type: "observable_representation_alpha_req",
        position: { x: 0, y: 0 },
        data: {},
      },
      {
        id: "rankme-viz",
        type: "observable_viz",
        position: { x: 0, y: 0 },
        data: { pairedObservableId: "rankme", vizVariant: "user" },
      },
      {
        id: "alpha-viz",
        type: "observable_viz",
        position: { x: 0, y: 0 },
        data: { pairedObservableId: "alpha", vizVariant: "user" },
      },
    ] as Node[];
    const edges = [
      {
        id: "rankme-trainer",
        source: "rankme",
        target: "trainer",
        sourceHandle: "observable",
        targetHandle: "observables",
      },
      {
        id: "alpha-trainer",
        source: "alpha",
        target: "trainer",
        sourceHandle: "observable",
        targetHandle: "observables",
      },
      {
        id: "trainer-rankme-viz",
        source: "trainer",
        target: "rankme-viz",
        sourceHandle: "observable_results",
        targetHandle: "tensor",
      },
      {
        id: "trainer-alpha-viz",
        source: "trainer",
        target: "alpha-viz",
        sourceHandle: "observable_results",
        targetHandle: "tensor",
      },
    ] as Edge[];

    applyTrainerVizPayload(
      (update) => {
        nodes = update(nodes);
      },
      {
        loss_history: [3, 2, 1],
        step_ticks: [0, 100, 200],
        plot_png_base64: "",
        visualization_node_ids: [],
        observable_metric_histories: {
          rankme: [1.1, 1.8, 2.4],
          alpha: [0.7, 0.9, 1.2],
        },
      },
      "trainer",
      undefined,
      edges,
    );

    expect(nodes.find((node) => node.id === "rankme-viz")?.data).toMatchObject({
      valueHistory: [1.1, 1.8, 2.4],
      stepTicks: [0, 100, 200],
    });
    expect(nodes.find((node) => node.id === "alpha-viz")?.data).toMatchObject({
      valueHistory: [0.7, 0.9, 1.2],
      stepTicks: [0, 100, 200],
    });
  });
});
