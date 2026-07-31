export type TrainStreamProgress = { type: "progress"; step: number; total: number };
export type TrainStreamPhase = { type: "phase"; phase: string; message: string };
export type TrainStreamMetrics = {
  type: "metrics";
  step: number;
  loss_history: number[];
  test_loss_history?: number[];
  reg_loss_history?: number[];
  step_ticks: number[];
  epoch_ticks?: number[];
  observable_viz_updates?: {
    node_id: string;
    paired_observable_id?: string;
    value_history?: number[];
    test_value_history?: number[];
    value_histories?: number[][];
    series_labels?: string[];
    embedding_history?: number[][][];
    attention_map_frames?: import("../components/nodes/attentionMapVizDefaults").AttentionMapFrame[];
  }[];
  observable_metric_histories?: Record<string, number[]>;
};
export type TrainStreamComplete = {
  type: "complete";
  checkpoint_b64?: string;
  loss_history: number[];
  test_loss_history?: number[];
  reg_loss_history?: number[];
  step_ticks: number[];
  epoch_ticks?: number[];
  plot_png_base64: string;
  visualization_node_ids: string[];
  observable_viz_updates?: {
    node_id: string;
    paired_observable_id?: string;
    value_history?: number[];
    test_value_history?: number[];
    value_histories?: number[][];
    embedding_history?: number[][][];
    attention_map_frames?: import("../components/nodes/attentionMapVizDefaults").AttentionMapFrame[];
  }[];
  observable_metric_histories?: Record<string, number[]>;
  observable_embedding_histories?: Record<string, number[][][]>;
  observable_attention_slice_histories?: Record<string, import("../components/nodes/attentionMapVizDefaults").AttentionMapFrame[]>;
  train_loop_seconds?: number;
};
type TrainStreamPaused = {
  type: "paused";
  next_step: number;
  checkpoint_b64: string;
  loss_history: number[];
  test_loss_history?: number[];
  step_ticks: number[];
  epoch_ticks?: number[];
  plot_png_base64: string;
  visualization_node_ids: string[];
  observable_viz_updates?: TrainStreamComplete["observable_viz_updates"];
  observable_metric_histories: Record<string, number[]>;
  observable_embedding_histories?: Record<string, number[][][]>;
  observable_attention_slice_histories?: Record<string, import("../components/nodes/attentionMapVizDefaults").AttentionMapFrame[]>;
  train_loop_seconds?: number;
};
type TrainStreamAborted = { type: "aborted" };
type TrainStreamError = { type: "error"; detail: string };

export type ReadNdjsonTrainStreamOptions = {
  onRemoteSession?: () => void;
  onPhase?: (ev: TrainStreamPhase) => void;
  onMetrics?: (ev: TrainStreamMetrics) => void;
};

/**
 * Parse NDJSON body from ``POST /api/train`` until ``complete``, ``paused``, ``aborted``, or ``error``.
 */
export async function readNdjsonTrainStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onProgress: (ev: TrainStreamProgress) => void,
  options?: ReadNdjsonTrainStreamOptions,
): Promise<{
  complete: TrainStreamComplete | null;
  paused: TrainStreamPaused | null;
  aborted: boolean;
  error: string | null;
}> {
  const decoder = new TextDecoder();
  let buffer = "";
  let complete: TrainStreamComplete | null = null;
  let paused: TrainStreamPaused | null = null;
  let aborted = false;
  let error: string | null = null;

  const dispatch = (trimmed: string) => {
    if (!trimmed) return;
    const ev = JSON.parse(trimmed) as
      | TrainStreamProgress
      | TrainStreamPhase
      | TrainStreamMetrics
      | TrainStreamComplete
      | TrainStreamPaused
      | TrainStreamAborted
      | TrainStreamError
      | { type: "remote_session" };
    if (ev.type === "complete") complete = ev;
    else if (ev.type === "paused") paused = ev;
    else if (ev.type === "aborted") aborted = true;
    else if (ev.type === "error") {
      error = typeof ev.detail === "string" ? ev.detail : JSON.stringify(ev.detail);
    } else if (ev.type === "phase") options?.onPhase?.(ev);
    else if (ev.type === "metrics") options?.onMetrics?.(ev);
    else if (ev.type === "remote_session") {
      options?.onRemoteSession?.();
    } else onProgress(ev as TrainStreamProgress);
  };

  while (true) {
    const { done, value } = await reader.read();
    if (value) buffer += decoder.decode(value, { stream: true });
    for (;;) {
      const nl = buffer.indexOf("\n");
      if (nl < 0) break;
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      dispatch(line.trim());
    }
    if (done) break;
  }
  dispatch(buffer.trim());
  return { complete, paused, aborted, error };
}
