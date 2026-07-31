export type ModelCheckpointSource = "memory" | "file";

export type ModelCheckpointNodeData = {
  /** Standard base64 of torch.save bytes (model + optimizer state), same as trainer stream. */
  checkpoint_b64: string;
  /** Last checkpoint pushed from a connected Trainer (for “Load from memory” after loading a file). */
  memoryCheckpoint_b64: string;
  /** Whether the active `checkpoint_b64` came from the trainer or from disk. */
  checkpointSource: ModelCheckpointSource;
  /** Original file name when `checkpointSource` is `"file"`. */
  checkpointFileName: string;
};

export function defaultModelCheckpointData(): ModelCheckpointNodeData {
  return {
    checkpoint_b64: "",
    memoryCheckpoint_b64: "",
    checkpointSource: "memory",
    checkpointFileName: "",
  };
}
