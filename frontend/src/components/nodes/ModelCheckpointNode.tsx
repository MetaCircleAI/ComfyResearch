import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useRef, useState, type ChangeEvent } from "react";
import { defaultModelCheckpointData, type ModelCheckpointNodeData } from "./modelCheckpointDefaults";

function patchModelCheckpointData(
  id: string,
  patch: Partial<ModelCheckpointNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultModelCheckpointData();
      const cur = (n.data ?? {}) as Partial<ModelCheckpointNodeData>;
      const prev: ModelCheckpointNodeData = {
        checkpoint_b64: cur.checkpoint_b64 ?? def.checkpoint_b64,
        memoryCheckpoint_b64: cur.memoryCheckpoint_b64 ?? def.memoryCheckpoint_b64,
        checkpointSource: cur.checkpointSource ?? def.checkpointSource,
        checkpointFileName: cur.checkpointFileName ?? def.checkpointFileName,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

function base64DecodedByteLength(b64: string): number {
  const pad = b64.endsWith("==") ? 2 : b64.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((b64.length * 3) / 4) - pad);
}

function uint8ToStandardBase64(u8: Uint8Array): string {
  const chunk = 0x8000;
  let binary = "";
  for (let i = 0; i < u8.length; i += chunk) {
    binary += String.fromCharCode.apply(null, u8.subarray(i, i + chunk) as unknown as number[]);
  }
  return btoa(binary);
}

function triggerDownloadBytes(bytes: Uint8Array, filename: string) {
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** File System Access API — not in all DOM lib versions; optional on `window`. */
type WindowWithSaveFilePicker = Window & {
  showSaveFilePicker?: (options?: {
    suggestedName?: string;
    types?: { description: string; accept: Record<string, string[]> }[];
  }) => Promise<FileSystemFileHandle>;
};

function checkpointB64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/** Opens the system “Save as” dialog when supported (Chromium); otherwise downloads with a default name. */
async function saveCheckpointBytes(bytes: Uint8Array, suggestedName: string): Promise<void> {
  const win = window as WindowWithSaveFilePicker;
  if (typeof win.showSaveFilePicker !== "function") {
    triggerDownloadBytes(bytes, suggestedName);
    return;
  }
  try {
    const handle = await win.showSaveFilePicker({
      suggestedName,
      types: [
        {
          description: "PyTorch checkpoint",
          accept: { "application/octet-stream": [".pt", ".pth"] },
        },
      ],
    });
    const writable = await handle.createWritable();
    await writable.write(bytes);
    await writable.close();
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      return;
    }
    throw e;
  }
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Passthrough checkpoint slot: Trainer `checkpoint` → `model_checkpoint` → downstream `model`; optional file I/O. */
export function ModelCheckpointNode({ id, data, selected }: NodeProps) {
  const def = defaultModelCheckpointData();
  const raw = (data ?? {}) as Partial<ModelCheckpointNodeData>;
  const d: ModelCheckpointNodeData = {
    checkpoint_b64: raw.checkpoint_b64 ?? def.checkpoint_b64,
    memoryCheckpoint_b64: raw.memoryCheckpoint_b64 ?? def.memoryCheckpoint_b64,
    checkpointSource: raw.checkpointSource ?? def.checkpointSource,
    checkpointFileName: raw.checkpointFileName ?? def.checkpointFileName,
  };
  const { setNodes } = useReactFlow();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileErr, setFileErr] = useState<string | null>(null);

  const update = useCallback(
    (patch: Partial<ModelCheckpointNodeData>) => patchModelCheckpointData(id, patch, setNodes),
    [id, setNodes],
  );

  const hasCkpt = d.checkpoint_b64.length > 0;
  /** Trainer-side snapshot: explicit field, or legacy graphs where only `checkpoint_b64` was set from training. */
  const trainerCheckpointB64 =
    d.memoryCheckpoint_b64.length > 0
      ? d.memoryCheckpoint_b64
      : d.checkpointSource === "memory"
        ? d.checkpoint_b64
        : "";
  const hasTrainerCheckpoint = trainerCheckpointB64.length > 0;
  const approxBytes = hasCkpt ? base64DecodedByteLength(d.checkpoint_b64) : 0;

  const onLoadFromMemory = useCallback(() => {
    setFileErr(null);
    if (!trainerCheckpointB64) {
      setFileErr("No trainer checkpoint in memory yet.");
      return;
    }
    update({
      checkpoint_b64: trainerCheckpointB64,
      checkpointSource: "memory",
      checkpointFileName: "",
    });
  }, [trainerCheckpointB64, update]);

  const onSaveFile = useCallback(() => {
    setFileErr(null);
    if (!d.checkpoint_b64) {
      setFileErr("No checkpoint in this node yet.");
      return;
    }
    void (async () => {
      try {
        const bytes = checkpointB64ToBytes(d.checkpoint_b64);
        await saveCheckpointBytes(bytes, "model_checkpoint.pt");
      } catch (e) {
        setFileErr(e instanceof Error ? e.message : "Could not save checkpoint.");
      }
    })();
  }, [d.checkpoint_b64]);

  const onPickFile = useCallback(() => {
    setFileErr(null);
    fileInputRef.current?.click();
  }, []);

  const onFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setFileErr(null);
      const f = e.target.files?.[0];
      e.target.value = "";
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const result = reader.result;
          if (!(result instanceof ArrayBuffer)) {
            setFileErr("Could not read file.");
            return;
          }
          const u8 = new Uint8Array(result);
          const b64 = uint8ToStandardBase64(u8);
          update({
            checkpoint_b64: b64,
            checkpointSource: "file",
            checkpointFileName: f.name,
          });
        } catch (err) {
          setFileErr(err instanceof Error ? err.message : "Could not load file.");
        }
      };
      reader.onerror = () => setFileErr("File read failed.");
      reader.readAsArrayBuffer(f);
    },
    [update],
  );

  return (
    <div
      className={`cr-node cr-node--model-checkpoint${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-checkpoint)" }}
    >
      <div className="cr-node__header">Model Checkpoint</div>
      <div className="cr-node__body cr-node__body--compact">
        <div className="cr-trainer-io" aria-label="Checkpoint in and model out">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="model_checkpoint"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--model_checkpoint"
              />
              <span className="cr-trainer-socket-label">model checkpoint</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">model</span>
              <Handle
                type="source"
                position={Position.Right}
                id="model"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--model"
              />
            </div>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          style={{ display: "none" }}
          accept=".pt,.pth,.bin,application/octet-stream"
          aria-hidden
          tabIndex={-1}
          onChange={onFileChange}
        />
        <div className="cr-model-ckpt__files nodrag nopan">
          <button type="button" className="cr-btn" disabled={!hasCkpt} onClick={onSaveFile}>
            Save to file
          </button>
          <button type="button" className="cr-btn" onClick={onPickFile}>
            Load from file
          </button>
          <button type="button" className="cr-btn" disabled={!hasTrainerCheckpoint} onClick={onLoadFromMemory}>
            Load from memory
          </button>
        </div>
        <p className="cr-model-ckpt__status" aria-live="polite">
          {!hasCkpt
            ? "No checkpoint yet (train or load a file)."
            : d.checkpointSource === "file"
              ? `using checkpoint from file ${d.checkpointFileName || "unknown"}.`
              : `using checkpoint in memory (~${formatBytes(approxBytes)}).`}
        </p>
        {fileErr ? <p className="cr-model-ckpt__err">{fileErr}</p> : null}
      </div>
    </div>
  );
}
