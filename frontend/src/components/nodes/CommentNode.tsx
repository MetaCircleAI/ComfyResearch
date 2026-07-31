import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { defaultCommentData, type CommentNodeData } from "./commentDefaults";

function normalizeHttpUrl(raw: string): string | null {
  const t = raw.trim();
  if (!t) return null;
  let candidate = t;
  if (!/^https?:\/\//i.test(candidate) && !/^[a-z][a-z0-9+.-]*:/i.test(candidate)) {
    candidate = `https://${candidate}`;
  }
  try {
    const u = new URL(candidate);
    if (u.protocol !== "http:" && u.protocol !== "https:") return null;
    return u.href;
  } catch {
    return null;
  }
}

function patchCommentData(
  id: string,
  patch: Partial<CommentNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultCommentData();
      const cur = (n.data ?? {}) as Partial<CommentNodeData>;
      const prev: CommentNodeData = {
        text: cur.text ?? def.text ?? "",
        url: cur.url ?? def.url ?? "",
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function CommentNode({ id, data, selected }: NodeProps) {
  const def = defaultCommentData();
  const raw = (data ?? {}) as Partial<CommentNodeData>;
  const text = raw.text ?? def.text ?? "";
  const url = raw.url ?? def.url ?? "";
  const href = normalizeHttpUrl(url);
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<CommentNodeData>) => patchCommentData(id, patch, setNodes);

  return (
    <div
      className={`cr-node cr-node--comment${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-muted)" }}
    >
      <div className="cr-node__header">Comment</div>
      <div className="cr-node__body cr-node__body--comment">
        <div className="cr-comment-socket-row">
          <div className="cr-comment-socket-row__leftwrap">
            <Handle
              type="target"
              position={Position.Left}
              id="comment"
              className="cr-handle-target cr-handle-target--trainer-row cr-handle-target--comment"
            />
            <span className="cr-trainer-socket-label">comment</span>
          </div>
        </div>
        <textarea
          className="cr-comment-textarea nodrag nopan"
          value={text}
          onChange={(e) => update({ text: e.target.value })}
          placeholder="Notes about this visualization…"
          rows={5}
          spellCheck
        />
        <label className="cr-comment-url-label" htmlFor={`cr-comment-url-${id}`}>
          URL
        </label>
        <input
          id={`cr-comment-url-${id}`}
          className="cr-comment-url-input nodrag nopan"
          type="text"
          value={url}
          onChange={(e) => update({ url: e.target.value })}
          placeholder="https://example.com"
          spellCheck={false}
          autoComplete="off"
        />
        <div className="cr-comment-url-row">
          {href ? (
            <a
              className="cr-comment-url-link nodrag nopan"
              href={href}
              target="_blank"
              rel="noopener noreferrer"
            >
              {href}
            </a>
          ) : (
            <span className="cr-comment-url-placeholder">
              {url.trim() ? "Not a valid http(s) URL yet" : "Clickable URL appears here"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
