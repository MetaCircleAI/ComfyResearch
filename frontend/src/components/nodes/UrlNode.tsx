import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { defaultUrlNodeData, type UrlNodeData } from "./urlNodeDefaults";

function patchUrlNodeData(
  id: string,
  patch: Partial<UrlNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultUrlNodeData();
      const cur = (n.data ?? {}) as Partial<UrlNodeData>;
      const prev: UrlNodeData = {
        url: cur.url ?? def.url ?? "",
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

/** Returns a safe http(s) href, or null if the string cannot be parsed as such. */
export function normalizeHttpUrl(raw: string): string | null {
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

export function UrlNode({ id, data, selected }: NodeProps) {
  const def = defaultUrlNodeData();
  const raw = (data ?? {}) as Partial<UrlNodeData>;
  const url = raw.url ?? def.url ?? "";
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<UrlNodeData>) => patchUrlNodeData(id, patch, setNodes);
  const href = normalizeHttpUrl(url);

  return (
    <div
      className={`cr-node cr-node--url_node${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-muted)" }}
    >
      <div className="cr-node__header">{readInstanceTitle(data, "URL")}</div>
      <div className="cr-node__body cr-node__body--url-node">
        <div className="cr-trainer-io cr-trainer-io--url-node-spacer" aria-hidden />
        <label className="cr-url-node__label" htmlFor={`cr-url-node-url-${id}`}>
          Address
        </label>
        <input
          id={`cr-url-node-url-${id}`}
          className="cr-url-node__input nodrag nopan"
          type="text"
          value={url}
          onChange={(e) => update({ url: e.target.value })}
          placeholder="https://example.com"
          spellCheck={false}
          autoComplete="off"
        />
        <div className="cr-url-node__link-row">
          {href ? (
            <a
              className="cr-url-node__link nodrag nopan"
              href={href}
              target="_blank"
              rel="noopener noreferrer"
            >
              {href}
            </a>
          ) : (
            <span className="cr-url-node__link-placeholder">
              {url.trim() ? "Not a valid http(s) link yet" : "Link appears here when valid"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
