import type { Node } from "@xyflow/react";
import { useEffect, useState } from "react";
import { nodeRegistryHint, nodeRegistryTitle } from "../graph/nodeRegistrySpec";
import { InfoRichText } from "./nodes/InfoRichText";


function nodeLabel(node: Node): string {
  const data = (node.data ?? {}) as Record<string, unknown>;
  const customTitle = [data.instanceTitle, data.title, data.label, data.name].find(
    (value): value is string => typeof value === "string" && value.trim().length > 0,
  );
  return customTitle ?? nodeRegistryTitle(node.type ?? "") ?? node.type ?? "Untitled node";
}

function formatValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "string") return value || "-";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    const text = JSON.stringify(value, null, 2);
    return text.length > 900 ? `${text.slice(0, 900)}...` : text;
  } catch {
    return String(value);
  }
}

function ParameterEditor({
  name,
  value,
  onCommit,
}: {
  name: string;
  value: unknown;
  onCommit: (value: unknown) => void;
}) {
  const isStructured = typeof value === "object" && value != null;
  const [draft, setDraft] = useState(() => formatValue(value));

  useEffect(() => {
    setDraft(formatValue(value));
  }, [value]);

  if (typeof value === "boolean") {
    return (
      <input
        type="checkbox"
        aria-label={name}
        checked={value}
        onChange={(event) => onCommit(event.target.checked)}
      />
    );
  }

  if (typeof value === "number") {
    return (
      <input
        type="text"
        inputMode="decimal"
        aria-label={name}
        value={Number.isFinite(value) ? value : ""}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) onCommit(next);
        }}
      />
    );
  }

  if (isStructured) {
    return (
      <textarea
        aria-label={name}
        value={draft}
        spellCheck={false}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => {
          try {
            onCommit(JSON.parse(draft));
          } catch {
            setDraft(formatValue(value));
          }
        }}
      />
    );
  }

  return <input type="text" aria-label={name} value={draft} onChange={(event) => { setDraft(event.target.value); onCommit(event.target.value); }} />;
}

export function NodeInformationPanel({
  node,
  informationTitle,
  informationText,
  code,
  initialMode,
  onUpdateData,
}: {
  node: Node;
  informationTitle?: string;
  informationText?: string;
  code?: string;
  initialMode?: "parameters" | "code";
  onUpdateData: (key: string, value: unknown) => void;
}) {
  const dataEntries = Object.entries((node.data ?? {}) as Record<string, unknown>);
  const [mode, setMode] = useState<"parameters" | "code">(initialMode ?? "parameters");
  const [nearScrollbar, setNearScrollbar] = useState(false);

  useEffect(() => {
    setMode(initialMode ?? "parameters");
  }, [initialMode, node.id]);

  return (
    <aside
      className={`cr-node-information-panel nodrag nopan${nearScrollbar ? " is-scrollbar-near" : ""}`}
      aria-label="Node information"
      onMouseMove={(event) => {
        const { right } = event.currentTarget.getBoundingClientRect();
        const next = right - event.clientX <= 18;
        if (next !== nearScrollbar) setNearScrollbar(next);
      }}
      onMouseLeave={() => setNearScrollbar(false)}
    >
      <header className="cr-node-information-panel__header">
        <div>
          <h2>Node information</h2>
          <p>{nodeLabel(node)}</p>
        </div>
        <div className="cr-node-information-panel__modes" role="tablist" aria-label="Node information mode">
          <button type="button" role="tab" aria-selected={mode === "parameters"} className={`cr-node-information-panel__mode${mode === "parameters" ? " is-active" : ""}`} onClick={() => setMode("parameters")}>Parameters</button>
          <button type="button" role="tab" aria-selected={mode === "code"} className={`cr-node-information-panel__mode${mode === "code" ? " is-active" : ""}`} onClick={() => setMode("code")}>Code</button>
        </div>
      </header>
      <div className="cr-node-information-panel__scroll">
        <div className={`cr-node-information-panel__track${mode === "code" ? " is-code" : ""}`}>
          <section className="cr-node-information-panel__details cr-node-information-panel__page">
              <h3>{nodeLabel(node)}</h3>
              {informationText ? (
                <div className="cr-node-information-panel__about">
                  <h4>{informationTitle ?? "About"}</h4>
                  <p><InfoRichText text={informationText} /></p>
                </div>
              ) : nodeRegistryHint(node.type ?? "") ? (
                <p className="cr-node-information-panel__hint"><InfoRichText text={nodeRegistryHint(node.type ?? "")} /></p>
              ) : null}

              <dl className="cr-node-information-panel__meta">
                <div><dt>Type</dt><dd>{node.type ?? "unknown"}</dd></div>
              </dl>

              <h4>Parameters</h4>
              {dataEntries.length === 0 ? <p className="cr-node-information-panel__empty">No parameters.</p> : null}
              {dataEntries.map(([key, value]) => (
                <div key={key} className="cr-node-information-panel__field">
                  <span>{key}</span>
                  <ParameterEditor name={key} value={value} onCommit={(next) => onUpdateData(key, next)} />
                </div>
              ))}

          </section>
          <section className="cr-node-information-panel__details cr-node-information-panel__page">
            <h3>Code</h3>
            {code ? <pre className="cr-node-information-panel__code-view">{code}</pre> : <p className="cr-node-information-panel__empty">No code is available for this node.</p>}
          </section>
        </div>
      </div>
    </aside>
  );
}
