import {
  Fragment,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useState,
  type ChangeEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import katex from "katex";
import {
  DATASET_NODE_INFO_MARKDOWN,
  type DatasetNodeInfoKind,
  datasetNodeInfoTitle,
} from "./datasetNodeInfoContent";
import { parseLatexDelimited } from "../../utils/katexDelimited";
import { renderNodeInfoTextWithLinks } from "./nodeInfoInlineLinks";
import { openNodeInformation } from "../nodeInformationEvents";

type DatasetInfoEditorTab = "write" | "preview";
/** Bump when bundled defaults should replace prior localStorage (same key would keep stale user-saved text). */
const DATASET_INFO_STORAGE_VERSION = 3;
const DATASET_INFO_STORAGE_PREFIX = "cr.datasetInfoMarkdown.";
function datasetInfoStorageKey(nodeType: DatasetNodeInfoKind): string {
  return `${DATASET_INFO_STORAGE_PREFIX}v${DATASET_INFO_STORAGE_VERSION}.${nodeType}`;
}

function DatasetInfoRichParagraph({ markdown }: { markdown: string }) {
  const segments = useMemo(() => parseLatexDelimited(markdown.trim()), [markdown]);
  return (
    <p className="cr-dataset-info-modal__p">
      {segments.map((seg, idx) => {
        if (seg.kind === "text") {
          return <span key={idx}>{renderNodeInfoTextWithLinks(seg.text, `p-${idx}`)}</span>;
        }
        const t = seg.latex;
        if (!t) return null;
        let html = "";
        try {
          html = katex.renderToString(t, { throwOnError: false, displayMode: seg.display });
        } catch {
          html = "";
        }
        if (!html) {
          return (
            <code key={idx} className="cr-dataset-info-modal__math-fallback">
              {t}
            </code>
          );
        }
        if (seg.display) {
          return (
            <span
              key={idx}
              className="cr-dataset-info-modal__math cr-dataset-info-modal__math--display nodrag nopan"
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        }
        return (
          <span
            key={idx}
            className="cr-dataset-info-modal__math cr-dataset-info-modal__math--inline nodrag nopan"
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      })}
    </p>
  );
}

function DatasetInfoRichListItem({ markdown }: { markdown: string }) {
  const segments = useMemo(() => parseLatexDelimited(markdown.trim()), [markdown]);
  return (
    <>
      {segments.map((seg, idx) => {
        if (seg.kind === "text") {
          return <span key={idx}>{renderNodeInfoTextWithLinks(seg.text, `li-${idx}`)}</span>;
        }
        const t = seg.latex;
        if (!t) return null;
        let html = "";
        try {
          html = katex.renderToString(t, { throwOnError: false, displayMode: false });
        } catch {
          html = "";
        }
        if (!html) {
          return (
            <code key={idx} className="cr-dataset-info-modal__math-fallback">
              {t}
            </code>
          );
        }
        return (
          <span
            key={idx}
            className="cr-dataset-info-modal__math cr-dataset-info-modal__math--inline nodrag nopan"
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      })}
    </>
  );
}

function isDatasetInfoBulletLine(line: string): boolean {
  return /^\*\s+/.test(line) || /^-\s+/.test(line);
}

function stripDatasetInfoBulletPrefix(line: string): string {
  const m = /^\*\s+|^-\s+/.exec(line);
  return m ? line.slice(m[0].length) : line;
}

/** One double-newline-separated block may mix prose and lists; split into runs. */
function parseDatasetInfoBlockRuns(lines: string[]): Array<
  { kind: "p"; text: string } | { kind: "ul"; items: string[] }
> {
  const runs: Array<{ kind: "p"; text: string } | { kind: "ul"; items: string[] }> = [];
  let idx = 0;
  while (idx < lines.length) {
    if (isDatasetInfoBulletLine(lines[idx])) {
      const items: string[] = [];
      while (idx < lines.length && isDatasetInfoBulletLine(lines[idx])) {
        items.push(stripDatasetInfoBulletPrefix(lines[idx]));
        idx += 1;
      }
      runs.push({ kind: "ul", items });
    } else {
      const paraLines: string[] = [];
      while (idx < lines.length && !isDatasetInfoBulletLine(lines[idx])) {
        paraLines.push(lines[idx]);
        idx += 1;
      }
      runs.push({ kind: "p", text: paraLines.join("\n") });
    }
  }
  return runs;
}

function DatasetInfoModalBody({ markdown }: { markdown: string }) {
  const blocks = useMemo(
    () =>
      markdown
        .trim()
        .split(/\n\n+/)
        .map((b) => b.trim())
        .filter(Boolean),
    [markdown],
  );
  return (
    <div className="cr-dataset-info-modal__body">
      {blocks.map((block, i) => {
        const lines = block
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean);
        const runs = parseDatasetInfoBlockRuns(lines);
        return (
          <Fragment key={i}>
            {runs.map((run, ri) =>
              run.kind === "ul" ? (
                <ul key={`${i}-${ri}`} className="cr-dataset-info-modal__ul">
                  {run.items.map((line, li) => (
                    <li key={li} className="cr-dataset-info-modal__li">
                      <DatasetInfoRichListItem markdown={line} />
                    </li>
                  ))}
                </ul>
              ) : (
                <DatasetInfoRichParagraph key={`${i}-${ri}`} markdown={run.text} />
              ),
            )}
          </Fragment>
        );
      })}
    </div>
  );
}

function DatasetNodeInfoModal({
  open,
  title,
  nodeType,
  defaultMarkdown,
  onClose,
}: {
  open: boolean;
  title: string;
  nodeType: DatasetNodeInfoKind;
  defaultMarkdown: string;
  onClose: () => void;
}) {
  const titleId = useId();
  const [activeTab, setActiveTab] = useState<DatasetInfoEditorTab>("preview");
  const [draftMarkdown, setDraftMarkdown] = useState(defaultMarkdown);

  useEffect(() => {
    if (!open) return;
    setActiveTab("preview");
    if (typeof window === "undefined") {
      setDraftMarkdown(defaultMarkdown);
      return;
    }
    const stored = window.localStorage.getItem(datasetInfoStorageKey(nodeType));
    setDraftMarkdown(stored ?? defaultMarkdown);
  }, [defaultMarkdown, nodeType, open]);

  const onBackdropMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose],
  );

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const onChangeDraft = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    setDraftMarkdown(e.target.value);
  }, []);

  const onSave = useCallback(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(datasetInfoStorageKey(nodeType), draftMarkdown);
    }
    onClose();
  }, [draftMarkdown, nodeType, onClose]);

  const onCancel = useCallback(() => {
    onClose();
  }, [onClose]);

  if (!open) return null;

  const node = (
    <div
      className="cr-modal-backdrop cr-dataset-info-modal-backdrop"
      style={{ zIndex: 10040 }}
      role="presentation"
      onMouseDown={onBackdropMouseDown}
    >
      <div
        className="cr-modal cr-dataset-info-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id={titleId} className="cr-modal__title">
          {title}
        </h2>
        <div className="cr-dataset-info-modal__tabs" role="tablist" aria-label="Description editor mode">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "write"}
            className={`cr-dataset-info-modal__tab${activeTab === "write" ? " is-active" : ""}`}
            onClick={() => setActiveTab("write")}
          >
            Write
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "preview"}
            className={`cr-dataset-info-modal__tab${activeTab === "preview" ? " is-active" : ""}`}
            onClick={() => setActiveTab("preview")}
          >
            Preview
          </button>
        </div>

        {activeTab === "write" ? (
          <div className="cr-dataset-info-modal__editor-wrap">
            <textarea
              className="cr-dataset-info-modal__editor"
              value={draftMarkdown}
              onChange={onChangeDraft}
              spellCheck={false}
              aria-label={`Edit ${title} description`}
            />
            <div className="cr-dataset-info-modal__editor-hint">TeX is supported</div>
          </div>
        ) : (
          <DatasetInfoModalBody markdown={draftMarkdown} />
        )}
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn" onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={onSave}>
            Save
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}

export function DatasetNodeHeaderWithInfo({
  nodeType,
  nodeId,
  specPythonCode,
  fallbackTitle,
  fallbackMarkdown,
  children,
}: {
  nodeType: DatasetNodeInfoKind;
  /** React Flow node id (required when ``specPythonCode`` is provided for the code icon). */
  nodeId?: string;
  /** Flow ``type`` string for the code notebook gutter (e.g. ``linear_dataset``). */
  graphNodeType?: string;
  /** Generated Python spec for this node — shown in Code mode when the user clicks the code icon. */
  specPythonCode?: string;
  /** NodeDef-channel fallback:datasetNodeInfoContent 无该 type
   * 条目时用 generated spec 的 label/hint——保住"新增 dataset = 1 py"路径,不逼
   * 每个 Python-only dataset 再补一张 TS info 表。 */
  fallbackTitle?: string;
  fallbackMarkdown?: string;
  children: ReactNode;
}) {
  const defaultMarkdown = DATASET_NODE_INFO_MARKDOWN[nodeType] ?? fallbackMarkdown ?? "";
  const title = (DATASET_NODE_INFO_MARKDOWN[nodeType] != null ? datasetNodeInfoTitle(nodeType) : undefined) ?? fallbackTitle ?? String(nodeType);
  const onOpen = useCallback((e: ReactMouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (!nodeId) return;
    openNodeInformation({ nodeId, title, text: defaultMarkdown, code: specPythonCode, mode: "parameters" });
  }, [defaultMarkdown, nodeId, specPythonCode, title]);

  const onPointerDownInfo = useCallback((e: ReactPointerEvent<HTMLButtonElement>) => {
    e.stopPropagation();
  }, []);

  return (
    <>
      <div className="cr-node__header cr-node__header--dataset-info-row">
        <div className="cr-node__header-main">{children}</div>
        <div className="cr-dataset-node-header-actions">
          <button
            type="button"
            className="cr-dataset-node-info-btn nodrag nopan"
            aria-label={`About ${title}`}
            title="Dataset information"
            onClick={onOpen}
            onPointerDown={onPointerDownInfo}
          >
            <svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path
                d="M10 18.5a8.5 8.5 0 1 0 0-17 8.5 8.5 0 0 0 0 17Z"
                stroke="currentColor"
                strokeWidth="1.4"
              />
              <path d="M10 9.2V14" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              <circle cx="10" cy="6.3" r="0.9" fill="currentColor" />
            </svg>
          </button>
        </div>
      </div>
      <DatasetNodeInfoModal
        open={false}
        title={title}
        nodeType={nodeType}
        defaultMarkdown={defaultMarkdown}
        onClose={() => {}}
      />
    </>
  );
}
