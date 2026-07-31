import type { ReactNode } from "react";

export type NodeInfoInlineSegment =
  | { kind: "text"; text: string }
  | { kind: "link"; label: string; href: string }
  | { kind: "bold"; text: string };

function isSafeNodeInfoHref(href: string): boolean {
  const t = href.trim();
  return /^https?:\/\//i.test(t);
}

/** Parses `[label](url)`, `<https://...>`, and `**bold**` in one line (same rules as dataset info). */
export function parseNodeInfoInlineSegments(text: string): NodeInfoInlineSegment[] {
  const out: NodeInfoInlineSegment[] = [];
  const inlineToken = /\[([^\]]+)\]\(([^)\s]+)\)|<(https?:\/\/[^>\s]+)>|\*\*([^*]+)\*\*/g;
  let lastIdx = 0;
  let m: RegExpExecArray | null = null;
  while ((m = inlineToken.exec(text))) {
    if (m.index > lastIdx) {
      out.push({ kind: "text", text: text.slice(lastIdx, m.index) });
    }
    const markdownLabel = m[1];
    const markdownHref = m[2];
    const angleHref = m[3];
    const boldText = m[4];
    const href = (markdownHref ?? angleHref ?? "").trim();
    const label = (markdownLabel ?? href).trim();
    if (boldText?.trim()) {
      out.push({ kind: "bold", text: boldText.trim() });
    } else if (href && label && isSafeNodeInfoHref(href)) {
      out.push({ kind: "link", label, href });
    } else {
      out.push({ kind: "text", text: m[0] });
    }
    lastIdx = m.index + m[0].length;
  }
  if (lastIdx < text.length) {
    out.push({ kind: "text", text: text.slice(lastIdx) });
  }
  return out;
}

export function renderNodeInfoTextWithLinks(text: string, keyPrefix: string): ReactNode[] {
  const lines = text.split("\n");
  const nodes: ReactNode[] = [];
  lines.forEach((line, lineIdx) => {
    if (lineIdx > 0) nodes.push(<br key={`${keyPrefix}-br-${lineIdx}`} />);
    const segments = parseNodeInfoInlineSegments(line);
    segments.forEach((seg, segIdx) => {
      if (seg.kind === "link") {
        nodes.push(
          <a
            key={`${keyPrefix}-link-${lineIdx}-${segIdx}`}
            className="cr-dataset-info-modal__link"
            href={seg.href}
            target="_blank"
            rel="noreferrer noopener"
          >
            {seg.label}
          </a>,
        );
      } else if (seg.kind === "bold") {
        nodes.push(<strong key={`${keyPrefix}-bold-${lineIdx}-${segIdx}`}>{seg.text}</strong>);
      } else {
        nodes.push(<span key={`${keyPrefix}-text-${lineIdx}-${segIdx}`}>{seg.text}</span>);
      }
    });
  });
  return nodes;
}
