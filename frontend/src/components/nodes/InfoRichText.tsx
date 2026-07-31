/**
 * Rich info text shared by node-info surfaces: $..$/$$..$$ via KaTeX plus
 * **bold** and [label](url) links — the same rules as node-header popovers.
 */
import { useMemo } from "react";
import katex from "katex";
import { parseLatexDelimited } from "../../utils/katexDelimited";
import { renderNodeInfoTextWithLinks } from "./nodeInfoInlineLinks";

export function InfoRichText({ text }: { text: string }) {
  const segments = useMemo(() => parseLatexDelimited(text.trim()), [text]);
  return (
    <span className="cr-info-rich">
      {segments.map((seg, idx) => {
        if (seg.kind === "text") {
          return <span key={idx}>{renderNodeInfoTextWithLinks(seg.text, `info-${idx}`)}</span>;
        }
        const t = seg.latex;
        if (!t) return null;
        let html = "";
        try {
          html = katex.renderToString(t, { throwOnError: false, displayMode: seg.display });
        } catch {
          html = "";
        }
        if (!html) return <code key={idx}>{t}</code>;
        return (
          <span
            key={idx}
            className={seg.display ? "cr-info-rich__math--display" : undefined}
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      })}
    </span>
  );
}
