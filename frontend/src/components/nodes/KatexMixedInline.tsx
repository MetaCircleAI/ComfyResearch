import { useMemo } from "react";
import katex from "katex";
import { parseLatexDelimited } from "../../utils/katexDelimited";

/** Renders ``text`` with inline (``$...$``) and display (``$$...$$``) KaTeX segments. Plain text passes through. */
export function KatexMixedInline({
  text,
  className = "cr-katex-mixed-inline",
}: {
  text: string;
  className?: string;
}) {
  const segments = useMemo(() => parseLatexDelimited(text), [text]);
  return (
    <span className={className}>
      {segments.map((seg, idx) => {
        if (seg.kind === "text") {
          return (
            <span key={idx}>
              {seg.text.split("\n").map((line, li) => (
                <span key={li}>
                  {li > 0 ? <br /> : null}
                  {line}
                </span>
              ))}
            </span>
          );
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
            <code key={idx} className="cr-katex-mixed-inline__fallback">
              {t}
            </code>
          );
        }
        if (seg.display) {
          return (
            <span
              key={idx}
              className="cr-katex-mixed-inline__math cr-katex-mixed-inline__math--display nodrag nopan"
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        }
        return (
          <span
            key={idx}
            className="cr-katex-mixed-inline__math nodrag nopan"
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      })}
    </span>
  );
}
