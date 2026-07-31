import { useEffect, useState, type KeyboardEvent } from "react";
import { parseFloatList, parseIntList } from "./multiValueUtils";
import { parsePositiveFloat } from "./numericParse";
import { KatexMixedInline } from "./KatexMixedInline";

const inputCls = "cr-input cr-comfy-widget__control cr-comfy-widget__control--num";

function blurOnEnter(e: KeyboardEvent<HTMLInputElement>) {
  if (e.key === "Enter") e.currentTarget.blur();
}

type ShellProps = {
  label: string;
  title?: string;
  warn: string | null;
  multi: boolean;
  children: React.ReactNode;
};

function MultiFieldShell({ label, title, warn, multi, children }: ShellProps) {
  return (
    <div className={`cr-comfy-field${multi ? " cr-comfy-field--multi-choice" : ""} nodrag nopan`}>
      <div className="cr-comfy-widget cr-comfy-widget--flush">
        <span className="cr-comfy-widget__label" title={title}>
          <KatexMixedInline text={label} className="cr-katex-mixed-inline" />
        </span>
        <div className="cr-comfy-widget__control-col">
          {multi ? <div className="cr-comfy-sweep-anchor">{children}</div> : children}
          {warn ? (
            <span className="cr-comfy-widget__warn" role="alert">
              {warn}
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export type ComfyIntListFieldProps = {
  label: string;
  values: number[];
  onCommit: (next: number[]) => void;
  min?: number;
  max?: number;
  title?: string;
  ariaLabel: string;
};

/** Comma / space separated integers; one or more values. Multiple values → red highlight on parent. */
export function ComfyIntListField({ label, values, onCommit, min, max, title, ariaLabel }: ComfyIntListFieldProps) {
  const [text, setText] = useState(() => values.join(", "));
  const [warn, setWarn] = useState<string | null>(null);
  const multi = values.length > 1;
  /** Parent often passes a fresh `values` array each render (`intChoices`); sync on contents, not reference. */
  const valuesSerialized = values.join(", ");

  useEffect(() => {
    setText(valuesSerialized);
  }, [valuesSerialized]);

  const commit = () => {
    const t = text.trim();
    if (t === "") {
      setWarn("Enter at least one integer.");
      setText(valuesSerialized);
      return;
    }
    try {
      const parsed = parseIntList(t, min, max);
      if (!parsed.length) {
        setWarn("Enter at least one integer.");
        setText(valuesSerialized);
        return;
      }
      setWarn(null);
      onCommit(parsed);
      setText(parsed.join(", "));
    } catch (e) {
      setWarn(e instanceof Error ? e.message : String(e));
      setText(valuesSerialized);
    }
  };

  return (
    <MultiFieldShell label={label} title={title} warn={warn} multi={multi}>
      <input
        type="text"
        className={`${inputCls} nodrag nopan`}
        inputMode="numeric"
        autoComplete="off"
        spellCheck={false}
        aria-label={ariaLabel}
        value={text}
        onChange={(e) => {
          setWarn(null);
          setText(e.target.value);
        }}
        onBlur={commit}
        onKeyDown={blurOnEnter}
      />
    </MultiFieldShell>
  );
}

export type ComfyFloatListFieldProps = {
  label: string;
  values: number[];
  onCommit: (next: number[]) => void;
  min?: number;
  max?: number;
  positiveOnly?: boolean;
  title?: string;
  ariaLabel: string;
};

export function ComfyFloatListField({
  label,
  values,
  onCommit,
  min,
  max,
  positiveOnly,
  title,
  ariaLabel,
}: ComfyFloatListFieldProps) {
  const [text, setText] = useState(() => values.join(", "));
  const [warn, setWarn] = useState<string | null>(null);
  const multi = values.length > 1;
  const valuesSerialized = values.join(", ");

  useEffect(() => {
    setText(valuesSerialized);
  }, [valuesSerialized]);

  const commit = () => {
    const t = text.trim();
    if (t === "") {
      setWarn("Enter at least one number.");
      setText(valuesSerialized);
      return;
    }
    try {
      let parsed: number[];
      if (positiveOnly) {
        const parts = t.split(/[,;\s]+/).map((x) => x.trim()).filter(Boolean);
        parsed = [];
        for (const p of parts) {
          const n = parsePositiveFloat(p);
          if (n === null) throw new Error(`Not a positive number: ${p}`);
          if (min !== undefined && n < min) throw new Error(`Must be >= ${min}`);
          if (max !== undefined && n > max) throw new Error(`Must be <= ${max}`);
          parsed.push(n);
        }
      } else {
        parsed = parseFloatList(t, { min, max, positiveOnly: false });
      }
      if (!parsed.length) {
        setWarn("Enter at least one number.");
        setText(valuesSerialized);
        return;
      }
      setWarn(null);
      onCommit(parsed);
      setText(parsed.join(", "));
    } catch (e) {
      setWarn(e instanceof Error ? e.message : String(e));
      setText(valuesSerialized);
    }
  };

  return (
    <MultiFieldShell label={label} title={title} warn={warn} multi={multi}>
      <input
        type="text"
        className={`${inputCls} nodrag nopan`}
        inputMode="decimal"
        autoComplete="off"
        spellCheck={false}
        aria-label={ariaLabel}
        value={text}
        onChange={(e) => {
          setWarn(null);
          setText(e.target.value);
        }}
        onBlur={commit}
        onKeyDown={blurOnEnter}
      />
    </MultiFieldShell>
  );
}
