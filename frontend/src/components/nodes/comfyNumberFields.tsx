import { useEffect, useState, type KeyboardEvent, type ReactNode } from "react";
import { parsePositiveFloat } from "./numericParse";
import { KatexMixedInline } from "./KatexMixedInline";
import { useOptionalTheme } from "../../themeContext";
import { normalizeTheme } from "../../theme";

const inputCls = "cr-input cr-comfy-widget__control cr-comfy-widget__control--num";

function blurOnEnter(e: KeyboardEvent<HTMLInputElement>) {
  if (e.key === "Enter") {
    e.currentTarget.blur();
  }
}

type FieldShellProps = {
  label?: string;
  labelNode?: ReactNode;
  title?: string;
  warn: string | null;
  children: ReactNode;
};

function FieldShell({ label, labelNode, title, warn, children }: FieldShellProps) {
  return (
    <div className="cr-comfy-field">
      <div className="cr-comfy-widget cr-comfy-widget--flush">
        <span className="cr-comfy-widget__label" title={title}>
          {labelNode ?? <KatexMixedInline text={label ?? ""} className="cr-katex-mixed-inline" />}
        </span>
        <div className="cr-comfy-widget__control-col">
          {children}
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

export type ComfyIntFieldProps = {
  label?: string;
  labelNode?: ReactNode;
  value: number;
  onCommit: (n: number) => void;
  min?: number;
  max?: number;
  title?: string;
  ariaLabel: string;
};

/** Text input (no native spinners); commit on blur / Enter. Empty or invalid → warning + revert. */
export function ComfyIntField({ label, labelNode, value, onCommit, min, max, title, ariaLabel }: ComfyIntFieldProps) {
  const [text, setText] = useState(() => String(value));
  const [warn, setWarn] = useState<string | null>(null);

  useEffect(() => {
    setText(String(value));
  }, [value]);

  const commit = () => {
    const t = text.trim();
    if (t === "") {
      setWarn("Cannot be empty.");
      setText(String(value));
      return;
    }
    const n = Number(t);
    if (!Number.isFinite(n) || !Number.isInteger(n)) {
      setWarn(Number.isFinite(n) ? "Enter a whole number." : "Invalid number.");
      setText(String(value));
      return;
    }
    if (min !== undefined && n < min) {
      setWarn(`Must be at least ${min}.`);
      setText(String(value));
      return;
    }
    if (max !== undefined && n > max) {
      setWarn(`Must be at most ${max}.`);
      setText(String(value));
      return;
    }
    setWarn(null);
    onCommit(n);
    setText(String(n));
  };

  // Studio theme: hover-revealed −/+ steppers. They step from the last
  // COMMITTED value (not the in-progress text) and preventDefault on
  // pointerdown so clicking them never blur-commits partial input first.
  // Without a provider, follow the applied <html data-cr-theme> so bare
  // renders under classic keep the exact legacy DOM.
  const contextTheme = useOptionalTheme()?.theme;
  const isStudio =
    (contextTheme ?? normalizeTheme(document.documentElement.dataset.crTheme)) !== "classic";
  const stepBy = (delta: number) => {
    let next = value + delta;
    if (min !== undefined) next = Math.max(min, next);
    if (max !== undefined) next = Math.min(max, next);
    setWarn(null);
    setText(String(next));
    if (next !== value) onCommit(next);
  };

  const input = (
    <input
      type="text"
      className={inputCls}
      inputMode="numeric"
      autoComplete="off"
      spellCheck={false}
      aria-label={ariaLabel}
      value={text}
      onChange={(e) => {
        setWarn(null);
        setText(e.target.value);
      }}
      onBlur={(e) => {
        // Keyboard path (Tab to a stepper): don't commit in-progress text —
        // the stepper must act on the last committed value. Revert instead.
        const to = e.relatedTarget as HTMLElement | null;
        if (to?.classList?.contains("cr-num-step")) {
          setWarn(null);
          setText(String(value));
          return;
        }
        commit();
      }}
      onKeyDown={blurOnEnter}
    />
  );

  return (
    <FieldShell label={label} labelNode={labelNode} title={title} warn={warn}>
      {isStudio ? (
        <span className="cr-num-wrap">
          <button
            type="button"
            className="cr-num-step nodrag nopan"
            aria-label={`${ariaLabel} decrease`}
            onPointerDown={(e) => e.preventDefault()}
            onClick={() => stepBy(-1)}
          >
            −
          </button>
          {input}
          <button
            type="button"
            className="cr-num-step nodrag nopan"
            aria-label={`${ariaLabel} increase`}
            onPointerDown={(e) => e.preventDefault()}
            onClick={() => stepBy(1)}
          >
            +
          </button>
        </span>
      ) : (
        input
      )}
    </FieldShell>
  );
}

export type ComfyFloatFieldProps = {
  label: string;
  value: number;
  onCommit: (n: number) => void;
  min?: number;
  max?: number;
  /** If true, only accepts values &gt; 0 (e.g. learning rate, epsilon). Allows 1e-4 style. */
  positiveOnly?: boolean;
  title?: string;
  ariaLabel: string;
};

export function ComfyFloatField({
  label,
  value,
  onCommit,
  min,
  max,
  positiveOnly,
  title,
  ariaLabel,
}: ComfyFloatFieldProps) {
  const [text, setText] = useState(() => String(value));
  const [warn, setWarn] = useState<string | null>(null);

  useEffect(() => {
    setText(String(value));
  }, [value]);

  const commit = () => {
    const t = text.trim();
    if (t === "") {
      setWarn("Cannot be empty.");
      setText(String(value));
      return;
    }
    let n: number | null;
    if (positiveOnly) {
      n = parsePositiveFloat(t);
      if (n === null) {
        setWarn("Enter a positive number (e.g. 0.001 or 1e-4).");
        setText(String(value));
        return;
      }
    } else {
      const raw = t.replaceAll(",", "");
      const parsed = Number(raw);
      n = Number.isFinite(parsed) ? parsed : null;
      if (n === null) {
        setWarn("Invalid number.");
        setText(String(value));
        return;
      }
    }
    if (min !== undefined && n < min) {
      setWarn(`Must be at least ${min}.`);
      setText(String(value));
      return;
    }
    if (max !== undefined && n > max) {
      setWarn(`Must be at most ${max}.`);
      setText(String(value));
      return;
    }
    setWarn(null);
    onCommit(n);
    setText(String(n));
  };

  return (
    <FieldShell label={label} title={title} warn={warn}>
      <input
        type="text"
        className={inputCls}
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
    </FieldShell>
  );
}
