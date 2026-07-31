import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useDismissOnOutsidePointer } from "../../hooks/useDismissOnOutsidePointer";
import { KatexMixedInline } from "./KatexMixedInline";
import { enumChoices, isMultiChoice, packEnumList, type ListOr1 } from "./multiValueUtils";
import { useOptionalTheme } from "../../themeContext";
import { normalizeTheme } from "../../theme";

export type DiscreteMultiSelectOptionGroup<T extends string> = {
  title: string;
  options: { id: T; label: string }[];
};

export function DiscreteMultiSelect<T extends string>({
  label,
  options: optionsProp,
  optionGroups,
  value,
  onCommit,
  ariaLabel,
  allowEmpty = false,
  /** When true, always pick exactly one option (no sweep multi-select); clicking another row replaces the value. */
  singleSelect = false,
  disabled = false,
  labelLayout = "inline",
  matchModalInput = false,
  presentation = "dropdown",
  segmentLabels,
}: {
  label: string;
  /** Flat list; omit when using `optionGroups`. */
  options?: { id: T; label: string }[];
  /** When set, panel shows titled sections (e.g. curve annotator labels). Flattened for value logic. */
  optionGroups?: DiscreteMultiSelectOptionGroup<T>[];
  value: ListOr1<T>;
  onCommit: (next: ListOr1<T>) => void;
  ariaLabel: string;
  /** When true, the last selected option can be unchecked (zero selections). */
  allowEmpty?: boolean;
  singleSelect?: boolean;
  /** Disable trigger + panel (e.g. while a long-running graph assist action is in progress). */
  disabled?: boolean;
  labelLayout?: "inline" | "stacked";
  matchModalInput?: boolean;
  /**
   * "segmented" renders inline radio segments instead of the popup under the
   * studio theme (audited call sites only: requires singleSelect and short
   * labels). Classic — and non-single-select — always keeps the dropdown DOM.
   */
  presentation?: "dropdown" | "segmented";
  /** Optional compact per-option labels for segmented rendering; the full
   * label stays available via each segment's title tooltip. */
  segmentLabels?: Partial<Record<T, string>>;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const uid = useId().replace(/:/g, "");

  const options = useMemo(() => {
    if (optionGroups?.length) return optionGroups.flatMap((g) => g.options);
    return optionsProp ?? [];
  }, [optionGroups, optionsProp]);

  const allowed = useMemo(() => new Set(options.map((o) => o.id)), [options]);
  const fallback = options[0]?.id ?? ("" as T);
  const selected = enumChoices(value, allowed, fallback, allowEmpty);
  const multi =
    !singleSelect && (isMultiChoice(value) || (allowEmpty && Array.isArray(value) && value.length === 0));

  const summary = useMemo(() => {
    if (selected.length === 0) return allowEmpty ? "None" : (options.find((o) => o.id === fallback)?.label ?? "—");
    if (selected.length === 1) {
      return options.find((o) => o.id === selected[0])?.label ?? String(selected[0]);
    }
    return `${selected.length} selected`;
  }, [selected, options, allowEmpty, fallback]);

  useDismissOnOutsidePointer(open, () => setOpen(false), wrapRef);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  const toggle = (id: T) => {
    const set = new Set(selected);
    if (set.has(id)) {
      if (!allowEmpty && set.size <= 1) return;
      set.delete(id);
    } else {
      set.add(id);
    }
    onCommit(packEnumList([...set]) as ListOr1<T>);
  };

  const pickOption = (id: T) => {
    if (disabled) return;
    if (!singleSelect) {
      toggle(id);
      return;
    }
    if (selected.includes(id) && selected.length === 1) {
      if (allowEmpty) {
        onCommit([] as ListOr1<T>);
        setOpen(false);
      }
      return;
    }
    onCommit(id);
    setOpen(false);
  };

  const triggerBtn = (
    <button
      type="button"
      className={`cr-select cr-discrete-multi-dd__btn nodrag nopan${matchModalInput ? " cr-discrete-multi-dd__btn--modal-input" : ""}`}
      aria-expanded={open}
      aria-haspopup="listbox"
      aria-label={ariaLabel}
      disabled={disabled}
      aria-disabled={disabled || undefined}
      onClick={() => {
        if (disabled) return;
        setOpen((o) => !o);
      }}
    >
      <span className="cr-discrete-multi-dd__btn-text">
        <KatexMixedInline text={summary} className="cr-katex-mixed-inline cr-discrete-multi-dd__katex" />
      </span>
      <span className="cr-discrete-multi-dd__caret" aria-hidden>
        ▾
      </span>
    </button>
  );

  const widgetRowClass =
    labelLayout === "stacked"
      ? "cr-comfy-widget cr-comfy-widget--flush cr-comfy-widget--stack"
      : "cr-comfy-widget cr-comfy-widget--flush";

  // Studio-only segmented rendering; classic keeps the exact dropdown DOM.
  const contextTheme = useOptionalTheme()?.theme;
  const isStudio =
    (contextTheme ?? normalizeTheme(document.documentElement.dataset.crTheme)) !== "classic";
  const segmented = presentation === "segmented" && singleSelect && isStudio && !optionGroups?.length;
  const segmentedControl = segmented ? (
    <div className="cr-segmented nodrag nopan" role="radiogroup" aria-label={ariaLabel}>
      {options.map((o) => {
        const on = selected.length === 1 && selected[0] === o.id;
        return (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={on}
            disabled={disabled}
            className={`cr-segmented__seg${on ? " cr-segmented__seg--on" : ""}`}
            title={o.label}
            onClick={() => {
              if (!disabled && !on) onCommit(o.id);
            }}
          >
            {segmentLabels?.[o.id] ?? o.label}
          </button>
        );
      })}
    </div>
  ) : null;

  return (
    <div
      ref={wrapRef}
      className={`cr-comfy-field nodrag nopan${multi ? " cr-comfy-field--multi-choice" : ""}`}
    >
      <div className={widgetRowClass}>
        {labelLayout === "stacked" ? (
          <span className="cr-discrete-multi-dd__stacked-label">{label}</span>
        ) : (
          <span className="cr-comfy-widget__label">{label}</span>
        )}
        <div className="cr-comfy-widget__control-col cr-discrete-multi-dd">
          {segmentedControl ??
            (multi ? <div className="cr-comfy-sweep-anchor">{triggerBtn}</div> : triggerBtn)}
          {open && !segmented ? (
            <div
              className="cr-discrete-multi-dd__panel nodrag nopan"
              role="listbox"
              aria-label={ariaLabel}
              onMouseDown={(e) => e.stopPropagation()}
              onWheel={(e) => e.stopPropagation()}
            >
              {optionGroups?.length
                ? optionGroups.map((group) => (
                    <div key={group.title} className="cr-discrete-multi-dd__group">
                      <div className="cr-discrete-multi-dd__group-title" role="presentation">
                        {group.title}
                      </div>
                      {group.options.map((o) => {
                        const rowId = `dm-${uid}-${String(o.id).replace(/\W/g, "_")}`;
                        return (
                          <label key={o.id} className="cr-discrete-multi-dd__row" htmlFor={rowId}>
                            <input
                              id={rowId}
                              type="checkbox"
                              className="nodrag nopan"
                              disabled={disabled}
                              checked={selected.includes(o.id)}
                              onChange={() => pickOption(o.id)}
                            />
                            <span className="cr-discrete-multi-dd__label">
                              <KatexMixedInline
                                text={o.label}
                                className="cr-katex-mixed-inline cr-discrete-multi-dd__katex"
                              />
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  ))
                : options.map((o) => {
                    const rowId = `dm-${uid}-${String(o.id).replace(/\W/g, "_")}`;
                    return (
                      <label key={o.id} className="cr-discrete-multi-dd__row" htmlFor={rowId}>
                        <input
                          id={rowId}
                          type="checkbox"
                          className="nodrag nopan"
                          disabled={disabled}
                          checked={selected.includes(o.id)}
                          onChange={() => pickOption(o.id)}
                        />
                        <span className="cr-discrete-multi-dd__label">
                          <KatexMixedInline text={o.label} className="cr-katex-mixed-inline cr-discrete-multi-dd__katex" />
                        </span>
                      </label>
                    );
                  })}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
