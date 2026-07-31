import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent,
} from "react";
import { createPortal } from "react-dom";
import type { SavedGraphEntry } from "../graph/savedGraphLibrary";
import type { GraphFileExportTier } from "../graph/graphFileExportTier";

/** Native `title` waits ~1s on first hover; custom tip is faster and still quick between rows. */
const NAME_TIP_DELAY_MS = 220;
const NAME_TIP_SWITCH_ROW_MS = 45;

function tierLabel(t: GraphFileExportTier): string {
  switch (t) {
    case "small":
      return "S";
    case "medium":
      return "M";
    case "large":
      return "L";
    default:
      return "?";
  }
}

export type SavedGraphLibrarySectionGroup = {
  title: string;
  entries: SavedGraphEntry[];
  /** Optional presentation-only name override; saved template metadata is unchanged. */
  displayEntryName?: (entry: SavedGraphEntry) => string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
};

type SavedGraphLibraryPanelProps = {
  title: string;
  emptyHint: string;
  onOpen: (entry: SavedGraphEntry) => void;
  onDelete: (entry: SavedGraphEntry) => void;
  /** When set, right-click a row → Rename for in-place edit (e.g. templates). */
  onRename?: (entry: SavedGraphEntry, newName: string) => void | Promise<void>;
} & (
  | { entries: SavedGraphEntry[]; sectionGroups?: never }
  | { sectionGroups: SavedGraphLibrarySectionGroup[]; entries?: never }
);

function entryMatchesQuery(entry: SavedGraphEntry, q: string): boolean {
  if (!q) return true;
  return (
    entry.name.toLowerCase().includes(q) ||
    entry.id.toLowerCase().includes(q) ||
    (entry.libraryOrigin != null && String(entry.libraryOrigin).toLowerCase().includes(q))
  );
}

export function SavedGraphLibraryPanel({
  title,
  emptyHint,
  onOpen,
  onDelete,
  onRename,
  ...rest
}: SavedGraphLibraryPanelProps) {
  const sectionGroups = "sectionGroups" in rest ? rest.sectionGroups : undefined;
  const entries = "entries" in rest ? rest.entries : undefined;
  const useSectionGroups = Array.isArray(sectionGroups) && sectionGroups.length > 0;
  const combinedEntries = useSectionGroups
    ? sectionGroups.flatMap((g) => g.entries)
    : (entries ?? []);
  const [query, setQuery] = useState("");
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(
    () => new Set(sectionGroups?.filter((group) => group.defaultCollapsed).map((group) => group.title)),
  );
  const [ctx, setCtx] = useState<{ x: number; y: number; entry: SavedGraphEntry } | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const renameInputRef = useRef<HTMLInputElement | null>(null);
  const commitRenameLockRef = useRef(false);
  const nameTipTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Which row's ``setTimeout`` is pending; avoids canceling the next row when leave fires after enter. */
  const pendingNameTipEntryIdRef = useRef<string | null>(null);
  const [nameTip, setNameTip] = useState<{ text: string; left: number; top: number; maxW: number } | null>(null);

  const clearNameTipTimer = useCallback(() => {
    if (nameTipTimerRef.current !== null) {
      clearTimeout(nameTipTimerRef.current);
      nameTipTimerRef.current = null;
    }
  }, []);

  const hideNameTip = useCallback(() => {
    pendingNameTipEntryIdRef.current = null;
    clearNameTipTimer();
    setNameTip(null);
  }, [clearNameTipTimer]);

  const scheduleNameTip = useCallback(
    (openButton: HTMLElement, entryId: string, text: string, delayMs: number) => {
      clearNameTipTimer();
      pendingNameTipEntryIdRef.current = entryId;
      const row = openButton.closest(".cr-library-panel__row");
      const rect = (row instanceof HTMLElement ? row : openButton).getBoundingClientRect();
      const maxW = Math.min(420, Math.max(160, window.innerWidth - 16));
      const gap = 8;
      let left = rect.right + gap;
      const maxRight = window.innerWidth - 8;
      if (left + maxW > maxRight) {
        left = Math.max(8, maxRight - maxW);
      }
      const top = Math.max(16, Math.min(window.innerHeight - 16, rect.top + rect.height / 2));
      nameTipTimerRef.current = setTimeout(() => {
        nameTipTimerRef.current = null;
        pendingNameTipEntryIdRef.current = null;
        setNameTip({ text, left, top, maxW });
      }, delayMs);
    },
    [clearNameTipTimer],
  );

  const onOpenButtonMouseEnter = useCallback(
    (entry: SavedGraphEntry, ev: MouseEvent<HTMLButtonElement>, displayName = entry.name) => {
      const btn = ev.currentTarget;
      const from = ev.nativeEvent.relatedTarget;
      const fromOtherRowOpen =
        from instanceof Element &&
        from !== btn &&
        Boolean(from.closest(".cr-library-panel__open"));
      const delayMs = fromOtherRowOpen ? NAME_TIP_SWITCH_ROW_MS : NAME_TIP_DELAY_MS;
      scheduleNameTip(btn, entry.id, displayName, delayMs);
    },
    [scheduleNameTip],
  );

  const onOpenButtonMouseLeave = useCallback(
    (entry: SavedGraphEntry, ev: MouseEvent<HTMLButtonElement>) => {
      const to = ev.nativeEvent.relatedTarget;
      const destOpen =
        to instanceof Element ? (to.closest(".cr-library-panel__open") as HTMLElement | null) : null;
      if (destOpen && destOpen !== ev.currentTarget) {
        setNameTip(null);
        if (pendingNameTipEntryIdRef.current === entry.id) {
          clearNameTipTimer();
          pendingNameTipEntryIdRef.current = null;
        }
        return;
      }
      hideNameTip();
    },
    [clearNameTipTimer, hideNameTip],
  );

  useEffect(() => {
    return () => {
      pendingNameTipEntryIdRef.current = null;
      clearNameTipTimer();
    };
  }, [clearNameTipTimer]);

  const panelScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!nameTip) return;
    const el = panelScrollRef.current;
    const onScrollOrResize = () => hideNameTip();
    window.addEventListener("resize", onScrollOrResize);
    el?.addEventListener("scroll", onScrollOrResize);
    return () => {
      window.removeEventListener("resize", onScrollOrResize);
      el?.removeEventListener("scroll", onScrollOrResize);
    };
  }, [nameTip, hideNameTip]);

  const qNorm = query.trim().toLowerCase();

  const filteredEntries = useMemo(() => {
    if (useSectionGroups) return [];
    const list = entries ?? [];
    if (!qNorm) return list;
    return list.filter((e) => entryMatchesQuery(e, qNorm));
  }, [entries, qNorm, useSectionGroups]);

  const filteredSectionGroups = useMemo(() => {
    if (!useSectionGroups || !sectionGroups) return [];
    return sectionGroups.map((g) => ({
      ...g,
      entries: !qNorm ? g.entries : g.entries.filter((e) => entryMatchesQuery(e, qNorm)),
    }));
  }, [sectionGroups, qNorm, useSectionGroups]);

  const closeCtx = useCallback(() => setCtx(null), []);

  useEffect(() => {
    if (!ctx) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeCtx();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ctx, closeCtx]);

  useLayoutEffect(() => {
    if (!editingId || !renameInputRef.current) return;
    const el = renameInputRef.current;
    el.focus();
    el.select();
  }, [editingId]);

  const onRowContextMenu = useCallback(
    (entry: SavedGraphEntry, ev: MouseEvent) => {
      if (!onRename) return;
      ev.preventDefault();
      setCtx({ x: ev.clientX, y: ev.clientY, entry });
    },
    [onRename],
  );

  const commitRename = useCallback(
    async (entry: SavedGraphEntry) => {
      if (!onRename || commitRenameLockRef.current) return;
      const trimmed = draftName.trim();
      if (!trimmed) {
        setDraftName(entry.name);
        setEditingId(null);
        return;
      }
      if (trimmed === entry.name) {
        setEditingId(null);
        return;
      }
      commitRenameLockRef.current = true;
      try {
        await onRename(entry, trimmed);
        setEditingId(null);
      } catch {
        /* parent shows notice; stay in edit mode */
      } finally {
        commitRenameLockRef.current = false;
      }
    },
    [draftName, onRename],
  );

  const ctxMenu =
    ctx && onRename
      ? createPortal(
          <>
            <button
              type="button"
              className="cr-library-panel__ctx-backdrop"
              aria-label="Close menu"
              onClick={closeCtx}
            />
            <div
              className="cr-library-panel__ctx-menu"
              role="menu"
              style={{ left: ctx.x, top: ctx.y }}
            >
              <button
                type="button"
                role="menuitem"
                className="cr-library-panel__ctx-item"
                onClick={() => {
                  setEditingId(ctx.entry.id);
                  setDraftName(ctx.entry.name);
                  closeCtx();
                }}
              >
                Rename…
              </button>
            </div>
          </>,
          document.body,
        )
      : null;

  const nameTipPortal =
    nameTip &&
    createPortal(
      <div
        className="cr-library-panel__name-tip"
        role="tooltip"
        style={{ left: nameTip.left, top: nameTip.top, maxWidth: nameTip.maxW }}
      >
        {nameTip.text}
      </div>,
      document.body,
    );

  const searchPlaceholder = title.toLowerCase().includes("template")
    ? "Search templates…"
    : title.toLowerCase().includes("workflow")
      ? "Search workflows…"
      : `Search ${title.toLowerCase()}…`;

  return (
    <aside className="cr-nodes-panel" aria-label={title}>
      <header className="cr-nodes-panel__header">
        <h2 className="cr-nodes-panel__title">{title}</h2>
        <div className="cr-nodes-panel__search-row">
          <input
            type="search"
            className="cr-nodes-panel__search nodrag nopan"
            placeholder={searchPlaceholder}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={combinedEntries.length === 0}
            aria-label={`Search ${title}`}
          />
        </div>
      </header>
      <div ref={panelScrollRef} className="cr-nodes-panel__scroll">
        {combinedEntries.length === 0 ? (
          <p className="cr-library-panel__empty">{emptyHint}</p>
        ) : useSectionGroups ? (
          !filteredSectionGroups.some((g) => g.entries.length > 0) ? (
            <p className="cr-library-panel__empty">No {title.toLowerCase()} match your search.</p>
          ) : (
            <>
              {filteredSectionGroups.map((group) => {
                if (group.entries.length === 0 && qNorm) return null;
                const isCollapsed = Boolean(group.collapsible && !qNorm && collapsedGroups.has(group.title));
                return (
                  <div key={group.title} className="cr-nodes-panel__section">
                    <h3
                      className={
                        group.collapsible
                          ? "cr-nodes-panel__section-title cr-nodes-panel__section-title--collapsible"
                          : "cr-nodes-panel__section-title"
                      }
                    >
                      {group.collapsible ? (
                        <button
                          type="button"
                          className="cr-nodes-panel__section-toggle"
                          aria-expanded={!isCollapsed}
                          onClick={() => {
                            setCollapsedGroups((current) => {
                              const next = new Set(current);
                              if (next.has(group.title)) next.delete(group.title);
                              else next.add(group.title);
                              return next;
                            });
                          }}
                        >
                          <span aria-hidden>{isCollapsed ? "▸" : "▾"}</span>
                          {group.title} ({group.entries.length})
                        </button>
                      ) : (
                        group.title
                      )}
                    </h3>
                    <div className="cr-nodes-panel__section-body" hidden={isCollapsed}>
                      {group.entries.length > 0 ? (
                        <ul className="cr-library-panel__list">
                          {group.entries.map((entry) => (
                            <li
                              key={entry.id}
                              className="cr-library-panel__row"
                              onContextMenu={(ev) => onRowContextMenu(entry, ev)}
                            >
                              {editingId === entry.id && onRename ? (
                                <div className="cr-library-panel__open cr-library-panel__open--rename">
                                  <input
                                    ref={renameInputRef}
                                    type="text"
                                    className="cr-library-panel__rename-input nodrag nopan"
                                    value={draftName}
                                    aria-label="Template name"
                                    onChange={(e) => setDraftName(e.target.value)}
                                    onBlur={() => void commitRename(entry)}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") {
                                        e.preventDefault();
                                        void commitRename(entry);
                                        return;
                                      }
                                      if (e.key === "Escape") {
                                        e.preventDefault();
                                        setDraftName(entry.name);
                                        setEditingId(null);
                                      }
                                    }}
                                    onClick={(e) => e.stopPropagation()}
                                  />
                                  <span className="cr-library-panel__meta" title="Saved size tier">
                                    {tierLabel(entry.tier)}
                                  </span>
                                </div>
                              ) : (
                                <button
                                  type="button"
                                  className="cr-library-panel__open"
                                  onMouseEnter={(e) => onOpenButtonMouseEnter(entry, e, group.displayEntryName?.(entry) ?? entry.name)}
                                  onMouseLeave={(e) => onOpenButtonMouseLeave(entry, e)}
                                  onClick={() => onOpen(entry)}
                                >
                                  <span className="cr-library-panel__name">{group.displayEntryName?.(entry) ?? entry.name}</span>
                                  <span className="cr-library-panel__meta" title="Saved size tier">
                                    {tierLabel(entry.tier)}
                                  </span>
                                </button>
                              )}
                              <button
                                type="button"
                                className="cr-library-panel__delete"
                                title="Remove from library"
                                aria-label={`Delete ${group.displayEntryName?.(entry) ?? entry.name}`}
                                onClick={() => onDelete(entry)}
                              >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
                                  <path
                                    d="M9 3h6M4 7h16M6 7l1 14h10l1-14M10 11v6M14 11v6"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  />
                                </svg>
                              </button>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="cr-library-panel__empty">None yet.</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </>
          )
        ) : filteredEntries.length === 0 ? (
          <p className="cr-library-panel__empty">No {title.toLowerCase()} match your search.</p>
        ) : (
          <ul className="cr-library-panel__list">
            {filteredEntries.map((entry) => (
              <li
                key={entry.id}
                className="cr-library-panel__row"
                onContextMenu={(ev) => onRowContextMenu(entry, ev)}
              >
                {editingId === entry.id && onRename ? (
                  <div className="cr-library-panel__open cr-library-panel__open--rename">
                    <input
                      ref={renameInputRef}
                      type="text"
                      className="cr-library-panel__rename-input nodrag nopan"
                      value={draftName}
                      aria-label="Template name"
                      onChange={(e) => setDraftName(e.target.value)}
                      onBlur={() => void commitRename(entry)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          void commitRename(entry);
                          return;
                        }
                        if (e.key === "Escape") {
                          e.preventDefault();
                          setDraftName(entry.name);
                          setEditingId(null);
                        }
                      }}
                      onClick={(e) => e.stopPropagation()}
                    />
                    <span className="cr-library-panel__meta" title="Saved size tier">
                      {tierLabel(entry.tier)}
                    </span>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="cr-library-panel__open"
                    onMouseEnter={(e) => onOpenButtonMouseEnter(entry, e)}
                    onMouseLeave={(e) => onOpenButtonMouseLeave(entry, e)}
                    onClick={() => onOpen(entry)}
                  >
                    <span className="cr-library-panel__name">{entry.name}</span>
                    <span className="cr-library-panel__meta" title="Saved size tier">
                      {tierLabel(entry.tier)}
                    </span>
                  </button>
                )}
                <button
                  type="button"
                  className="cr-library-panel__delete"
                  title="Remove from library"
                  aria-label={`Delete ${entry.name}`}
                  onClick={() => onDelete(entry)}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
                    <path
                      d="M9 3h6M4 7h16M6 7l1 14h10l1-14M10 11v6M14 11v6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      {ctxMenu}
      {nameTipPortal}
    </aside>
  );
}
