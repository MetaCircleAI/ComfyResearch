// @vitest-environment jsdom

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const scriptPath = resolve(process.cwd(), "../docs/_static/comfyresearch.js");
const templatePath = resolve(
  process.cwd(),
  "../docs/_templates/search-button-design.html",
);
const scriptSource = readFileSync(scriptPath, "utf8");
const templateSource = readFileSync(templatePath, "utf8");

const messages = {
  searchEmpty: "请输入关键词",
  searchLoading: "正在搜索",
  searchReady: "搜索结果已就绪",
  searchUnavailable: "站内搜索暂不可用，请按回车打开搜索页",
  searchNoResults: "没有找到“%(query)s”",
  copyCode: "复制代码",
  copiedLabel: "已复制",
  copiedStatus: "代码已复制到剪贴板",
};

const localizedUi: Record<string, string> = {
  search_placeholder: "搜索文档",
  search_aria_label: "搜索文档内容",
  search_button_aria_label: "提交文档搜索",
  search_empty: messages.searchEmpty,
  search_loading: messages.searchLoading,
  search_ready: messages.searchReady,
  search_unavailable: messages.searchUnavailable,
  search_no_results: messages.searchNoResults,
  copy_code: messages.copyCode,
  copied_label: messages.copiedLabel,
  copied_status: messages.copiedStatus,
};

const englishFallbacks: Record<string, string> = {
  search_placeholder: "Search the docs",
  search_aria_label: "Search the documentation",
  search_button_aria_label: "Search documentation",
  search_empty: "Type to search documentation.",
  search_loading: "Searching documentation...",
  search_ready: "Search results are ready.",
  search_unavailable:
    "Search is unavailable. Press Enter to open the full search page.",
  search_no_results: "No results for “%(query)s”.",
  copy_code: "Copy code",
  copied_label: "Copied to clipboard",
  copied_status: "Copied to clipboard.",
};

const escapeAttribute = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

const renderSearchTemplate = (
  action: string,
  values: Record<string, string> = localizedUi,
) => {
  const rendered = templateSource
    .replace("{{ pathto('search') }}", escapeAttribute(action))
    .replace(
      /\{\{\s*\(cr_ui\s*\|\s*default\(\{\}\)\)\.(\w+)\s*\|\s*default\('([^']*)',\s*true\)\s*\|\s*e\s*\}\}/g,
      (_expression, key: string, fallback: string) =>
        escapeAttribute(values[key] ?? fallback),
    );
  if (rendered.includes("{{")) throw new Error("Unrendered search template value");
  return rendered;
};

const installFixture = (count = 2) => {
  document.documentElement.innerHTML = `
    <body>
      ${Array.from({ length: count }, (_, index) =>
        renderSearchTemplate(`/manual-search-${index + 1}`),
      ).join("")}
      <button class="copybtn">copy</button>
    </body>
  `;
  window.eval(`(() => { ${scriptSource}\n })()`);
  document.dispatchEvent(new Event("DOMContentLoaded"));
};

const inputAt = (index: number) =>
  document.querySelectorAll<HTMLInputElement>(".cr-search-input")[index];

const rootAt = (index: number) =>
  document.querySelectorAll<HTMLElement>(".cr-header-search")[index];

const resultsAt = (index: number) =>
  document.querySelectorAll<HTMLElement>(".cr-search-results")[index];

const statusAt = (index: number) =>
  document.querySelectorAll<HTMLElement>(".cr-search-status")[index];

const typeQuery = (index: number, query: string) => {
  const input = inputAt(index);
  input.value = query;
  input.dispatchEvent(new Event("input", { bubbles: true }));
};

const pressArrowDown = (index: number) => {
  const event = new KeyboardEvent("keydown", {
    key: "ArrowDown",
    bubbles: true,
    cancelable: true,
  });
  inputAt(index).dispatchEvent(event);
  return event;
};

const flushMutations = async () => {
  await Promise.resolve();
  await Promise.resolve();
};

beforeEach(() => {
  vi.useFakeTimers();
  delete (window as any).Search;
});

afterEach(() => {
  vi.useRealTimers();
  document.documentElement.innerHTML = "<body></body>";
  delete (window as any).Search;
});

describe("documentation header search", () => {
  test("assigns unique accessible IDs to every search instance", () => {
    installFixture();

    const inputs = Array.from(
      document.querySelectorAll<HTMLInputElement>(".cr-search-input"),
    );
    const panels = Array.from(
      document.querySelectorAll<HTMLElement>(".cr-search-panel"),
    );
    const results = Array.from(
      document.querySelectorAll<HTMLElement>(".cr-search-results"),
    );

    expect(new Set(inputs.map(({ id }) => id)).size).toBe(2);
    expect(new Set(panels.map(({ id }) => id)).size).toBe(2);
    expect(new Set(results.map(({ id }) => id)).size).toBe(2);
    inputs.forEach((input, index) => {
      expect(input.getAttribute("aria-controls")).toBe(panels[index].id);
      expect(input.getAttribute("aria-expanded")).toBe("false");
      expect(rootAt(index).dataset.searchState).toBe("idle");
    });
  });

  test("moves through loading and ready, then releases active search for only the latest queued query", async () => {
    const calls: string[] = [];
    const outputs: HTMLElement[] = [];
    (window as any).Search = {
      performSearch(query: string) {
        calls.push(query);
        outputs.push(document.getElementById("search-results") as HTMLElement);
      },
    };
    installFixture();

    typeQuery(0, "alpha");
    expect(rootAt(0).dataset.searchState).toBe("loading");
    expect(statusAt(0).textContent).toBe(messages.searchLoading);
    await vi.advanceTimersByTimeAsync(300);

    typeQuery(1, "beta");
    await vi.advanceTimersByTimeAsync(300);
    typeQuery(1, "gamma");
    await vi.advanceTimersByTimeAsync(300);
    expect(calls).toEqual(["alpha"]);

    outputs[0].innerHTML =
      '<p class="search-summary">one result</p><ul class="search"><li><a href="/a">A</a></li></ul>';
    await flushMutations();

    expect(rootAt(0).dataset.searchState).toBe("ready");
    expect(statusAt(0).textContent).toBe(messages.searchReady);
    expect(resultsAt(0).hidden).toBe(false);
    expect(calls).toEqual(["alpha", "gamma"]);
  });

  test("a newer input immediately supersedes a queued request from another instance", async () => {
    const calls: string[] = [];
    const outputs: HTMLElement[] = [];
    (window as any).Search = {
      performSearch(query: string) {
        calls.push(query);
        outputs.push(document.getElementById("search-results") as HTMLElement);
      },
    };
    installFixture();

    typeQuery(0, "alpha");
    await vi.advanceTimersByTimeAsync(300);
    typeQuery(1, "beta");
    await vi.advanceTimersByTimeAsync(300);
    typeQuery(0, "alpha newest");

    outputs[0].innerHTML =
      '<p class="search-summary">done</p><ul class="search"><li><a href="/a">A</a></li></ul>';
    await flushMutations();
    expect(calls).toEqual(["alpha"]);

    await vi.advanceTimersByTimeAsync(300);
    expect(calls).toEqual(["alpha", "alpha newest"]);
  });

  test("shows localized no-results and empty-query states", async () => {
    (window as any).Search = {
      performSearch() {
        const output = document.getElementById("search-results") as HTMLElement;
        output.innerHTML =
          '<p class="search-summary">complete</p><ul class="search"></ul>';
      },
    };
    installFixture(1);

    typeQuery(0, "缺失");
    await vi.advanceTimersByTimeAsync(300);
    await flushMutations();
    expect(rootAt(0).dataset.searchState).toBe("empty");
    expect(statusAt(0).textContent).toBe("没有找到“缺失”");

    typeQuery(0, "  ");
    expect(rootAt(0).dataset.searchState).toBe("empty");
    expect(statusAt(0).textContent).toBe(messages.searchEmpty);
    expect(resultsAt(0).hidden).toBe(true);
  });

  test.each(["missing", "throws"] as const)(
    "globally disables inline search when Search %s",
    async (failure) => {
      if (failure === "throws") {
        (window as any).Search = {
          performSearch() {
            throw new Error("broken search index");
          },
        };
      }
      installFixture();
      resultsAt(1).innerHTML =
        '<p class="search-summary">old</p><ul class="search"><li><a href="/">old</a></li></ul>';

      typeQuery(0, "alpha");
      await vi.advanceTimersByTimeAsync(300);
      await flushMutations();

      [0, 1].forEach((index) => {
        expect(rootAt(index).dataset.searchState).toBe("unavailable");
        expect(statusAt(index).textContent).toBe(messages.searchUnavailable);
        expect(resultsAt(index).hidden).toBe(true);
      });
    },
  );

  test("times out globally and ignores a late search mutation", async () => {
    let lateOutput: HTMLElement | undefined;
    (window as any).Search = {
      performSearch() {
        lateOutput = document.getElementById("search-results") as HTMLElement;
      },
    };
    installFixture();

    typeQuery(0, "slow");
    await vi.advanceTimersByTimeAsync(2800);
    expect(rootAt(0).dataset.searchState).toBe("unavailable");
    expect(rootAt(1).dataset.searchState).toBe("unavailable");

    lateOutput!.innerHTML =
      '<p class="search-summary">late</p><ul class="search"><li><a href="/">late</a></li></ul>';
    await flushMutations();

    expect(rootAt(0).dataset.searchState).toBe("unavailable");
    expect(resultsAt(0).hidden).toBe(true);
    expect(inputAt(0).getAttribute("aria-expanded")).toBe("true");
  });

  test("submits the current GET form action and query on Enter", () => {
    installFixture(1);
    const input = inputAt(0);
    const form = input.form!;
    input.value = "a & b";
    form.requestSubmit = vi.fn();

    const keydown = new KeyboardEvent("keydown", {
      key: "Enter",
      bubbles: true,
      cancelable: true,
    });
    input.dispatchEvent(keydown);

    const target = new URL(form.action);
    target.search = new URLSearchParams(
      Array.from(new FormData(form).entries()).map(([key, value]) => [
        key,
        String(value),
      ]),
    ).toString();
    expect(keydown.defaultPrevented).toBe(true);
    expect(form.requestSubmit).toHaveBeenCalledOnce();
    expect(target.pathname).toBe("/manual-search-1");
    expect(target.search).toBe("?q=a+%26+b");
  });

  test("keeps shortcut, result navigation, Escape, and focus return", async () => {
    (window as any).Search = {
      performSearch() {
        const output = document.getElementById("search-results") as HTMLElement;
        output.innerHTML =
          '<p class="search-summary">done</p><ul class="search"><li><a href="/first">First</a></li></ul>';
      },
    };
    installFixture(1);

    window.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "k",
        ctrlKey: true,
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(inputAt(0));

    typeQuery(0, "first");
    await vi.advanceTimersByTimeAsync(300);
    await flushMutations();
    inputAt(0).dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement?.textContent).toBe("First");

    rootAt(0).dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(inputAt(0).getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(inputAt(0));
  });

  test("ArrowDown ignores stale links after Escape", async () => {
    (window as any).Search = {
      performSearch() {
        const output = document.getElementById("search-results") as HTMLElement;
        output.innerHTML =
          '<p class="search-summary">done</p><ul class="search"><li><a href="/old">Old</a></li></ul>';
      },
    };
    installFixture(1);
    typeQuery(0, "old");
    await vi.advanceTimersByTimeAsync(300);
    await flushMutations();

    rootAt(0).dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    const event = pressArrowDown(0);

    expect(document.activeElement).toBe(inputAt(0));
    expect(event.defaultPrevented).toBe(false);
  });

  test("ArrowDown ignores stale links while a newer query is loading", async () => {
    (window as any).Search = {
      performSearch() {
        const output = document.getElementById("search-results") as HTMLElement;
        output.innerHTML =
          '<p class="search-summary">done</p><ul class="search"><li><a href="/old">Old</a></li></ul>';
      },
    };
    installFixture(1);
    typeQuery(0, "old");
    await vi.advanceTimersByTimeAsync(300);
    await flushMutations();

    typeQuery(0, "new");
    inputAt(0).focus();
    const event = pressArrowDown(0);

    expect(rootAt(0).dataset.searchState).toBe("loading");
    expect(document.activeElement).toBe(inputAt(0));
    expect(event.defaultPrevented).toBe(false);
  });

  test("ArrowDown ignores stale links after inline search becomes unavailable", async () => {
    installFixture(1);
    resultsAt(0).innerHTML =
      '<p class="search-summary">old</p><ul class="search"><li><a href="/old">Old</a></li></ul>';
    typeQuery(0, "broken");
    await vi.advanceTimersByTimeAsync(300);
    await flushMutations();

    inputAt(0).focus();
    const event = pressArrowDown(0);

    expect(rootAt(0).dataset.searchState).toBe("unavailable");
    expect(document.activeElement).toBe(inputAt(0));
    expect(event.defaultPrevented).toBe(false);
  });

  test("reads all search and copy UI text from template data", async () => {
    installFixture(1);
    expect(rootAt(0).dataset.searchEmpty).toBe(messages.searchEmpty);
    expect(inputAt(0).placeholder).toBe(localizedUi.search_placeholder);
    const copyButton = document.querySelector<HTMLButtonElement>(".copybtn")!;
    expect(copyButton.getAttribute("aria-label")).toBe(messages.copyCode);

    copyButton.classList.add("success");
    await flushMutations();
    await vi.advanceTimersByTimeAsync(20);
    expect(copyButton.getAttribute("aria-label")).toBe(messages.copiedLabel);
    expect(document.querySelector(".cr-copy-status")?.textContent).toBe(
      messages.copiedStatus,
    );

    [
      "Type to search documentation.",
      "Searching documentation...",
      "Search results are ready.",
      "Search is unavailable.",
      "No results for",
      "Copy code",
      "Copied to clipboard",
    ].forEach((text) => expect(scriptSource).not.toContain(text));
  });

  test("gives every cr_ui template access an English fallback", () => {
    const accesses = Array.from(
      templateSource.matchAll(
        /\(cr_ui\s*\|\s*default\(\{\}\)\)\.(\w+)/g,
      ),
      (match) => match[1],
    );
    expect(accesses.sort()).toEqual(Object.keys(englishFallbacks).sort());
    Object.entries(englishFallbacks).forEach(([key, fallback]) =>
      expect(templateSource).toContain(
        `(cr_ui | default({})).${key} | default('${fallback}', true) | e`,
      ),
    );

    document.body.innerHTML = renderSearchTemplate("/fallback-search", {});
    const root = rootAt(0);
    [
      "search_empty",
      "search_loading",
      "search_ready",
      "search_unavailable",
      "search_no_results",
      "copy_code",
      "copied_label",
      "copied_status",
    ].forEach((key) => {
      expect(root.getAttribute(`data-${key.replaceAll("_", "-")}`)).toBe(
        englishFallbacks[key],
      );
    });
    expect(inputAt(0).placeholder).toBe(englishFallbacks.search_placeholder);
    expect(inputAt(0).getAttribute("aria-label")).toBe(
      englishFallbacks.search_aria_label,
    );
    expect(
      root.querySelector(".cr-search-submit")?.getAttribute("aria-label"),
    ).toBe(englishFallbacks.search_button_aria_label);
  });
});
