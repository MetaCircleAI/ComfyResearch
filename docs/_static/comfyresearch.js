const setupHeaderSearch = () => {
  const roots = Array.from(document.querySelectorAll(".cr-header-search"));
  const instances = [];
  let activeRequest = null;
  let queuedRequest = null;
  let inlineSearchUnavailable = false;

  const setExpanded = (instance, expanded) => {
    instance.panel.hidden = !expanded;
    instance.input.setAttribute("aria-expanded", String(expanded));
  };

  const setState = (instance, state, message = "") => {
    instance.root.dataset.searchState = state;
    instance.status.dataset.state = state;
    instance.status.textContent = message;
    instance.status.hidden = !message;
  };

  const cancelQueuedRequest = (instance) => {
    if (queuedRequest?.instance === instance) queuedRequest = null;
  };

  const runNextRequest = () => {
    if (activeRequest || !queuedRequest || inlineSearchUnavailable) return;
    const request = queuedRequest;
    queuedRequest = null;

    if (request.version !== request.instance.version) {
      runNextRequest();
      return;
    }

    const { instance } = request;
    activeRequest = request;
    instance.results.replaceChildren();
    instance.results.hidden = false;
    setExpanded(instance, true);
    setState(instance, "loading", instance.messages.searchLoading);

    if (typeof Search === "undefined") {
      disableInlineSearch();
      return;
    }

    const pageResults = document.getElementById("search-results");
    pageResults?.removeAttribute("id");
    instance.results.id = "search-results";
    try {
      Search.performSearch(request.query);
    } catch {
      disableInlineSearch();
    } finally {
      instance.results.id = instance.resultsId;
      pageResults?.setAttribute("id", "search-results");
    }

    if (inlineSearchUnavailable || activeRequest !== request) return;
    request.failureTimer = window.setTimeout(() => {
      if (activeRequest === request) disableInlineSearch();
    }, 2500);
  };

  const queueRequest = (instance, query, version) => {
    if (inlineSearchUnavailable) return;
    queuedRequest = { instance, query, version, failureTimer: null };
    runNextRequest();
  };

  const finishRequest = (instance) => {
    if (inlineSearchUnavailable || activeRequest?.instance !== instance) return;

    const summary = instance.results.querySelector(".search-summary");
    const summaryText = summary?.textContent.trim();
    if (!summaryText) return;

    const request = activeRequest;
    window.clearTimeout(request.failureTimer);
    activeRequest = null;

    if (request.version === instance.version) {
      const hasResults = Boolean(instance.results.querySelector("ul.search a"));
      instance.results.hidden = false;
      setExpanded(instance, true);
      if (hasResults) {
        setState(instance, "ready", instance.messages.searchReady);
      } else {
        const message = instance.messages.searchNoResults
          .split("%(query)s")
          .join(request.query);
        setState(instance, "empty", message);
      }
    }

    runNextRequest();
  };

  function disableInlineSearch() {
    if (inlineSearchUnavailable) return;
    inlineSearchUnavailable = true;
    window.clearTimeout(activeRequest?.failureTimer);
    activeRequest = null;
    queuedRequest = null;

    instances.forEach((instance) => {
      instance.version += 1;
      window.clearTimeout(instance.searchTimer);
      instance.observer.disconnect();
      instance.results.hidden = true;
      setExpanded(instance, true);
      setState(instance, "unavailable", instance.messages.searchUnavailable);
    });
  }

  roots.forEach((root, index) => {
    const form = root.querySelector(".cr-search-form");
    const input = root.querySelector(".cr-search-input");
    const panel = root.querySelector(".cr-search-panel");
    const status = root.querySelector(".cr-search-status");
    const results = root.querySelector(".cr-search-results");
    if (!form || !input || !panel || !status || !results) return;

    const instanceNumber = index + 1;
    const instance = {
      root,
      input,
      panel,
      status,
      results,
      resultsId: `cr-search-results-${instanceNumber}`,
      searchTimer: null,
      version: 0,
      observer: null,
      messages: {
        searchEmpty: root.dataset.searchEmpty ?? "",
        searchLoading: root.dataset.searchLoading ?? "",
        searchReady: root.dataset.searchReady ?? "",
        searchUnavailable: root.dataset.searchUnavailable ?? "",
        searchNoResults: root.dataset.searchNoResults ?? "",
      },
    };

    input.id = `cr-search-input-${instanceNumber}`;
    panel.id = `cr-search-panel-${instanceNumber}`;
    results.id = instance.resultsId;
    input.setAttribute("aria-controls", panel.id);
    input.setAttribute("aria-expanded", "false");
    results.hidden = true;
    setState(instance, "idle");

    instance.observer = new MutationObserver(() => finishRequest(instance));
    instance.observer.observe(results, {
      childList: true,
      characterData: true,
      subtree: true,
    });

    const closePanel = () => {
      instance.version += 1;
      window.clearTimeout(instance.searchTimer);
      cancelQueuedRequest(instance);
      instance.results.hidden = true;
      setExpanded(instance, false);
      if (!inlineSearchUnavailable) setState(instance, "idle");
    };

    input.addEventListener("input", () => {
      instance.version += 1;
      const version = instance.version;
      window.clearTimeout(instance.searchTimer);
      queuedRequest = null;
      instance.results.hidden = true;

      if (inlineSearchUnavailable) {
        setExpanded(instance, true);
        setState(instance, "unavailable", instance.messages.searchUnavailable);
        return;
      }

      const query = input.value.trim();
      setExpanded(instance, true);
      if (!query) {
        setState(instance, "empty", instance.messages.searchEmpty);
        return;
      }

      setState(instance, "loading", instance.messages.searchLoading);
      instance.searchTimer = window.setTimeout(() => {
        queueRequest(instance, query, version);
      }, 300);
    });

    root.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closePanel();
        input.focus();
        return;
      }

      if (event.key === "Enter" && event.target === input) {
        event.preventDefault();
        form.requestSubmit();
        return;
      }

      if (
        event.key === "ArrowDown" &&
        event.target === input &&
        root.dataset.searchState === "ready" &&
        !results.hidden
      ) {
        const firstResult = results.querySelector("ul.search a");
        if (firstResult) {
          event.preventDefault();
          firstResult.focus();
        }
      }
    });

    document.addEventListener("pointerdown", (event) => {
      if (!root.contains(event.target)) closePanel();
    });

    instances.push(instance);
  });

  window.addEventListener("keydown", (event) => {
    const mac = navigator.platform.startsWith("Mac") || navigator.platform === "iPhone";
    const shortcut = mac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
    if (!shortcut || event.altKey || event.shiftKey || event.key.toLowerCase() !== "k") return;

    const visibleInstance =
      instances.find(({ root }) => root.getClientRects().length > 0) ?? instances[0];
    if (!visibleInstance) return;
    event.preventDefault();
    visibleInstance.input.focus();
    visibleInstance.input.select();
  }, true);
};

const setupCopyButtons = () => {
  const root = document.querySelector(".cr-header-search");
  const messages = {
    copyCode: root?.dataset.copyCode ?? "",
    copiedLabel: root?.dataset.copiedLabel ?? "",
    copiedStatus: root?.dataset.copiedStatus ?? "",
  };
  const status = document.createElement("span");
  status.className = "visually-hidden cr-copy-status";
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  document.body.append(status);

  const enhancedButtons = new WeakSet();
  const syncButton = (button) => {
    if (button.classList.contains("success")) {
      button.setAttribute("aria-label", messages.copiedLabel);
      status.textContent = "";
      window.setTimeout(() => {
        status.textContent = messages.copiedStatus;
      }, 0);
      return;
    }
    button.setAttribute("aria-label", messages.copyCode);
  };

  const enhanceButton = (button) => {
    if (enhancedButtons.has(button)) return;
    enhancedButtons.add(button);
    syncButton(button);
  };

  document.querySelectorAll("button.copybtn").forEach(enhanceButton);
  new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === "attributes" && mutation.target.matches("button.copybtn")) {
        syncButton(mutation.target);
        return;
      }
      mutation.addedNodes.forEach((node) => {
        if (!(node instanceof Element)) return;
        if (node.matches("button.copybtn")) enhanceButton(node);
        node.querySelectorAll?.("button.copybtn").forEach(enhanceButton);
      });
    });
  }).observe(document.body, {
    attributes: true,
    attributeFilter: ["class"],
    childList: true,
    subtree: true,
  });
};

const setupVersionSwitchers = () => {
  document.querySelectorAll(".cr-version-switch").forEach((root, index) => {
    const button = root.querySelector(".cr-version-button");
    const menu = root.querySelector(".cr-version-menu");
    if (!button || !menu) return;

    const menuId = `cr-version-menu-${index + 1}`;
    menu.id = menuId;
    button.setAttribute("aria-controls", menuId);

    const setExpanded = (expanded) => {
      root.dataset.open = String(expanded);
      menu.hidden = !expanded;
      button.setAttribute("aria-expanded", String(expanded));
    };

    setExpanded(false);

    button.addEventListener("click", () => {
      setExpanded(menu.hidden);
    });

    root.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || menu.hidden) return;
      event.preventDefault();
      setExpanded(false);
      button.focus();
    });

    document.addEventListener("pointerdown", (event) => {
      if (!root.contains(event.target)) setExpanded(false);
    });
  });
};

const setupDocumentationInteractions = () => {
  setupHeaderSearch();
  setupCopyButtons();
  setupVersionSwitchers();
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", setupDocumentationInteractions, { once: true });
} else {
  setupDocumentationInteractions();
}
