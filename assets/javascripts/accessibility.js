(function () {
  function improveAccessibility() {
    const searchDialog = document.querySelector('.md-search[role="dialog"]');

    if (
      searchDialog &&
      !searchDialog.hasAttribute("aria-label") &&
      !searchDialog.hasAttribute("aria-labelledby")
    ) {
      searchDialog.setAttribute(
        "aria-label",
        "Search XWhy: eXplain Why documentation"
      );
    }

    document.querySelectorAll('[role="progressbar"]').forEach(function (bar) {
      if (
        !bar.hasAttribute("aria-label") &&
        !bar.hasAttribute("aria-labelledby")
      ) {
        bar.setAttribute("aria-label", "Page loading progress");
      }
    });

    const askAiButton = document.querySelector(".xwhy-ask-ai__button");
    const askAiMenu = document.querySelector(".xwhy-ask-ai__menu");

    if (askAiButton && askAiMenu) {
      if (!askAiMenu.id) {
        askAiMenu.id = "xwhy-ask-ai-menu";
      }

      askAiButton.setAttribute("aria-controls", askAiMenu.id);
      askAiMenu.setAttribute("aria-label", "Ask an AI assistant about this page");
    }
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(improveAccessibility);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", improveAccessibility);
  } else {
    improveAccessibility();
  }
})();
