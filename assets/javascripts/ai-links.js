(function () {
  function cleanTitle(value) {
    return (value || "XWhy documentation")
      .replace(/\s+#$/, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function pageTitle() {
    var heading = document.querySelector("h1");
    if (heading && heading.textContent) {
      return cleanTitle(heading.textContent);
    }
    return cleanTitle(document.title.replace(/[-\u2014]\s*XWhy.*$/, ""));
  }

  function pageUrl() {
    var url = new URL(window.location.href);
    url.search = "";
    url.hash = "";
    return url.toString();
  }

  function promptText() {
    return [
      'Read this XWhy documentation page about "' + pageTitle() + '".',
      "",
      "Read " + pageUrl(),
      "",
      "Summarize the key points, usage steps, examples, caveats, and best practices."
    ].join("\n");
  }

  function providerUrl(provider) {
    var encoded = encodeURIComponent(promptText());

    if (provider === "openai") {
      return "https://chatgpt.com/?q=" + encoded + "&hints=search";
    }
    if (provider === "anthropic") {
      return "https://claude.ai/new?q=" + encoded;
    }
    if (provider === "google") {
      return "https://aistudio.google.com/prompts/new_chat?prompt=" + encoded;
    }
    if (provider === "kimi") {
      return "https://www.kimi.com/_prefill_chat?force_search=true&prefill_prompt=" + encoded + "&send_immediately=true";
    }
    return pageUrl();
  }

  function createLink(provider, icon, label, linkClass, iconClass) {
    var link = document.createElement("a");
    link.className = linkClass;
    link.href = providerUrl(provider);
    link.target = "_blank";
    link.rel = "noopener noreferrer";

    var iconNode = document.createElement("span");
    iconNode.className = iconClass;
    iconNode.setAttribute("aria-hidden", "true");
    iconNode.textContent = icon;

    var textNode = document.createElement("span");
    textNode.textContent = label;

    link.appendChild(iconNode);
    link.appendChild(textNode);
    return link;
  }

  function appendProviderLinks(parent, linkClass, iconClass) {
    parent.appendChild(createLink("openai", "AI", "Open in ChatGPT", linkClass, iconClass));
    parent.appendChild(createLink("anthropic", "A", "Open in Claude", linkClass, iconClass));
    parent.appendChild(createLink("google", "G", "Open in Gemini", linkClass, iconClass));
    parent.appendChild(createLink("kimi", "K", "Open in Kimi", linkClass, iconClass));
  }

  function injectPanel() {
    var sidebar = document.querySelector(".md-sidebar--primary .md-sidebar__scrollwrap");
    if (!sidebar || sidebar.querySelector(".xwhy-ai-panel")) {
      return;
    }

    var panel = document.createElement("div");
    panel.className = "xwhy-ai-panel";

    var title = document.createElement("p");
    title.className = "xwhy-ai-panel__title";
    title.textContent = "Explore with AI";

    panel.appendChild(title);
    appendProviderLinks(panel, "xwhy-ai-panel__link", "xwhy-ai-panel__icon");

    sidebar.appendChild(panel);
  }

  function injectAskAiButton() {
    if (document.querySelector(".xwhy-ask-ai")) {
      return;
    }

    var wrapper = document.createElement("div");
    wrapper.className = "xwhy-ask-ai";

    var button = document.createElement("button");
    button.type = "button";
    button.className = "xwhy-ask-ai__button";
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-haspopup", "true");

    var label = document.createElement("span");
    label.textContent = "Ask AI";

    var icon = document.createElement("span");
    icon.className = "xwhy-ask-ai__button-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "💬";

    button.appendChild(label);
    button.appendChild(icon);

    var menu = document.createElement("div");
    menu.className = "xwhy-ask-ai__menu";
    menu.hidden = true;
    appendProviderLinks(menu, "xwhy-ask-ai__link", "xwhy-ask-ai__icon");

    function closeMenu() {
      menu.hidden = true;
      button.setAttribute("aria-expanded", "false");
    }

    function toggleMenu() {
      var isHidden = menu.hidden;
      menu.hidden = !isHidden;
      button.setAttribute("aria-expanded", isHidden ? "true" : "false");
    }

    button.addEventListener("click", function (event) {
      event.stopPropagation();
      toggleMenu();
    });

    document.addEventListener("click", function (event) {
      if (!wrapper.contains(event.target)) {
        closeMenu();
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        closeMenu();
      }
    });

    wrapper.appendChild(menu);
    wrapper.appendChild(button);
    document.body.appendChild(wrapper);
  }

  function init() {
    injectPanel();
    injectAskAiButton();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
