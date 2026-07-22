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

  function createLink(provider, icon, label) {
    var link = document.createElement("a");
    link.className = "xwhy-ai-panel__link";
    link.href = providerUrl(provider);
    link.target = "_blank";
    link.rel = "noopener noreferrer";

    var iconNode = document.createElement("span");
    iconNode.className = "xwhy-ai-panel__icon";
    iconNode.setAttribute("aria-hidden", "true");
    iconNode.textContent = icon;

    var textNode = document.createElement("span");
    textNode.textContent = label;

    link.appendChild(iconNode);
    link.appendChild(textNode);
    return link;
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
    panel.appendChild(createLink("openai", "AI", "Open in ChatGPT"));
    panel.appendChild(createLink("anthropic", "A", "Open in Claude"));
    panel.appendChild(createLink("google", "G", "Open in Gemini"));
    panel.appendChild(createLink("kimi", "K", "Open in Kimi"));

    sidebar.appendChild(panel);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", injectPanel);
  } else {
    injectPanel();
  }
})();
