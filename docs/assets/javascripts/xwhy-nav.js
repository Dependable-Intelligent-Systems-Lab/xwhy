(function () {
  const explainerLinks = [
    ["All explainers", "explainers/"],
    ["Image Classification", "explainers/image/"],
    ["Image Generation", "explainers/image-generation/"],
    ["Image Editing", "explainers/image-generation/image-editing/"],
    ["Pix2Pix Models", "explainers/image-generation/pix2pix-models/"],
    ["LLM", "explainers/llm/"],
    ["Tabular", "explainers/tabular/"],
    ["Text", "explainers/text/"],
    ["Point Cloud", "explainers/point-cloud/"],
    ["Time Series", "explainers/time-series/"],
    ["Multimodal", "explainers/multimodal/"]
  ];

  const tutorialLinks = [
    ["All tutorials & examples", "tutorials-and-examples/"],
    ["Image Classification Tutorial", "image_classification_explainer/"],
    ["LLM Example", "llm_explainer/"],
    ["Image Examples", "examples/image/"],
    ["Tabular Examples", "examples/tabular/"],
    ["Text Examples", "examples/text/"],
    ["Point-Cloud Examples", "examples/point-cloud/"]
  ];

  const howToLinks = [
    ["How-to overview", "how-to/"],
    ["Configure logging", "how-to/logging/"],
    ["Connect a custom model", "how-to/custom-models/"],
    ["Configure an LLM provider", "how-to/providers/"],
    ["Make explanations reproducible", "how-to/reproducibility/"]
  ];

  const evaluationLinks = [
    ["Evaluation overview", "evaluation/"],
    ["ATT Fidelity", "evaluation/attribution-fidelity/"],
    ["ATT Accuracy", "evaluation/attribution-accuracy/"],
    ["ATT Stability", "evaluation/attribution-stability/"],
    ["ATT Consistency", "evaluation/attribution-consistency/"],
    ["ATT Faithfulness", "evaluation/attribution-faithfulness/"]
  ];

  const researchLinks = [
    ["Research overview", "research/"],
    ["Publications", "research/publications/"],
    ["Citation guidance", "research/citation/"]
  ];

  const navLinks = [
    ["Home", ""],
    ["Get Started", "getting-started/"],
    ["Explainers", "explainers/", explainerLinks],
    ["Tutorials & Examples", "tutorials-and-examples/", tutorialLinks],
    ["How-to Guides", "how-to/", howToLinks],
    ["Evaluation", "evaluation/", evaluationLinks],
    ["Research", "research/", researchLinks],
    ["Contributors", "contributors/"]
  ];

  const desktopNavigation = window.matchMedia("(min-width: 76.25em)");

  function siteRoot() {
    const marker = "/xwhy/";
    const path = window.location.pathname;
    const markerIndex = path.indexOf(marker);

    if (markerIndex >= 0) {
      return path.slice(0, markerIndex) + marker;
    }

    return "/";
  }

  function sitePath(relativePath) {
    return siteRoot() + relativePath.replace(/^\/+/, "");
  }

  function normalisePath(path) {
    if (!path.endsWith("/")) {
      return path + "/";
    }
    return path;
  }

  function currentPath() {
    return normalisePath(window.location.pathname);
  }

  function isExact(href) {
    return currentPath() === normalisePath(href);
  }

  function isActive(href) {
    const path = currentPath();
    const target = normalisePath(href);

    if (target === normalisePath(siteRoot())) {
      return path === target;
    }

    return path === target || path.startsWith(target);
  }

  function createLink(label, relativePath, className) {
    const anchor = document.createElement("a");
    const href = sitePath(relativePath);

    anchor.className = className;
    anchor.href = href;
    anchor.textContent = label;

    if (isActive(href)) {
      anchor.classList.add(className + "--active");
    }

    if (isExact(href)) {
      anchor.setAttribute("aria-current", "page");
    }

    return anchor;
  }

  function updateActiveState(nav) {
    nav.querySelectorAll("a[href]").forEach(function (anchor) {
      const href = new URL(anchor.href).pathname;
      const active = isActive(href);
      const exact = isExact(href);

      if (anchor.classList.contains("xwhy-topnav__link")) {
        anchor.classList.toggle("xwhy-topnav__link--active", active);
      }

      if (anchor.classList.contains("xwhy-topnav__dropdown-link")) {
        anchor.classList.toggle("xwhy-topnav__dropdown-link--active", active);
      }

      if (exact) {
        anchor.setAttribute("aria-current", "page");
      } else {
        anchor.removeAttribute("aria-current");
      }
    });
  }

  function mountTopNav() {
    const existing = document.querySelector("[data-xwhy-topnav]");

    if (!desktopNavigation.matches) {
      if (existing) {
        existing.remove();
      }
      return;
    }

    if (existing) {
      updateActiveState(existing);
      return;
    }

    const header = document.querySelector(".md-header");
    if (!header || !header.parentElement) {
      return;
    }

    const nav = document.createElement("nav");
    nav.className = "xwhy-topnav";
    nav.setAttribute("data-xwhy-topnav", "true");
    nav.setAttribute("aria-label", "Primary navigation");

    const inner = document.createElement("div");
    inner.className = "xwhy-topnav__inner";

    const list = document.createElement("ul");
    list.className = "xwhy-topnav__list";

    navLinks.forEach(function (item) {
      const label = item[0];
      const relativePath = item[1];
      const children = item[2];
      const listItem = document.createElement("li");
      listItem.className = "xwhy-topnav__item";

      const link = createLink(label, relativePath, "xwhy-topnav__link");
      listItem.appendChild(link);

      if (children) {
        listItem.classList.add("xwhy-topnav__item--has-menu");
        link.classList.add("xwhy-topnav__link--menu");
        link.setAttribute("aria-haspopup", "true");

        const submenu = document.createElement("ul");
        submenu.className = "xwhy-topnav__dropdown";
        submenu.setAttribute("aria-label", label + " pages");

        children.forEach(function (child) {
          const childItem = document.createElement("li");
          childItem.className = "xwhy-topnav__dropdown-item";
          childItem.appendChild(
            createLink(child[0], child[1], "xwhy-topnav__dropdown-link")
          );
          submenu.appendChild(childItem);
        });

        listItem.appendChild(submenu);
      }

      list.appendChild(listItem);
    });

    inner.appendChild(list);
    nav.appendChild(inner);
    updateActiveState(nav);
    header.insertAdjacentElement("afterend", nav);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(mountTopNav);
  }

  if (typeof desktopNavigation.addEventListener === "function") {
    desktopNavigation.addEventListener("change", mountTopNav);
  } else {
    desktopNavigation.addListener(mountTopNav);
  }

  document.addEventListener("DOMContentLoaded", mountTopNav);
})();
