(function () {
  const measurementId = "G-DV6YD3JCZ8";
  let analyticsLoaded = false;

  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function () {
    window.dataLayer.push(arguments);
  };

  function loadAnalytics() {
    if (analyticsLoaded) {
      return;
    }

    analyticsLoaded = true;

    const script = document.createElement("script");
    script.async = true;
    script.src =
      "https://www.googletagmanager.com/gtag/js?id=" +
      encodeURIComponent(measurementId);
    document.head.appendChild(script);

    window.gtag("js", new Date());
    window.gtag("config", measurementId, {
      page_path: window.location.pathname
    });

    if (typeof location$ !== "undefined") {
      location$.subscribe(function (url) {
        window.gtag("config", measurementId, {
          page_path: url.pathname
        });
      });
    }
  }

  function scheduleAnalytics() {
    const interactionEvents = ["pointerdown", "keydown", "touchstart"];

    interactionEvents.forEach(function (eventName) {
      window.addEventListener(eventName, loadAnalytics, {
        once: true,
        passive: true
      });
    });

    window.setTimeout(loadAnalytics, 10000);
  }

  if (document.readyState === "complete") {
    scheduleAnalytics();
  } else {
    window.addEventListener("load", scheduleAnalytics, { once: true });
  }
})();
