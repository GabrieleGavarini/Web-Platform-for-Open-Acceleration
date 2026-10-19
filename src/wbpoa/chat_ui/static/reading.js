/* A node's page has one scrollbar: while its citations pane stands beside the text, the pane is
   pinned (data-synced) and follows the page's scroll in proportion, so text and citations reach
   their ends together. The text pane is kept at least as tall as the citations, so a short
   provision still lets every citation scroll into view. Below the two-column width, or without
   this script, the pane simply flows with the page. */
(() => {
  const PANE = ".wbpoa-reading--app > .wbpoa-reading-side";
  const STACKED = matchMedia("(max-width: 1279px)");
  let frame = 0;

  const synced = () => {
    const side = document.querySelector(PANE);
    if (!side) return null;
    const main = side.previousElementSibling;
    if (STACKED.matches) {
      delete side.dataset.synced;
      main.style.minHeight = "";
      return null;
    }
    side.dataset.synced = "";
    main.style.minHeight = `${side.scrollHeight}px`;
    const page = document.documentElement;
    return { side, range: page.scrollHeight - page.clientHeight, extra: side.scrollHeight - side.clientHeight };
  };

  const sync = () => {
    frame = 0;
    const state = synced();
    if (!state) return;
    const { side, range, extra } = state;
    side.scrollTop = range > 0 ? (document.documentElement.scrollTop / range) * extra : 0;
  };

  const request = () => {
    if (!frame) frame = requestAnimationFrame(sync);
  };

  addEventListener("scroll", request, { passive: true });
  addEventListener("resize", request);
  STACKED.addEventListener("change", request);
  new MutationObserver(request).observe(document.body, { childList: true, subtree: true });
})();

/* The citation graph's corpus panel, drawn once per corpus while its nodes open beside it. The
   node being read is marked (aria-current). Where the panel stands beside the page, the folds
   holding the node are opened, the panel is scrolled to it when it is out of view, and the folds
   a reader opens stay open, per corpus, for the browser tab (storage that is blocked only loses
   the remembering). On a narrow window the panel stands above the page, so it starts folded and
   a node's page is scrolled past it to the text. */
(() => {
  const PANEL = ".wbpoa-graph-aside";
  const NARROW = matchMedia("(max-width: 960px)");
  const SETTLE_MS = 200;

  const read = (key) => {
    try {
      return JSON.parse(sessionStorage.getItem(key) || "[]");
    } catch {
      return [];
    }
  };

  const write = (key, opened) => {
    try {
      sessionStorage.setItem(key, JSON.stringify(opened));
    } catch {
      /* Storage is unavailable: nothing is remembered. */
    }
  };

  const prepare = (contents) => {
    contents.dataset.ready = "";
    const key = `wbpoa-contents:${contents.dataset.corpus}`;
    const folds = [...contents.querySelectorAll("details[data-key]")];
    const remembered = NARROW.matches ? [] : read(key);
    for (const fold of folds) {
      if (remembered.includes(fold.dataset.key)) fold.open = true;
    }
    contents.addEventListener("click", (event) => {
      if (!event.target.closest("summary")) return;
      /* The toggle lands after the click: record the folds' state once it has. */
      setTimeout(() => write(key, folds.filter((fold) => fold.open).map((fold) => fold.dataset.key)));
    });
  };

  const mark = (contents) => {
    if (contents.dataset.marked === location.pathname) return;
    contents.dataset.marked = location.pathname;
    contents.querySelector('[aria-current="page"]')?.removeAttribute("aria-current");
    const current = [...contents.querySelectorAll("a[href]")].find((link) => link.pathname === location.pathname);
    if (!current) return;
    current.setAttribute("aria-current", "page");
    if (NARROW.matches) {
      /* After the router's own scroll to the top, bring the node's text, under the panel, into view. */
      setTimeout(() => document.querySelector(".wbpoa-graph-main")?.scrollIntoView({ block: "start" }), SETTLE_MS);
      return;
    }
    for (let fold = current.closest("details"); fold; fold = fold.parentElement.closest("details")) fold.open = true;
    const panel = contents.closest(PANEL);
    if (!panel || panel.scrollHeight <= panel.clientHeight) return;
    const box = current.getBoundingClientRect();
    const view = panel.getBoundingClientRect();
    if (box.top < view.top || box.bottom > view.bottom) panel.scrollTop += box.top - view.top - panel.clientHeight / 2;
  };

  const scan = () => {
    document.querySelectorAll(`${PANEL} .wbpoa-contents:not([data-ready])`).forEach(prepare);
    document.querySelectorAll(`${PANEL} .wbpoa-contents`).forEach(mark);
  };
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  addEventListener("popstate", scan);
  scan();
})();
