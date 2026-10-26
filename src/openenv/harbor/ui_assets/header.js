// Header: a Docs link in the tab bar, and tabs that go back to their list when clicked again.
//
// The tab bar is Gradio's, so the link is placed into it from here and put back if Gradio redraws
// the bar. Clicking Tasks while a task is open, or Runs while a run is open, fires `select` with
// `{back: "tasks"}` or `{back: "runs"}`; Python clears that tab's open item, which shows its list.

const DOCS = (element.querySelector("[data-docs]") || {}).dataset?.docs;
const TITLE = element.querySelector(".hb-brand b")?.textContent;
if (TITLE) document.title = TITLE;   // OpenEnv's page title names the env class

// Inside the tab list, after the last tab: Gradio moves tabs that do not fit into a "more" menu by
// measuring that list, so it must keep its own width, and a link beside it would be pushed away.
function placeDocs() {
  const list = document.querySelector('.hb-tabs [role="tablist"]');
  if (!list || !DOCS) return;
  const last = [...list.querySelectorAll('[role="tab"]')].pop();
  const have = list.querySelector(".hb-docs");
  if (have && (!last || have.previousElementSibling === last)) return;
  if (have) have.remove();
  const a = document.createElement("a");
  a.className = "hb-docs";
  a.href = DOCS;
  a.target = "_blank";
  a.rel = "noopener";
  a.innerHTML = `Docs${icon("external", 13)}`;
  last ? last.after(a) : list.appendChild(a);
}
// Watched until the tab bar exists, then only the tab bar, so a live run redrawing every two seconds
// does not run this on each change.
const wide = new MutationObserver(() => {
  const bar = document.querySelector(".hb-tabs .tab-wrapper");
  if (!bar) return;
  placeDocs();
  wide.disconnect();
  new MutationObserver(placeDocs).observe(bar, { childList: true, subtree: true });
});
placeDocs();
wide.observe(document.querySelector(".gradio-container") || document.body, { childList: true, subtree: true });

// Capture phase, so the tab's state is read before Gradio switches it.
document.addEventListener("click", (ev) => {
  const tab = ev.target.closest('.hb-tabs [role="tab"]');
  if (!tab || tab.getAttribute("aria-selected") !== "true") return;
  const name = tab.textContent.trim();
  if (name === "Tasks" && document.querySelector(".tp-head")) {
    try { history.replaceState(null, "", location.pathname + location.search); } catch (_) { /* sandboxed frame */ }
    trigger("select", { back: "tasks" });
  }
  if (name === "Runs" && document.querySelector(".rp")) trigger("select", { back: "runs" });
}, true);
