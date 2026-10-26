// Task page head: back to the task list, and a link to this task.
// The breadcrumb fires `select` with `{back: true}`; Python swaps the list back in.

element.addEventListener("click", (ev) => {
  if (ev.target.closest("[data-back]")) {
    try { history.replaceState(null, "", location.pathname + location.search); } catch (_) { /* sandboxed frame */ }
    return trigger("select", { back: true });
  }
  const copy = ev.target.closest("[data-copy]");
  if (copy && navigator.clipboard) {
    navigator.clipboard.writeText(link() || location.href).then(() => {
      copy.innerHTML = `${icon("check", 14)}Copied`;
      setTimeout(() => { copy.innerHTML = `${icon("link", 14)}Copy link`; }, 1600);
    });
  }
});

// This task's own address, from the head itself: a task opened from a run page or the run list
// never went through the task list, which is what sets the hash.
function link() {
  const head = element.querySelector(".tp-head[data-spec]");
  return head ? `${location.origin}${location.pathname}${location.search}#task=${encodeURIComponent(head.dataset.spec)}:${head.dataset.index}` : "";
}

// A newly opened task starts at its top, wherever the list was scrolled to, and the address bar
// follows it.
watch("value", () => {
  const top = element.getBoundingClientRect().top;
  if (top < 0) window.scrollBy({ top: top - 12 });
  const to = link();
  if (to && to !== location.href) { try { history.replaceState(null, "", to); } catch (_) { /* sandboxed frame */ } }
});
