// Run page: its links, "Expand all", the downloads, and keeping what you opened open.
//
// While a rollout runs, the page is re-rendered every two seconds from Python. Without this, a row
// you unfolded would snap shut on the next refresh. So every toggle is remembered by the fold's
// position in the page (events only ever append, so positions are stable while a run grows) and
// re-applied after each render. Another run starts from its own defaults, at the top.
// Links fire `select`: `{back}` to the list, `{run}` for another run, `{task, index}` for its task.

const opened = new Map();
let run = null;
const folds = () => [...element.querySelectorAll(".rp-main details")];

// `toggle` fires after the change, including after the restore below sets `open`; recording those
// again is harmless, since it records the same value.
element.addEventListener("toggle", (ev) => {
  if (ev.target.tagName !== "DETAILS") return;
  const i = folds().indexOf(ev.target);
  if (i >= 0) opened.set(i, ev.target.open);
}, true);

async function download(button) {
  const label = button.innerHTML;
  button.disabled = true;
  button.innerHTML = `<span class="hb-spinner"></span>${esc(button.textContent)}`;
  try {
    const res = await server.hb_download([button.dataset.grant, button.dataset.dl]);
    if (res.error) throw new Error(res.error);
    const url = URL.createObjectURL(new Blob([res.text], { type: "application/json" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: res.name });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    button.innerHTML = label;
  } catch (e) {
    button.innerHTML = `${icon("alert", 14)}${esc(String(e.message || e).slice(0, 60))}`;
    setTimeout(() => { button.innerHTML = label; }, 3000);
  }
  button.disabled = false;
}

element.addEventListener("click", (ev) => {
  if (ev.target.closest("[data-back]")) return trigger("select", { back: true });
  const task = ev.target.closest("[data-task]");
  if (task) return trigger("select", { task: task.dataset.task, index: Number(task.dataset.index) });
  const other = ev.target.closest("[data-run]:not(.rp)");
  if (other) return trigger("select", { run: other.dataset.run });
  const dl = ev.target.closest("[data-dl]");
  if (dl) return download(dl);
  const b = ev.target.closest("[data-expand]");
  if (!b) return;
  const open = b.textContent.trim() === "Expand all";
  folds().forEach((d, i) => { d.open = open; opened.set(i, open); });
  b.textContent = open ? "Collapse all" : "Expand all";
});

function restore() {
  const root = element.querySelector("[data-run]");
  const id = root ? root.dataset.run : null;
  if (id !== run) {
    run = id; opened.clear();
    const top = element.getBoundingClientRect().top;
    if (top < 0) window.scrollBy({ top: top - 12 });
    return;
  }
  folds().forEach((d, i) => { if (opened.has(i)) d.open = opened.get(i); });
  const b = element.querySelector("[data-expand]");
  if (b && opened.size && [...opened.values()].every(Boolean)) b.textContent = "Collapse all";
}
watch("value", restore);
restore();
