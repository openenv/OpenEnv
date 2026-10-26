// Run list: open a run, filter by status or text, tick two to four to compare.
//
// The table is rendered in Python and re-rendered when a run changes state, so the filters live here
// and are re-applied after every render. Ticked runs go to Python on each change (`input`), which
// renders them ticked again; Compare sends them with `submit`; opening a row sends `select`.

const st = { q: "", s: "" };

function apply() {
  const q = st.q.toLowerCase();
  let n = 0;
  element.querySelectorAll(".rs-row[data-rs-id]").forEach((row) => {
    const hide = (st.s && row.dataset.s !== st.s) || (q && !row.textContent.toLowerCase().includes(q));
    row.hidden = !!hide;
    n += hide ? 0 : 1;
  });
  element.querySelectorAll("[data-rs-seg] button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.s === st.s)));
  const box = element.querySelector(".rs-q");
  if (box && box.value !== st.q) box.value = st.q;
  const ticked = [...element.querySelectorAll("[data-rs-pick]:checked")].map((c) => c.dataset.rsPick);
  const btn = element.querySelector("[data-rs-compare]");
  if (btn) {
    btn.disabled = ticked.length < 2;
    btn.innerHTML = `${icon("columns", 14)}${ticked.length >= 2 ? `Compare ${ticked.length}` : "Compare"}`;
  }
  const hint = element.querySelector(".rs-n");
  if (hint) hint.textContent = ticked.length ? `${ticked.length} ticked` : (st.q || st.s ? `${n} shown` : "Tick two to four to compare");
  element.querySelectorAll("[data-rs-pick]").forEach((c) => { c.disabled = !c.checked && ticked.length >= 4; });
}

element.addEventListener("click", (ev) => {
  if (ev.target.closest("[data-rs-pick]")) return;             // ticking is not opening
  const seg = ev.target.closest("[data-rs-seg] button");
  if (seg) { st.s = seg.dataset.s; return apply(); }
  if (ev.target.closest("[data-rs-compare]")) {
    const ids = [...element.querySelectorAll("[data-rs-pick]:checked")].map((c) => c.dataset.rsPick);
    if (ids.length >= 2) trigger("submit", { ids });
    return;
  }
  const row = ev.target.closest(".rs-row[data-rs-id]");
  if (row) trigger("select", { id: row.dataset.rsId });
});
element.addEventListener("change", (ev) => {
  if (!ev.target.closest("[data-rs-pick]")) return;
  apply();
  trigger("input", { ids: [...element.querySelectorAll("[data-rs-pick]:checked")].map((c) => c.dataset.rsPick) });
});
element.addEventListener("input", (ev) => {
  if (!ev.target.matches(".rs-q")) return;
  st.q = ev.target.value.trim();
  apply();
});
watch("value", apply);
apply();
