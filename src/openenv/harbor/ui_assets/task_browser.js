// Task browser: every served dataset's tasks, searched and filtered as you type.
//
// Rendered here rather than in Python because a server can hold thousands of tasks, and filtering them
// on every keystroke must not cost a round trip. Rows come from `server.hb_tasks(spec)` once per
// dataset. Opening a task fires `select` with `{spec, index}`, and the Python side shows the task page.
// `props.value` is never written: that would re-render the template and wipe the list.
// Adding a Hub dataset runs on the server in the background (`hb_add`), and the add panel polls it
// (`hb_add_status`) to show the download as it happens. `esc` and `icon` come from ui_icons.py.

const PAGE = 60;
const FACETS = [["spec", "Dataset"], ["category", "Category"], ["difficulty", "Difficulty"], ["tags", "Tags"]];
const SHOW = 8;
const st = { datasets: [], rows: [], loading: 0, q: "", sel: {}, more: {}, shown: PAGE, canAdd: false, common: {},
  add: { open: false, q: "", hits: null, error: "", timer: null, counts: {}, jobs: {}, poll: null },
  rm: { confirm: null, busy: null, error: "" } };
const $ = (sel) => element.querySelector(sel);
const fmt = (n) => Number(n || 0).toLocaleString();
// A dataset on the Space's bucket is a folder like `/data/org__name`; it is shown by its Hub id.
const labels = {};
const nameOf = (spec) => labels[spec] || spec;
const short = (spec) => String(nameOf(spec)).split("/").pop();

// Tags every task of a dataset carries (`terminal` on a terminal dataset) tell no two tasks apart,
// so they are left off the cards and out of the tag filter.
function tagsOf(r) { return (r.keywords || []).filter((k) => !(st.common[r.spec] || new Set()).has(k)); }
function values(r, key) {
  if (key === "tags") return tagsOf(r);
  const v = r[key];
  return v ? [v] : [];
}

function matches(r, skip) {
  for (const [key] of FACETS) {
    if (key === skip) continue;
    const want = st.sel[key];
    if (want && want.size && !values(r, key).some((v) => want.has(v))) return false;
  }
  if (!st.q) return true;
  return st.q.split(/\s+/).every((t) => r._hay.includes(t));
}

// Split the raw text on the search terms and escape each piece, so a match can never land inside an
// escaped entity (searching "39" in "Don't" must not break `&#39;`).
function mark(text) {
  const raw = String(text ?? "");
  const terms = st.q.split(/\s+/).filter((t) => t.length > 1).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!terms.length) return esc(raw);
  return raw.split(new RegExp(`(${terms.join("|")})`, "gi")).map((part, i) => (i % 2 ? `<mark>${esc(part)}</mark>` : esc(part))).join("");
}

function renderStats() {
  const ok = st.datasets.filter((d) => !d.error);
  $(".tb-stats").innerHTML = `<div><b>${fmt(ok.length)}</b><span>dataset${ok.length === 1 ? "" : "s"}</span></div>`
    + `<div><b>${fmt(st.rows.length)}</b><span>tasks${st.loading ? " so far" : ""}</span></div>`;
}

function renderFilters() {
  const parts = [`<div class="tb-filters-h"><span>Filters</span>${Object.values(st.sel).some((s) => s.size) ? '<button class="hb-link" type="button" data-clear>Clear all</button>' : ""}</div>`];
  for (const [key, label] of FACETS) {
    const counts = new Map();
    for (const r of st.rows) if (matches(r, key)) for (const v of values(r, key)) counts.set(v, (counts.get(v) || 0) + 1);
    const all = new Map();
    for (const r of st.rows) for (const v of values(r, key)) all.set(v, (all.get(v) || 0) + 1);
    if (key === "spec") for (const d of st.datasets) if (!all.has(d.spec)) all.set(d.spec, 0);
    if (all.size < (key === "spec" ? 1 : 2) && !(key === "spec" && st.canAdd)) continue;
    const chosen = st.sel[key] || new Set();
    const opts = [...all.keys()].sort((a, b) => (counts.get(b) || 0) - (counts.get(a) || 0) || String(a).localeCompare(String(b)));
    const limit = st.more[key] ? opts.length : SHOW;
    const rows = opts.slice(0, limit).map((v) => {
      const d = key === "spec" ? st.datasets.find((x) => x.spec === v) : null;
      const n = d && d.error ? "failed" : d && d.loading ? '<span class="hb-spinner"></span>' : fmt(counts.get(v) || 0);
      const opt = `<button class="tb-opt${counts.get(v) ? "" : " zero"}" type="button" data-f="${key}" data-v="${esc(v)}" aria-pressed="${chosen.has(v)}"
        title="${esc(d && d.error ? d.error : nameOf(v))}"><span class="box">${icon("check", 11)}</span><span class="lab">${esc(key === "spec" ? short(v) : v)}</span><span class="c">${n}</span></button>`;
      if (!(d && d.removable)) return opt;
      if (st.rm.busy === v) return `<div class="tb-rmrow"><span class="hb-spinner"></span><span class="lab">Removing ${esc(short(v))}…</span></div>`;
      if (st.rm.confirm === v) {
        const where = v.startsWith("/") ? "its copy in this Space's bucket" : "its download on this server";
        return `<div class="tb-confirm"><p>Remove <b>${esc(nameOf(v))}</b>? This deletes ${where}. Runs of its tasks stay in history.</p>
          ${st.rm.error ? `<p class="err">${esc(st.rm.error)}</p>` : ""}
          <div><button type="button" class="hb-btn sm" data-rm-yes="${esc(v)}">Remove</button><button type="button" class="hb-btn sm ghost" data-rm-no>Cancel</button></div></div>`;
      }
      return `<div class="tb-optrow">${opt}<button type="button" class="tb-rm" data-rm="${esc(v)}" title="Remove this dataset" aria-label="Remove ${esc(nameOf(v))}">${icon("x", 13)}</button></div>`;
    }).join("");
    const more = opts.length > SHOW ? `<button class="hb-link" type="button" data-more="${key}">${st.more[key] ? "Show fewer" : `Show ${opts.length - SHOW} more`}</button>` : "";
    const adding = key === "spec" ? Object.values(st.add.jobs).filter((j) => !["done", "error"].includes(j.state)) : [];
    const pending = adding.map((j) => `<div class="tb-opt zero" title="${esc(j.spec)}"><span class="hb-spinner"></span><span class="lab">${esc(short(j.spec))}</span><span class="c">${esc(jobWord(j))}</span></div>`).join("");
    const hub = key === "spec" && st.canAdd ? `${pending}<button class="hb-link" type="button" data-add-open>${icon("plus", 13)} Add from the Hub</button>` : "";
    parts.push(`<details class="tb-facet" data-facet="${key}" open><summary><h3>${label}</h3>${chosen.size ? `<span class="sel">${chosen.size}</span>` : ""}${icon("chevronRight", 14, "chev")}</summary>${rows}${more}${hub}</details>`);
  }
  const box = $(".tb-filters");
  // Which facets are folded survives re-renders. On a phone they start folded, so the tasks, not the
  // filters, fill the first screen.
  if (!st.folded) st.folded = new Set(matchMedia("(max-width: 900px)").matches ? FACETS.map(([k]) => k) : []);
  box.querySelectorAll(".tb-facet").forEach((d) => { d.open ? st.folded.delete(d.dataset.facet) : st.folded.add(d.dataset.facet); });
  box.innerHTML = parts.join("");
  box.querySelectorAll(".tb-facet").forEach((d) => { if (st.folded.has(d.dataset.facet)) d.open = false; });
}

function card(r) {
  const tags = tagsOf(r).slice(0, 4);
  const bits = [r.category, r.difficulty].filter(Boolean).map(esc);
  return `<li><button class="tb-card" type="button" data-spec="${esc(r.spec)}" data-i="${r.index}">
    <span class="r1"><span class="ds" title="${esc(r.spec)}">${esc(short(r.spec))}</span>${bits.map((b) => `<span class="sep">/</span><span class="cat">${b}</span>`).join("")}<span class="idx">#${r.index}</span></span>
    <span class="t">${mark(r.title)}</span>
    ${r.brief ? `<span class="s">${mark(r.brief)}</span>` : ""}
    ${tags.length ? `<span class="hb-chips">${tags.map((t) => `<span class="hb-chip">${esc(t)}</span>`).join("")}</span>` : ""}
  </button></li>`;
}

function renderList() {
  const hits = st.rows.filter((r) => matches(r));
  $(".tb-list").innerHTML = hits.slice(0, st.shown).map(card).join("");
  if (!hits.length) {
    $(".tb-list").innerHTML = st.loading && !st.rows.length ? "" : `<li style="grid-column:1/-1"><div class="hb-empty">${icon("search", 20)}<h3>No tasks match</h3><p>Try fewer words, or clear the filters.</p></div></li>`;
  }
  if (st.loading && !st.rows.length) $(".tb-list").innerHTML = Array.from({ length: 6 }, () => '<li><span class="hb-sk" style="height:112px;--hb-radius:14px"></span></li>').join("");
  $(".count").textContent = st.loading && !st.rows.length ? "Loading tasks…" : `${fmt(hits.length)} task${hits.length === 1 ? "" : "s"}`;
  const pills = [];
  for (const [key, label] of FACETS) for (const v of st.sel[key] || []) {
    pills.push(`<button class="tb-pill" type="button" data-f="${key}" data-v="${esc(v)}"><span>${label}</span>${esc(key === "spec" ? short(v) : v)}${icon("x", 12)}</button>`);
  }
  if (st.q) pills.push(`<button class="tb-pill" type="button" data-clear-q><span>Search</span>${esc(st.q)}${icon("x", 12)}</button>`);
  $(".tb-active").innerHTML = pills.join("");
  const more = $(".tb-more");
  more.hidden = hits.length <= st.shown;
  more.innerHTML = `<span>Showing ${fmt(Math.min(st.shown, hits.length))} of ${fmt(hits.length)}</span><button class="hb-btn sm" type="button" data-page>Show ${fmt(Math.min(PAGE, hits.length - st.shown))} more</button>`;
  $(".tb-random").disabled = !hits.length;
}

function render() { renderStats(); renderFilters(); renderList(); }

function open(spec, index) {
  try { history.replaceState(null, "", `#task=${encodeURIComponent(spec)}:${index}`); } catch (_) { /* sandboxed frame */ }
  trigger("select", { spec, index: Number(index) });
}

function learn(spec, rows) {
  const counts = new Map();
  for (const r of rows) for (const k of r.keywords || []) counts.set(k, (counts.get(k) || 0) + 1);
  st.common[spec] = new Set([...counts].filter(([, n]) => rows.length > 3 && n >= rows.length * 0.9).map(([k]) => k));
  for (const r of rows) {
    r.spec = spec;
    r._hay = `${r.title} ${r.brief || ""} ${r.name} ${r.category} ${r.difficulty} ${(r.keywords || []).join(" ")} ${spec} #${r.index}`.toLowerCase();
  }
  st.rows = st.rows.filter((r) => r.spec !== spec).concat(rows);
}

async function load(d) {
  d.loading = true; st.loading += 1; render();
  try {
    const res = await server.hb_tasks(d.spec);
    if (res.error) throw new Error(res.error);
    d.num_tasks = res.rows.length;
    learn(d.spec, res.rows);
  } catch (e) { d.error = String(e.message || e); }
  d.loading = false; st.loading -= 1;
  render();
}

// ── events ──────────────────────────────────────────────────────────────
element.addEventListener("click", (ev) => {
  const c = ev.target.closest(".tb-card");
  if (c) return open(c.dataset.spec, c.dataset.i);
  const f = ev.target.closest("[data-f]");
  if (f) {
    const set = st.sel[f.dataset.f] || (st.sel[f.dataset.f] = new Set());
    set.has(f.dataset.v) ? set.delete(f.dataset.v) : set.add(f.dataset.v);
    st.shown = PAGE; return render();
  }
  if (ev.target.closest("[data-more]")) { const k = ev.target.closest("[data-more]").dataset.more; st.more[k] = !st.more[k]; return renderFilters(); }
  if (ev.target.closest("[data-clear]")) { st.sel = {}; st.shown = PAGE; return render(); }
  if (ev.target.closest("[data-clear-q]")) { st.q = ""; $(".tb-q").value = ""; return render(); }
  if (ev.target.closest("[data-page]")) { st.shown += PAGE; return renderList(); }
  if (ev.target.closest(".tb-random")) {
    const hits = st.rows.filter((r) => matches(r));
    if (hits.length) { const r = hits[Math.floor(Math.random() * hits.length)]; open(r.spec, r.index); }
    return;
  }
  if (ev.target.closest("[data-add-open]")) { st.add.open = true; renderAdd(); $(".tb-addq")?.focus(); if (!st.add.hits) hubSearch(""); return; }
  if (ev.target.closest("[data-add-close]")) { st.add.open = false; return renderAdd(); }
  const add = ev.target.closest("[data-add]");
  if (add && !add.disabled) return addDataset(add.dataset.add);
  const rm = ev.target.closest("[data-rm]");
  if (rm) { st.rm.confirm = rm.dataset.rm; st.rm.error = ""; return renderFilters(); }
  if (ev.target.closest("[data-rm-no]")) { st.rm.confirm = null; return renderFilters(); }
  const yes = ev.target.closest("[data-rm-yes]");
  if (yes) return removeDataset(yes.dataset.rmYes);
  const show = ev.target.closest("[data-add-show]");
  if (show) { const d = st.datasets.find((x) => x.spec === show.dataset.addShow || x.label === show.dataset.addShow); st.sel = { spec: new Set([d ? d.spec : show.dataset.addShow]) }; st.add.open = false; st.shown = PAGE; render(); return renderAdd(); }
});
let qTimer = null;
element.addEventListener("input", (ev) => {
  if (ev.target.matches(".tb-q")) {
    clearTimeout(qTimer);
    qTimer = setTimeout(() => { st.q = ev.target.value.trim().toLowerCase(); st.shown = PAGE; renderFilters(); renderList(); }, 120);
  }
  if (ev.target.matches(".tb-addq")) { st.add.q = ev.target.value; hubSearch(ev.target.value); }
});
element.addEventListener("keydown", (ev) => {
  if (ev.target.matches(".tb-addq") && ev.key === "Enter" && ev.target.value.includes("/")) addDataset(ev.target.value.trim());
  if (ev.target.matches(".tb-addq") && ev.key === "Escape") { st.add.open = false; renderAdd(); }
});
// "/" searches tasks from anywhere on the list page, as long as nothing else is being typed into.
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "/" || ev.metaKey || ev.ctrlKey || !element.offsetParent) return;
  if (ev.target.closest && ev.target.closest("input, textarea, select, [contenteditable]")) return;
  ev.preventDefault(); $(".tb-q").focus();
});

// ── adding a dataset from the Hub ───────────────────────────────────────
function bytes(n) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${Math.round(n / 1e6)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`;
}

function jobWord(j) {
  if (j.state === "checking") return "checking";
  if (j.state === "copying") return "copying";
  if (j.state === "mounting") return "mounting";
  if (j.state === "downloading") return j.total ? `${Math.floor((100 * j.done) / j.total)}%` : "downloading";
  if (j.state === "indexing") return "reading";
  return j.state;
}

function addRow(h) {
  const spec = h.id;
  const job = st.add.jobs[spec];
  const have = st.datasets.some((d) => d.spec === spec || d.label === spec);
  const info = st.add.counts[spec];
  const n = info === undefined ? undefined : info.tasks;
  const [org, name] = spec.includes("/") ? spec.split("/") : ["", spec];
  const meta = [h.downloads != null ? `${fmt(h.downloads)} downloads` : "", h.rl_environment ? "rl-environment" : ""].filter(Boolean).join(" · ");
  const big = info && info.bytes > 1e9;
  const tasks = n === undefined ? '<span class="faint">…</span>'
    : info.error ? `<span class="faint" title="${esc(info.error)}">couldn't check yet</span>`
    : n === null ? '<span class="faint">not in Harbor\'s tasks/ layout</span>'
    : `${fmt(n)} task${n === 1 ? "" : "s"}${info.bytes ? ` · <span class="${big ? "big" : ""}">${bytes(info.bytes)}</span>` : ""}`;
  let action;
  if (have && (!job || job.state === "done")) {
    action = `<span class="ok">${icon("check", 14)}Added</span><button type="button" class="hb-btn sm" data-add-show="${esc(spec)}">Show tasks</button>`;
  } else if (job && job.state === "error") {
    action = `<span class="err" title="${esc(job.error)}">${icon("alert", 14)}${esc(job.error)}</span><button type="button" class="hb-btn sm" data-add="${esc(spec)}">Try again</button>`;
  } else if (job) {
    const pct = job.state === "downloading" && job.total ? (100 * job.done) / job.total
      : job.state === "indexing" ? 100 : job.state === "mounting" ? 60 : job.state === "copying" ? 30 : 4;
    const words = job.state === "checking" ? "Checking the layout…"
      : job.state === "copying" ? "Copying into the bucket…"
      : job.state === "mounting" ? "Waiting for the bucket mount…"
      : job.state === "downloading" ? (job.total ? `Downloading ${fmt(job.done)} of ${fmt(job.total)} files` : "Starting the download…")
      : "Reading the tasks…";
    action = `<div class="prog"><span class="w"><span class="hb-spinner"></span>${esc(words)}</span><span class="bar"><i style="width:${pct.toFixed(1)}%"></i></span></div>`;
  } else {
    action = `<button type="button" class="hb-btn sm primary" data-add="${esc(spec)}" ${n === null && !info.error ? "disabled" : ""}>${icon("plus", 13)}Add</button>`;
  }
  return `<div class="tb-add-row"><div class="nm"><b title="${esc(spec)}">${org ? `<span>${esc(org)}/</span>` : ""}${esc(name)}</b><em>${esc(meta)}</em></div>
    <span class="tk">${tasks}</span><div class="ac">${action}</div></div>`;
}

function renderAdd() {
  const box = $(".tb-addwrap");
  if (!st.add.open) { box.innerHTML = ""; return; }
  const typed = st.add.q.trim();
  let hits = st.add.hits || [];
  if (typed.includes("/") && !hits.some((h) => h.id === typed)) hits = [{ id: typed }, ...hits];
  const body = st.add.error ? `<div class="hb-note err">${icon("alert", 15)}<span>${esc(st.add.error)}</span></div>`
    : st.add.hits === null ? '<div class="tb-add-empty"><span class="hb-spinner"></span>Searching the Hub…</div>'
    : hits.length ? hits.map(addRow).join("") : '<div class="tb-add-empty">No Harbor datasets match. Type an id like <code>org/name</code> to add one directly.</div>';
  const had = $(".tb-addq");
  const focus = had && document.activeElement === had;
  const list = box.querySelector(".tb-add-list");
  if (had && list) { list.innerHTML = body; return; }   // keep the search box, and its focus, as it is
  box.innerHTML = `<section class="hb-panel tb-add">
    <div class="hb-panel-h"><h3>${icon("plus", 15)}Add a Harbor dataset</h3><span class="aside">public Hub datasets tagged <code>harbor</code></span>
      <button type="button" class="hb-btn sm ghost" data-add-close aria-label="Close">${icon("x", 14)}</button></div>
    <div class="hb-panel-b"><div class="tb-search">${icon("search", 15)}<input class="hb-input tb-addq" type="search" placeholder="Search by name, or type org/name" value="${esc(st.add.q)}" autocomplete="off" spellcheck="false" aria-label="Search Harbor datasets on the Hub"></div>
      <div class="tb-add-list">${body}</div></div></section>`;
  if (focus) $(".tb-addq")?.focus();
}

function hubSearch(q) {
  clearTimeout(st.add.timer);
  st.add.timer = setTimeout(async () => {
    st.add.hits = null; st.add.error = ""; renderAdd();
    try {
      st.add.hits = await server.hb_hub(q.trim());
    } catch (e) { st.add.hits = []; st.add.error = String(e.message || e); }
    renderAdd();
    inspect();
  }, 250);
}

// Task counts for what is listed, one at a time: it is what tells a Harbor dataset from one this
// server cannot load, before anything is downloaded.
async function inspect() {
  const want = [...(st.add.hits || []).map((h) => h.id), ...(st.add.q.includes("/") ? [st.add.q.trim()] : [])];
  for (const spec of want.slice(0, 24)) {
    if (!st.add.open) return;
    if (spec in st.add.counts && !st.add.counts[spec].error) continue;   // a Hub error is asked again
    try { st.add.counts[spec] = await server.hb_inspect(spec); } catch (_) { st.add.counts[spec] = { tasks: null, bytes: null }; }
    renderAdd();
  }
}

async function addDataset(spec) {
  try {
    const job = await server.hb_add(spec);
    st.add.jobs[spec] = job;
    if (job.state === "done") return finished(spec, job);
    renderAdd(); renderFilters();
    follow();
  } catch (e) { st.add.jobs[spec] = { spec, state: "error", error: String(e.message || e) }; renderAdd(); }
}

function follow() {
  if (st.add.poll) return;
  st.add.poll = setInterval(async () => {
    const live = Object.values(st.add.jobs).filter((j) => !["done", "error"].includes(j.state));
    if (!live.length) { clearInterval(st.add.poll); st.add.poll = null; return; }
    for (const j of live) {
      try {
        const now = await server.hb_add_status(j.spec);
        st.add.jobs[j.spec] = now;
        if (now.state === "done") finished(j.spec, now);
      } catch (_) { /* keep polling */ }
    }
    renderAdd(); renderFilters();
  }, 700);
}

async function finished(spec, job) {
  const target = job.target || spec;   // on a Space, the dataset's folder on the bucket mount
  labels[target] = spec;
  let d = st.datasets.find((x) => x.spec === target);
  if (!d) { d = { spec: target, label: spec, num_tasks: job.tasks, added: true, removable: true }; st.datasets.push(d); }
  if (job.tasks != null) st.add.counts[spec] = { ...(st.add.counts[spec] || {}), tasks: job.tasks };
  st.sel = { spec: new Set([target]) };
  st.shown = PAGE;
  renderAdd();
  await load(d);
}

async function removeDataset(spec) {
  st.rm.busy = spec; st.rm.error = ""; renderFilters();
  try {
    const res = await server.hb_remove(spec);
    if (res.error) throw new Error(res.error);
    st.datasets = st.datasets.filter((d) => d.spec !== spec);
    st.rows = st.rows.filter((r) => r.spec !== spec);
    if (st.sel.spec) st.sel.spec.delete(spec);
    delete st.add.jobs[labels[spec] || spec];
    st.rm.confirm = null;
  } catch (e) { st.rm.error = String(e.message || e); st.rm.confirm = spec; }
  st.rm.busy = null;
  render(); renderAdd();
}

// A `#task=<spec>:<index>` link opens that task: on load, and when one is pasted into an open page
// (the page's own URL updates use replaceState, which fires no hashchange).
function openFromHash() {
  const m = /#task=([^:]+):(\d+)/.exec(location.hash || "");
  if (!m) return;
  const want = decodeURIComponent(m[1]);
  if (st.datasets.some((d) => d.spec === want)) trigger("select", { spec: want, index: Number(m[2]) });
}
window.addEventListener("hashchange", openFromHash);

// ── start ───────────────────────────────────────────────────────────────
$(".tb-search-ic").outerHTML = icon("search", 16);
$(".tb-random").innerHTML = `${icon("shuffle", 15)} Random`;
(async () => {
  st.loading += 1; renderList();
  try {
    const info = await server.hb_datasets();
    st.datasets = info.datasets.map((d) => ({ ...d })); st.canAdd = info.can_add;
    for (const d of st.datasets) labels[d.spec] = d.label || d.spec;
    st.loading -= 1;
    openFromHash();
    if (!st.datasets.length) {
      render();
      $(".tb-list").innerHTML = `<li style="grid-column:1/-1"><div class="hb-empty">${icon("database", 20)}<h3>No datasets</h3><p>Start the server with <code>--dataset</code>${st.canAdd ? ", or add one from the Hub under Filters" : ""}.</p></div></li>`;
      return;
    }
    await Promise.all(st.datasets.filter((d) => !d.error).map(load));
  } catch (e) {
    st.loading = 0;
    $(".count").textContent = String(e.message || e);
  }
})();
