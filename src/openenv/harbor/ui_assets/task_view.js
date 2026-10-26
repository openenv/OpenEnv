// Task page: the contents list, the file viewer, and links to this task's runs.
//
// The HTML is rendered in Python and replaced each time a task opens, so every handler is delegated
// from `element`, which persists. File contents are fetched on click with `server.hb_file`, never
// shipped up front: a task can hold thousands of files and a person reads three.

const size = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`);

const files = {};   // path -> text, for Copy

function lines(text) {
  const rows = text.replace(/\n$/, "").split("\n");
  return rows.map((l) => `<span class="l">${esc(l)}</span>`).join("");
}

async function openFile(path) {
  const root = element.querySelector(".tp");
  const view = element.querySelector(".tp-code"), head = element.querySelector(".tp-view-h");
  if (!root || !view) return;
  element.querySelectorAll("[data-file]").forEach((b) => b.setAttribute("aria-current", String(b.dataset.file === path)));
  head.querySelector(".p").textContent = path;
  head.querySelector("em").textContent = "loading…";
  view.dataset.path = path;
  try {
    const f = await server.hb_file([root.dataset.spec, Number(root.dataset.index), path]);
    if (view.dataset.path !== path) return;          // another file was picked meanwhile
    if (f.error) throw new Error(f.error);
    head.querySelector("em").textContent = `${size(f.size || 0)}${f.truncated ? " · first 256 KB" : ""}`;
    files[path] = f.binary ? "" : f.text;
    view.innerHTML = f.binary ? '<div class="msg">Binary file, not shown.</div>' : lines(f.text);
  } catch (e) {
    head.querySelector("em").textContent = "";
    view.innerHTML = `<div class="msg">${esc(e.message || e)}</div>`;
  }
  view.scrollTop = 0; view.scrollLeft = 0;
}

function full(on) {
  const box = element.querySelector(".tp-files");
  if (!box) return;
  const now = on ?? !box.classList.contains("full");
  box.classList.toggle("full", now);
  element.querySelectorAll("[data-full]").forEach((b) => {
    if (b.closest(".tools")) { b.innerHTML = icon(now ? "minimize" : "maximize", 14); b.title = now ? "Close full view (Esc)" : "Full view (Esc to close)"; }
  });
  let scrim = element.querySelector(".hb-scrim");
  if (now && !scrim) { scrim = document.createElement("div"); scrim.className = "hb-scrim"; box.before(scrim); }
  if (!now && scrim) scrim.remove();
}

function go(sec) {
  const target = element.querySelector(`[data-sec="${sec}"]`);
  if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
}

// The contents list marks the last section whose top has scrolled past the top of the window.
function spyOn() {
  const secs = [...element.querySelectorAll("[data-sec]")];
  const links = [...element.querySelectorAll(".tp-toc a")];
  if (!secs.length || !links.length || !element.offsetParent) return;
  let on = secs[0].dataset.sec;
  for (const s of secs) if (s.getBoundingClientRect().top < 140) on = s.dataset.sec;
  links.forEach((a) => a.classList.toggle("on", a.dataset.to === on));
}
let ticking = false;
window.addEventListener("scroll", () => {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(() => { ticking = false; spyOn(); });
}, { passive: true });

function bind() {
  const first = element.querySelector(".tp-code[data-path]");
  if (first && first.dataset.path) openFile(first.dataset.path);
  spyOn();
}

element.addEventListener("click", (ev) => {
  const to = ev.target.closest("[data-to]");
  if (to) { ev.preventDefault(); return go(to.dataset.to); }
  const file = ev.target.closest("[data-file]");
  if (file) return openFile(file.dataset.file);
  const jump = ev.target.closest("[data-file-open]");
  if (jump) { go("files"); return openFile(jump.dataset.fileOpen); }
  const run = ev.target.closest("[data-run]");
  if (run) return trigger("select", { run: run.dataset.run });
  if (ev.target.closest("[data-full]")) return full();
  if (ev.target.closest(".hb-scrim")) return full(false);
  const wrap = ev.target.closest("[data-wrap]");
  if (wrap) {
    const on = wrap.getAttribute("aria-pressed") !== "true";
    wrap.setAttribute("aria-pressed", String(on));
    element.querySelector(".tp-code")?.classList.toggle("wrap", on);
    return;
  }
  const copy = ev.target.closest("[data-copy-file]");
  if (copy && navigator.clipboard) {
    const path = element.querySelector(".tp-code")?.dataset.path;
    navigator.clipboard.writeText(files[path] || "").then(() => {
      copy.innerHTML = icon("check", 14);
      setTimeout(() => { copy.innerHTML = icon("copy", 14); }, 1400);
    });
  }
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && element.querySelector(".tp-files.full")) full(false);
});

watch("value", bind);
bind();
