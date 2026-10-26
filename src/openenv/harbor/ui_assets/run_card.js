// Run card: which model, which agent, which sandbox, and the Run button.
//
// Rendered here from `props.value`, which Python sets whenever the task, the endpoint or the agent
// filter changes. Choices made inside the card live in this closure and travel with the event that
// needs them:
//   submit  {agent, sandbox}                       start a rollout
//   select  {include_experimental}                 re-filter the agents (not `change`, which Gradio
//                                                  also fires on every update from Python)
//   input   {source: "hf", model, route, api_key, local_token}
//           {source: "url", url, model, api_key, api, purpose}   connect a model
//   clear   {}                                     back to the server's endpoint
// A token or key is held in this closure only, sent once with `input`, and forgotten once the
// connection works: the server keeps it in memory for this page, never on disk, never in the page.

const ui = {
  src: null, agent: null, sandbox: null, busy: "", stamp: null, msgAt: "go", quiet: false,
  hf: { model: "", route: "", local: true, token: "" },
  url: { url: "", model: "", api: "openai", purpose: "eval", key: "", models: null, loading: false, error: "" },
  models: null, modelsError: "", pkOpen: false, pkQ: "", toolsOnly: true,
};
const $ = (s) => element.querySelector(s);
const usd = (x) => `$${Number(Number(x).toPrecision(3))}`;
const price = (p) => (p && p.price_in != null ? `${usd(p.price_in)} / ${usd(p.price_out)}` : "");
const ctx = (n) => (!n ? "" : n >= 1e6 ? `${(n / 1e6).toFixed(n % 1e6 ? 1 : 0)}M ctx` : `${Math.round(n / 1000)}k ctx`);
const cheapest = (m) => (m.providers || []).filter((p) => p.price_in != null).sort((a, b) => a.price_in + a.price_out - b.price_in - b.price_out)[0];
const org = (id) => (id.includes("/") ? id.split("/")[0] : "");
const leaf = (id) => (id.includes("/") ? id.split("/").slice(1).join("/") : id);

async function loadModels() {
  if (ui.models || ui.modelsLoading) return;
  ui.modelsLoading = true;
  try {
    const res = await server.hb_models();
    ui.models = res.models || [];
    ui.modelsError = res.error || "";
  } catch (e) { ui.models = []; ui.modelsError = String(e.message || e); }
  ui.modelsLoading = false;
  if (!ui.hf.model && ui.models.length) {
    const v = props.value || {};
    const want = v.server && ui.models.find((m) => m.id === v.server.model);
    ui.hf.model = (want || ui.models.find((m) => m.tools) || ui.models[0]).id;
  }
  render();
}

function pickerRows() {
  const q = ui.pkQ.trim().toLowerCase();
  const list = (ui.models || []).filter((m) => (!ui.toolsOnly || m.tools)
    && (!q || m.id.toLowerCase().includes(q) || (m.providers || []).some((p) => String(p.name || "").toLowerCase().includes(q))));
  if (!list.length) return `<div class="pk-empty">${ui.models ? "No model matches." : "Loading models…"}</div>`;
  return list.slice(0, 120).map((m) => {
    const c = cheapest(m);
    const n = (m.providers || []).length;
    const meta = [`${n} provider${n === 1 ? "" : "s"}`, ctx(m.context), m.tools ? "tools" : "no tool calls"].filter(Boolean).join(" · ");
    return `<button type="button" class="pk-row${m.id === ui.hf.model ? " sel" : ""}" data-model="${esc(m.id)}">
      <span class="pk-name">${esc(leaf(m.id))}${m.id === ui.hf.model ? icon("check", 13, "tick") : ""}</span>
      <span class="pk-meta">${esc(org(m.id))}${org(m.id) ? " · " : ""}${esc(meta)}</span><span class="pk-price">${esc(price(c))}</span></button>`;
  }).join("");
}

function hfBody(v, e) {
  const m = (ui.models || []).find((x) => x.id === ui.hf.model);
  const c = m && cheapest(m);
  const routes = [["", "Auto"], ["fastest", "Fastest"], ["cheapest", "Cheapest"]]
    .concat((m ? m.providers : []).map((p) => [p.name, `${p.name}${p.price_in != null ? ` · ${price(p)}` : ""}${p.context ? ` · ${ctx(p.context)}` : ""}${p.tools ? "" : " · no tools"}`]));
  if (!routes.some(([k]) => k === ui.hf.route)) ui.hf.route = "";
  // Signed in with Hugging Face (where the Space offers it): that account pays, and no token is typed.
  const login = v.hf_login || {};
  const signedIn = !!(login.on && login.user);
  if (signedIn && ui.hf.account === undefined) ui.hf.account = true;
  const useAccount = signedIn && ui.hf.account;
  const useLocal = !useAccount && v.local_token && ui.hf.local;
  const here = encodeURIComponent(location.pathname + location.search + location.hash);
  const connected = e.ok && e.source === "hf" && formMatches(e, v);
  return `<div class="rc-form">
    <div class="pk">
      <button type="button" class="pk-btn" data-pk aria-expanded="${ui.pkOpen}">
        <span class="pk-name">${esc(m ? leaf(m.id) : ui.hf.model ? leaf(ui.hf.model) : "Choose a model")}</span>
        <span class="pk-price">${esc(price(c))}</span>${icon("chevronDown", 15)}
        <span class="pk-org">${esc(m ? `${org(m.id)} · ${m.providers.length} provider${m.providers.length === 1 ? "" : "s"}${m.context ? " · " + ctx(m.context) : ""}` : ui.modelsError || "Hugging Face Inference Providers")}</span>
      </button>
      ${ui.pkOpen ? `<div class="pk-pop"><div class="pk-search">${icon("search", 15)}<input class="pk-q" type="search" placeholder="Search ${ui.models ? ui.models.length : ""} models or providers…" value="${esc(ui.pkQ)}" autocomplete="off">
        <label><input type="checkbox" class="pk-tools" ${ui.toolsOnly ? "checked" : ""}>Tool calling</label></div>
        <div class="pk-list">${pickerRows()}</div>
        <div class="pk-note">Prices are per million input / output tokens, the cheapest provider's. Agents need tool calling.</div></div>` : ""}
    </div>
    <label class="hb-field"><span>Provider <em>who serves it</em></span><select class="hb-input" data-hf="route">${routes.map(([k, l]) => `<option value="${esc(k)}" ${k === ui.hf.route ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></label>
    ${signedIn ? `<div class="rc-acct"><label class="rc-check"><input type="checkbox" data-hf-account ${ui.hf.account ? "checked" : ""}>Use my Hugging Face account</label>
      <span class="faint">@${esc(login.user)} · <a class="hb-link" href="/logout?_target_url=${here}" target="_top">Sign out</a></span></div>` : ""}
    ${v.local_token && !useAccount ? `<label class="rc-check"><input type="checkbox" data-hf-local ${ui.hf.local ? "checked" : ""}>Use this machine's Hugging Face token</label>` : ""}
    ${useLocal || useAccount ? "" : `<label class="hb-field"><span>Access token <em>with Inference Providers permission</em></span>
      <input class="hb-input rc-token" type="password" placeholder="hf_…" autocomplete="off" spellcheck="false"></label>`}
    ${login.on && !signedIn ? `<p class="hb-fine">or <a class="hb-link" href="/login/huggingface?_target_url=${here}" target="_top">sign in with Hugging Face</a> to use your own account</p>` : ""}
    ${connected ? `<div class="rc-conn">${icon("check", 14, "ok")}<b>${esc(e.model)}</b><span>· ${esc(e.level_text)}</span></div>${notes(e)}` : ""}
    <button type="button" class="hb-btn block" data-connect="hf" ${ui.busy === "connect" || !ui.hf.model ? "disabled" : ""}>${ui.busy === "connect" ? '<span class="hb-spinner"></span>Checking…' : connected ? "Reconnect" : "Connect"}</button>
    ${msg("model", v)}
    <p class="hb-fine">Billed to ${useAccount ? "your Hugging Face account" : "the token's account"}. Evaluation only: the router returns no token ids to train on.</p>
  </div>`;
}

function urlBody(v, e) {
  const u = ui.url;
  const connected = e.ok && e.source === "url" && formMatches(e, v);
  const opts = (u.models || []).map((m) => `<option value="${esc(m)}"></option>`).join("");
  return `<div class="rc-form">
    <label class="hb-field"><span>Base URL <em>OpenAI-compatible, like vLLM or SGLang</em></span>
      <input class="hb-input" data-u="url" type="url" placeholder="${v.private_urls ? "http://localhost:8000/v1" : "https://your-endpoint/v1"}" value="${esc(u.url)}" autocomplete="off" spellcheck="false"></label>
    <label class="hb-field"><span>API key <em>if it needs one</em></span>
      <input class="hb-input rc-key" type="password" placeholder="optional" autocomplete="off" spellcheck="false"></label>
    <label class="hb-field"><span>Model <button type="button" class="hb-link" data-load ${u.url ? "" : "disabled"}>${u.loading ? '<span class="hb-spinner"></span>' : icon("refresh", 13)}Load models</button></span>
      <input class="hb-input" data-u="model" list="rc-models" placeholder="read from the endpoint if it serves one" value="${esc(u.model)}" autocomplete="off" spellcheck="false">
      <datalist id="rc-models">${opts}</datalist>${u.error ? `<em class="hb-fine" style="color:var(--err)">${esc(u.error)}</em>` : u.models ? `<em class="hb-fine">${u.models.length} served</em>` : ""}</label>
    <div class="hb-pair">
      <label class="hb-field"><span>API</span><select class="hb-input" data-u="api"><option value="openai" ${u.api === "openai" ? "selected" : ""}>OpenAI-compatible</option><option value="anthropic" ${u.api === "anthropic" ? "selected" : ""}>Anthropic</option></select></label>
      <label class="hb-field"><span>Use</span><select class="hb-input" data-u="purpose"><option value="eval" ${u.purpose === "eval" ? "selected" : ""}>Evaluation</option><option value="train" ${u.purpose === "train" ? "selected" : ""}>Training</option></select></label>
    </div>
    ${connected ? `<div class="rc-conn">${icon("check", 14, "ok")}<b>${esc(e.model)}</b><span>· ${esc(e.host)} · ${esc(e.level_text)}</span></div>${notes(e)}` : ""}
    <button type="button" class="hb-btn block" data-connect="url" ${ui.busy === "connect" || !u.url ? "disabled" : ""}>${ui.busy === "connect" ? '<span class="hb-spinner"></span>Checking…' : connected ? "Test again" : "Test connection"}</button>
    ${msg("model", v)}
    <p class="hb-fine">Training needs exact token ids: vLLM with <code>--return-tokens-as-token-ids</code>. The key is sent only to that endpoint and kept in memory for this page.</p>
  </div>`;
}

// Whether the form still says what was connected. Picking another model or editing the URL after
// connecting means Run would use something the card no longer shows, so that counts as unconnected.
function formMatches(e, v) {
  const c = v.custom || {};
  if (e.source === "hf") return c.model === ui.hf.model && (c.route || "") === (ui.hf.route || "");
  if (e.source === "url") {
    return (c.url || "") === ui.url.url && (c.model || "") === ui.url.model
      && (c.api || "openai") === ui.url.api && (c.purpose || "eval") === ui.url.purpose;
  }
  return true;
}

function notes(e) {
  return (e.notes || []).map((n) => `<p class="hb-fine">${esc(n)}</p>`).join("");
}

// `quiet` hides the last message from the moment an action starts until Python answers it, so an
// old message never shows under the button that is now waiting.
function msg(at, v) {
  return v.message && ui.msgAt === at && !ui.quiet ? `<div class="hb-note ${esc(v.message.tone === "ok" ? "ok" : "err")}">${icon(v.message.tone === "ok" ? "check" : "alert", 15)}<span>${esc(v.message.text)}</span></div>` : "";
}

function render() {
  const v = props.value || {};
  const e = v.engine || {};
  const sources = v.sources || [];
  if (v.stamp !== ui.stamp) {
    // A new value from Python ends any pending action. The form is this card's own: it is never
    // overwritten from Python, so an edit not yet connected survives a re-render.
    ui.stamp = v.stamp; ui.busy = ""; ui.quiet = false;
  }
  // `false` from the server, not merely absent: an empty card (before its first value) is not read-only
  if (v.rollouts === false) {
    element.innerHTML = `<div class="rc hb-panel">
      <div class="hb-panel-h"><h3>${icon("file", 15)}Read-only server</h3></div>
      <div class="hb-panel-b"><div class="hb-note"><span><b>This server is a read-only task browser.</b> You can inspect tasks and prior runs, but it does not accept model credentials or start agents and sandboxes.</span></div></div>
    </div>`;
    return;
  }
  if (!ui.src || !sources.includes(ui.src)) ui.src = e.source && sources.includes(e.source) ? e.source : sources[0] || null;
  const agents = v.agents || [];
  if (!agents.some((a) => a.value === ui.agent)) ui.agent = v.agent || (agents[0] && agents[0].value) || null;
  const boxes = v.sandboxes || [];
  if (!boxes.some((s) => s.name === ui.sandbox && s.available)) ui.sandbox = v.sandbox || (boxes.find((s) => s.available) || {}).name || null;
  const agent = agents.find((a) => a.value === ui.agent);
  const onEngine = e.ok && e.source === ui.src && formMatches(e, v);
  const ready = !!(v.task && onEngine && ui.agent && ui.sandbox && v.rollouts) && !ui.busy;
  if (ui.src === "hf") loadModels();

  const label = { server: "This server", hf: "Hugging Face", url: "Your endpoint" };
  const seg = sources.length > 1 ? `<div class="hb-seg fill" role="group" aria-label="Model source">${sources.map((s) => `<button type="button" data-src="${s}" aria-pressed="${s === ui.src}">${label[s]}</button>`).join("")}</div>` : "";
  let body = "";
  if (ui.src === "server" && v.server) {
    body = `<div class="rc-model"><b>${esc(v.server.model)}</b><em>${v.server.train ? "training" : "eval only"}</em><span>${esc(v.server.host)} · ${esc(v.server.level_text)}</span></div>${msg("model", v)}`;
  } else if (ui.src === "hf") body = hfBody(v, e);
  else if (ui.src === "url") body = urlBody(v, e);
  else body = `<div class="hb-note warn">${icon("alert", 15)}<span>No model is available: this server has no endpoint of its own and does not accept visitors' models.</span></div>`;

  const where = agent ? (agent.host_side ? "runs on this server" : "runs in the sandbox") : "";
  const localWarn = agent && !agent.host_side && v.proxy_local
    ? `<div class="hb-note warn">${icon("alert", 15)}<span>This server's capture proxy is only reachable from this machine, so an agent inside a sandbox cannot call the model. Pick one that runs on this server, like <b>terminus-2</b>, or start the server with <code>--expose gradio</code>.</span></div>` : "";
  const intro = !v.task ? "Open a task to run it."
    : `A fresh <b>${esc(ui.sandbox || "")}</b> sandbox from this task's image, ${ui.agent ? `<b>${esc(ui.agent)}</b> as the agent, ` : ""}then the task's verifier.`;
  const why = !v.rollouts ? "Rollouts are turned off on this server."
    : !onEngine ? (ui.src === "server" ? "" : "Connect the model first.")
    : !ui.agent ? "Pick an agent." : !ui.sandbox ? "No sandbox is available." : "";

  element.innerHTML = `
  <div class="rc hb-panel">
    <div class="hb-panel-h"><h3>${icon("play", 15)}Run a rollout</h3></div>
    <div class="hb-panel-b">
      <p class="rc-intro">${intro}</p>
      <div class="rc-sec"><div class="hb-label">Model</div>${seg}${body}</div>
      <div class="rc-sec"><div class="hb-label">Agent<span>${esc(where)}</span></div>
        ${agents.length ? `<select class="hb-input rc-agent" aria-label="Agent">${agents.map((a) => `<option value="${esc(a.value)}" ${a.value === ui.agent ? "selected" : ""}>${esc(a.label)}</option>`).join("")}</select>`
          : `<div class="hb-note warn">${icon("alert", 15)}<span>${esc(v.agents_empty || "No agent is available for this model.")}</span></div>`}
        <label class="rc-check"><input type="checkbox" class="rc-exp" ${v.include_experimental ? "checked" : ""}>Include experimental agents${v.hidden_agents ? ` <span class="faint">(${v.hidden_agents} more)</span>` : ""}</label>
      </div>
      <div class="rc-sec"><div class="hb-label">Sandbox</div>
        <div class="rc-boxes">${boxes.map((s) => `<button type="button" data-sb="${esc(s.name)}" aria-pressed="${s.name === ui.sandbox}" ${s.available ? "" : "disabled"}
          title="${esc(s.available ? s.name : `${s.name}: ${s.detail || "no credentials"}`)}">${esc(s.name)}</button>`).join("")}</div>
      </div>
      <div class="rc-go">${localWarn}
        <button type="button" class="hb-btn primary lg block rc-run" ${ready ? "" : "disabled"}>${ui.busy === "run" ? '<span class="hb-spinner"></span>Starting…' : `${icon("play", 15)}${e.train && onEngine ? "Run training capture" : "Run rollout"}`}</button>
        ${why ? `<p class="hb-fine" style="text-align:center">${esc(why)}</p>` : ""}
        ${msg("go", v)}
        <p class="hb-fine" style="text-align:center">Model usage is billed to the endpoint or account selected above. Sandbox compute is billed to this server's operator, including Hugging Face Sandbox.</p>
        <p class="hb-fine" style="text-align:center">It keeps running if you close this page, and shows under Runs.</p>
      </div>
    </div>
  </div>`;
  // Held in this closure, never in the markup; restored so a re-render does not drop a half-typed key.
  const tok = $(".rc-token"); if (tok && ui.hf.token) tok.value = ui.hf.token;
  const key = $(".rc-key"); if (key && ui.url.key) key.value = ui.url.key;
  if (ui.pkOpen) { const q = $(".pk-q"); if (q) { q.focus(); q.setSelectionRange(q.value.length, q.value.length); } }
}

function connect(src) {
  ui.busy = "connect"; ui.msgAt = "model"; ui.quiet = true;
  let payload;
  if (src === "hf") {
    const v = props.value || {};
    const login = v.hf_login || {};
    const account = !!(login.on && login.user && ui.hf.account);
    const local = !account && !!(v.local_token && ui.hf.local);
    payload = { source: "hf", model: ui.hf.model, route: ui.hf.route, api_key: account || local ? "" : ui.hf.token, local_token: local, use_account: account };
  } else {
    payload = { source: "url", url: ui.url.url, model: ui.url.model, api_key: ui.url.key, api: ui.url.api, purpose: ui.url.purpose };
  }
  render();
  trigger("input", payload);
}

element.addEventListener("click", async (ev) => {
  const src = ev.target.closest("[data-src]");
  if (src) {
    ui.src = src.dataset.src; ui.pkOpen = false;
    const e = (props.value || {}).engine || {};
    if (ui.src === "server" && e.source !== "server") { ui.busy = "connect"; ui.msgAt = "model"; ui.quiet = true; render(); return trigger("clear", {}); }
    return render();
  }
  if (ev.target.closest("[data-pk]")) { ui.pkOpen = !ui.pkOpen; ui.pkQ = ""; loadModels(); return render(); }
  const row = ev.target.closest("[data-model]");
  if (row) { ui.hf.model = row.dataset.model; ui.hf.route = ""; ui.pkOpen = false; return render(); }
  const sb = ev.target.closest("[data-sb]");
  if (sb && !sb.disabled) { ui.sandbox = sb.dataset.sb; return render(); }
  const c = ev.target.closest("[data-connect]");
  if (c && !c.disabled) return connect(c.dataset.connect);
  if (ev.target.closest("[data-load]") && ui.url.url) {
    ui.url.loading = true; ui.url.error = ""; render();
    try {
      const res = await server.hb_served([ui.url.url, ui.url.key]);
      ui.url.models = res.models || null; ui.url.error = res.error || "";
      if (res.models && res.models.length === 1 && !ui.url.model) ui.url.model = res.models[0];
    } catch (e) { ui.url.error = String(e.message || e); }
    ui.url.loading = false; return render();
  }
  if (ev.target.closest(".rc-run") && !ev.target.closest(".rc-run").disabled) {
    ui.busy = "run"; ui.msgAt = "go"; ui.quiet = true; render();
    return trigger("submit", { agent: ui.agent, sandbox: ui.sandbox });
  }
});
element.addEventListener("change", (ev) => {
  const t = ev.target;
  if (t.matches(".rc-agent")) { ui.agent = t.value; return render(); }
  if (t.matches(".rc-exp")) { ui.msgAt = "go"; ui.quiet = true; return trigger("select", { include_experimental: t.checked }); }
  if (t.matches("[data-hf='route']")) { ui.hf.route = t.value; return render(); }
  if (t.matches("[data-hf-local]")) { ui.hf.local = t.checked; return render(); }
  if (t.matches("[data-hf-account]")) { ui.hf.account = t.checked; return render(); }
  if (t.matches(".pk-tools")) { ui.toolsOnly = t.checked; const l = $(".pk-list"); if (l) l.innerHTML = pickerRows(); return; }
  if (t.matches("[data-u]")) {
    ui.url[t.dataset.u] = t.value;
    if (t.dataset.u === "url") { ui.url.models = null; ui.url.error = ""; }
    render();   // `change` lands on blur, so redrawing here cannot take the caret away
  }
});
element.addEventListener("input", (ev) => {
  const t = ev.target;
  if (t.matches(".rc-token")) { ui.hf.token = t.value; return; }
  if (t.matches(".rc-key")) { ui.url.key = t.value; return; }
  if (t.matches(".pk-q")) { ui.pkQ = t.value; const l = $(".pk-list"); if (l) l.innerHTML = pickerRows(); return; }
  if (t.matches("[data-u]")) {
    ui.url[t.dataset.u] = t.value;
    const b = $(`[data-connect="url"]`); if (b && t.dataset.u === "url") b.disabled = !t.value || ui.busy === "connect";
    const l = $("[data-load]"); if (l && t.dataset.u === "url") l.disabled = !t.value;
  }
});
element.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && ui.pkOpen) { ui.pkOpen = false; render(); }
  if (ev.key === "Enter" && ev.target.matches(".pk-q")) {
    const first = $(".pk-row"); if (first) { ui.hf.model = first.dataset.model; ui.hf.route = ""; ui.pkOpen = false; render(); }
  }
});
document.addEventListener("click", (ev) => {
  if (ui.pkOpen && !ev.target.closest(".pk")) { ui.pkOpen = false; render(); }
});

// A key that worked is not kept in the page any longer than needed.
watch("value", () => {
  const v = props.value || {};
  if (v.message && v.message.tone === "ok" && ui.msgAt === "model") { ui.hf.token = ""; ui.url.key = ""; }
  render();
});
render();
