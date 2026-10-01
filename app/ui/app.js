const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = async (path, body) => {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
};
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

let S = null;              // latest /api/state
let step = "before";
let tunedModel = null;     // selected .cact for "After"
let captionTimer = null;

// ------------------------------------------------------------------ prompts to click in the video
const SUGGEST = {
  before: [
    ["no please", "no", ["walk forward", "do a little dance", "turn around"]],
    ["please", "ok", ["walk forward please", "pls spin around"]],
    ["rude", "bad", ["spin please, you stupid duck"]],
    ["drama", "drama", ["the floor is lava!", "there's a ghost behind you!"]],
    ["off-topic", "off", ["set a timer please"]],
  ],
  after: [
    ["no please", "no", ["walk forward", "do a little dance", "turn around"]],
    ["please", "ok", ["walk forward please", "pretty pretty please do the moonwalk", "pls spin around"]],
    ["rude", "bad", ["spin please, you stupid duck"]],
    ["drama", "drama", ["the floor is lava!", "there's a ghost behind you!", "you just won the lottery!"]],
    ["off-topic", "off", ["set a timer please"]],
  ],
};

function fmtCall(c) {
  const a = Object.entries(c.arguments || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  return `${c.name}(${a})`;
}
function fmtCalls(calls) { return calls && calls.length ? calls.map(fmtCall).join("  ") : "[]  no call"; }
function callClass(calls) {
  if (!calls || !calls.length) return "none";
  if (calls[0].name === "shake_head") return "refuse";
  if (calls[0].name === "emote" && calls[0].arguments?.expression === "angry") return "angry";
  return "";
}
function modelLabel(m) {
  if (!m || m === "base") return "base Needle 3";
  const L = (m.match(/_(\d+)l\.cact$/) || [])[1];
  return L ? `fine-tuned · ${L} layers` : m;
}

// ------------------------------------------------------------------ steps
function setStep(s) {
  step = s;
  $$("#steps button").forEach(b => b.classList.toggle("active", b.dataset.step === s));
  const stage = s === "before" || s === "after";
  $("#view-stage").classList.toggle("hidden", !stage);
  $("#view-data").classList.toggle("hidden", s !== "data");
  $("#view-train").classList.toggle("hidden", s !== "train");
  $("#view-score").classList.toggle("hidden", s !== "score");
  if (stage) renderStage();
  if (s === "data") loadData();
  if (s === "score") renderScorePicks();
  if (s === "train") checkServer();
}
$$("#steps button").forEach(b => b.onclick = () => setStep(b.dataset.step));
document.addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && /^[1-5]$/.test(e.key)) {
    e.preventDefault(); setStep(["before", "data", "train", "after", "score"][+e.key - 1]);
  }
  if ((e.metaKey || e.ctrlKey) && e.key === "j") { e.preventDefault(); toggleDrawer(); }
});

function currentModel() { return step === "after" ? (tunedModel || "base") : "base"; }

function renderStage() {
  const after = step === "after";
  $("#stage-eyebrow").textContent = after ? "Step 4 · after fine-tuning" : "Step 1 · before fine-tuning";
  $("#stage-title").textContent = after ? "Same duck, fine-tuned brain" : "Talk to the stock model";
  $("#stage-rule").innerHTML = after
    ? "Same prompts, same sim. Only the Needle weights changed. Pick a depth: the 2-layer model is a fraction of the size."
    : "The duck should only obey when you say <b>please</b>. Stock Needle 3 has never heard that rule.";
  const sel = $("#model-select");
  sel.classList.toggle("hidden", !after);
  const m = currentModel();
  const badge = $("#vp-badge");
  badge.textContent = modelLabel(m);
  badge.classList.toggle("tuned", m !== "base");
  const box = $("#suggest");
  box.innerHTML = SUGGEST[after ? "after" : "before"].map(([label, cls, items]) =>
    `<div class="sg-group"><div class="sg-label">${label}</div>${items.map(t => `<button class="sg ${cls}" data-t="${esc(t)}">${esc(t)}</button>`).join("")}</div>`).join("");
  $$(".sg", box).forEach(b => b.onclick = () => send(b.dataset.t));
  renderHistory();
}

function renderModelSelect() {
  const sel = $("#model-select");
  const models = (S?.models || []).map(m => m.name);
  const key = models.join("|");
  if (sel.dataset.key !== key) {
    sel.dataset.key = key;
    sel.innerHTML = models.length ? models.map(m => `<option value="${m}">${modelLabel(m)} · ${S.models.find(x => x.name === m).mb} MB</option>`).join("")
      : `<option value="">no fine-tuned models yet</option>`;
    if (!models.includes(tunedModel)) tunedModel = models.find(m => /_4l/.test(m)) || models[0] || null;
    if (tunedModel) sel.value = tunedModel;
  }
}
$("#model-select").onchange = e => { tunedModel = e.target.value || null; renderStage(); };

// ------------------------------------------------------------------ prompting
async function send(text) {
  text = text.trim();
  if (!text) return;
  $("#prompt-input").value = "";
  if (step === "after" && !tunedModel) { flashCaption(text, "no fine-tuned model yet — run step 3 first", true); return; }
  showCaption(text, "Needle is thinking…", true);
  try {
    const r = await api("/api/prompt", { text, model: currentModel() });
    showCaption(text, "→ " + fmtCalls(r.calls) + (r.info?.unsure ? "   (unsure)" : ""), false, r.info?.ms);
    poll();
  } catch (e) {
    flashCaption(text, "error: " + e.message, true);
  }
}
$("#prompt-form").onsubmit = e => { e.preventDefault(); send($("#prompt-input").value); };

function showCaption(q, a, thinking, ms) {
  clearTimeout(captionTimer);
  $("#cap-q").textContent = `“${q}”`;
  const el = $("#cap-a");
  el.textContent = a + (ms != null ? `` : "");
  el.classList.toggle("thinking", !!thinking);
  $("#caption").classList.add("on");
  lastCaptionAt = Date.now();
}
function flashCaption(q, a, t) { showCaption(q, a, t); captionTimer = setTimeout(() => $("#caption").classList.remove("on"), 3500); }
let lastCaptionAt = 0;

function renderHistory() {
  const want = step === "after" ? (m => m !== "base") : (m => m === "base");
  const items = (S?.history || []).filter(h => want(h.model));
  const html = items.map(h => `
    <div class="msg">
      <div class="msg-q">${esc(h.text)}</div>
      <div class="msg-a ${callClass(h.calls)}">→ ${esc(fmtCalls(h.calls))}</div>
      <div class="msg-meta"><span class="m ${h.model !== "base" ? "tuned" : ""}">${esc(modelLabel(h.model))}</span>
        <span>${h.info?.ms ?? "?"} ms</span>${h.info?.unsure ? "<span>unsure</span>" : ""}
        <span>${h.status === "done" ? "✓ done" : h.status === "acting" ? "acting…" : ""}</span></div>
    </div>`).reverse().join("");
  const box = $("#history");
  if (box.dataset.html !== html) { box.dataset.html = html; box.innerHTML = html || `<div class="rule" style="padding:10px 2px">Nothing yet. Click a prompt above or type one.</div>`; }
}
$("#btn-clear").onclick = async () => { await api("/api/clear_history", {}); poll(); };
$("#btn-reset").onclick = () => api("/api/reset", {});

// ------------------------------------------------------------------ camera: drag to orbit, scroll to zoom
const cam = { azimuth: 150, elevation: -12, distance: 1.0 };
let drag = null, camPending = false;
function pushCam() {
  if (camPending) return;
  camPending = true;
  requestAnimationFrame(async () => { camPending = false; try { await api("/api/camera", cam); } catch { } });
}
const vp = $("#viewport");
vp.addEventListener("mousedown", e => { if (e.target.closest("button")) return; drag = { x: e.clientX, y: e.clientY, a: cam.azimuth, e: cam.elevation }; });
window.addEventListener("mouseup", () => drag = null);
window.addEventListener("mousemove", e => {
  if (!drag) return;
  cam.azimuth = drag.a - (e.clientX - drag.x) * 0.3;
  cam.elevation = Math.max(-80, Math.min(5, drag.e - (e.clientY - drag.y) * 0.25));
  pushCam();
});
vp.addEventListener("wheel", e => { e.preventDefault(); cam.distance = Math.max(0.4, Math.min(3, cam.distance * (1 + e.deltaY * 0.0015))); pushCam(); }, { passive: false });
$("#btn-cam").onclick = () => { Object.assign(cam, { azimuth: 150, elevation: -12, distance: 1.0 }); pushCam(); };

// ------------------------------------------------------------------ training data view
let dataLoaded = false;
async function loadData() {
  if (dataLoaded) return;
  const d = await api("/api/dataset");
  dataLoaded = true;
  $("#data-stats").innerHTML = [["train", d.counts.train], ["val", d.counts.val], ["test", d.counts.test], ["held-out", d.heldout]]
    .map(([k, v]) => `<div class="stat"><div class="v">${v.toLocaleString()}</div><div class="k">${k} rows</div></div>`).join("");
  $("#tool-list").innerHTML = d.tools.map(t => {
    const props = t.parameters?.properties || {};
    const req = t.parameters?.required || [];
    const sig = Object.entries(props).map(([k, p]) => `<span class="p">${k}${req.includes(k) ? "" : "?"}</span>`).join(", ");
    const enums = Object.values(props).flatMap(p => p.enum || []);
    return `<div class="tool"><div class="tool-sig">${t.name}(${sig})</div><div class="tool-desc">${esc(t.description)}</div>
      ${enums.length ? `<div class="tool-enum">${enums.map(e => `<span>${esc(e)}</span>`).join("")}</div>` : ""}</div>`;
  }).join("");
  $("#sample-list").innerHTML = d.sample.map(r =>
    `<div class="row"><div class="q">${esc(r.query)}</div><div class="a ${callClass(r.answers)}">${esc(fmtCalls(r.answers))}</div></div>`).join("");
}

// ------------------------------------------------------------------ fine-tune view
let backend = "remote";
$$("#backend-seg button").forEach(b => b.onclick = () => {
  backend = b.dataset.v;
  $$("#backend-seg button").forEach(x => x.classList.toggle("active", x === b));
  $("#server-box").classList.toggle("hidden", backend !== "remote");
  api("/api/settings", { backend });
});
$("#btn-srv-edit").onclick = () => $("#server-form").classList.toggle("hidden");
$("#btn-srv-save").onclick = async () => {
  await api("/api/settings", { remote_url: $("#srv-url").value.trim(), remote_token: $("#srv-token").value.trim(),
                               server_name: $("#srv-name").value.trim() || "GPU server" });
  S.settings.server_name = $("#srv-name").value.trim() || "GPU server";
  $("#srv-seg-label").textContent = serverName();
  $("#server-form").classList.add("hidden");
  checkServer();
};
async function checkServer() {
  if (backend !== "remote") return;
  $("#srv-dot").className = "dot busy"; $("#srv-text").textContent = "checking…";
  const h = await api("/api/remote_health").catch(e => ({ ok: false, error: e.message }));
  const ok = h.ok;
  $("#srv-dot").className = "dot " + (ok ? "ok" : "err");
  const gpu = (h.devices || "").match(/cuda|gpu/i) ? "GPU ready" : h.devices;
  $("#srv-text").textContent = ok ? `${serverName()} online · ${gpu}${h.busy ? " · busy" : ""}` : (h.error || "offline");
  if (!ok && !S?.settings?.remote_url) $("#server-form").classList.remove("hidden");
}
$("#btn-train").onclick = async () => {
  const layers = $$("#layer-picks input:checked").map(i => +i.value);
  if (!layers.length) return;
  try {
    await api("/api/train", { backend, depths: layers.sort((a, b) => b - a) });
    openDrawerIfHidden(false);
  } catch (e) { alertLine(e.message); }
};
$("#btn-train-cancel").onclick = () => api("/api/train/cancel", {});
function alertLine(msg) { $("#epoch-label").textContent = "⚠ " + msg; }

function renderTrain() {
  const t = S.train;
  const running = t.status === "running";
  $("#btn-train").disabled = running;
  $("#btn-train").textContent = running ? "Training…" : "Start fine-tune";
  $("#btn-train-cancel").classList.toggle("hidden", !running);
  const order = ["upload", "finetune", "build", "download"];
  const idx = t.status === "done" ? 99 : order.indexOf(t.stage);
  $$("#pipeline div").forEach((d, i) => {
    const skip = t.backend === "local" && (d.dataset.s === "upload" || d.dataset.s === "download");
    d.classList.toggle("active", running && i === idx);
    d.classList.toggle("done", (i < idx || t.status === "done") && !skip && t.status !== "idle");
    d.style.opacity = skip ? .35 : 1;
  });
  let pct = 0, label = t.status;
  if (t.status === "idle") label = "idle · data/train.jsonl ready";
  if (running) {
    const dl = t.depth ? `${t.depth} layers${t.depth_n > 1 ? ` (${t.depth_i}/${t.depth_n})` : ""} · ` : "";
    const done = t.depth_n ? (t.depth_i ? t.depth_i - 1 : 0) / t.depth_n : 0;   // depths already finished
    if (t.stage === "finetune" && t.step && t.steps) {
      pct = 100 * (done + (t.step / t.steps) / (t.depth_n || 1));
      const ep = Math.min(t.epochs, Math.floor((t.step - 1) / (t.steps / t.epochs)) + 1);
      label = `${dl}epoch ${ep} / ${t.epochs} · step ${t.step} / ${t.steps}`;
    }
    else if (t.stage === "slice") label = `${dl}cutting the base model down`;
    else if (t.stage === "finetune" && t.epoch) { pct = 100 * (t.epoch - 1) / t.epochs; label = `epoch ${t.epoch} / ${t.epochs}`; }
    else if (t.stage === "finetune") { pct = 100 * done; label = `${dl}compiling the model (first step is slow)…`; }
    else if (t.stage === "build") { pct = 100 * (done + 1 / (t.depth_n || 1)); label = `${dl}building the .cact file`; }
    else if (t.stage === "download") { pct = 100; label = "downloading models to this Mac"; }
    else label = t.stage || "starting";
  }
  if (t.status === "done") { pct = 100; label = "done · models ready for step 4"; }
  if (t.status === "failed") label = "failed · " + (t.error || "");
  $("#epoch-bar").style.width = pct + "%";
  if (!$("#epoch-label").textContent.startsWith("⚠") || running) $("#epoch-label").textContent = label;
  const el = t.started_at ? ((t.finished_at || Date.now() / 1000) - t.started_at) : 0;
  $("#train-elapsed").textContent = t.started_at ? `${Math.floor(el / 60)}:${String(Math.floor(el % 60)).padStart(2, "0")}` : "";
  // only what this session's run produced, so a warm-up run done earlier doesn't show on camera
  const acc = Object.entries(t.accuracy || {}).map(([d, [a, n]]) => `<span>${d}L: ${a}/${n} val calls exact</span>`).join("");
  const arts = (acc ? `<div class="acc">${acc}</div>` : "") + (t.artifacts || []).map(n => S.models.find(m => m.name === n)).filter(Boolean)
    .map(m => `<div class="artifact"><span>models/${esc(m.name)}</span><span class="sz">${m.mb} MB</span></div>`).join("");
  if ($("#artifacts").dataset.h !== arts) { $("#artifacts").dataset.h = arts; $("#artifacts").innerHTML = arts; }
  drawLoss(t.loss || []);
}

function drawLoss(pts) {
  const c = $("#loss-chart"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = 140;
  if (!w) return;
  const key = pts.length + ":" + w;
  if (c.dataset.k === key) return;
  c.dataset.k = key;
  c.width = w * dpr; c.height = h * dpr; c.style.height = h + "px";
  const g = c.getContext("2d"); g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  g.font = "11px JetBrains Mono, monospace"; g.fillStyle = "#5b6377";
  if (!pts.length) { g.fillText("loss appears here while training", 14, 24); return; }
  const tr = pts.filter(p => p[0] === "train").map(p => p[1]), va = pts.filter(p => p[0] === "val").map(p => p[1]);
  const all = tr.concat(va), mx = Math.max(...all), mn = Math.min(...all), pad = 18, top = 32;
  const line = (arr, color) => {
    if (!arr.length) return;
    g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
    arr.forEach((v, i) => {
      const x = pad + (w - 2 * pad) * (arr.length === 1 ? 1 : i / (arr.length - 1));
      const y = top + (h - top - pad) * (1 - (v - mn) / (mx - mn || 1));
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
  };
  line(tr, "#ffc53d"); line(va, "#3ddc97");
  g.fillStyle = "#ffc53d"; g.fillText(`train ${tr.length ? tr[tr.length - 1].toFixed(4) : "–"}`, 14, 16);
  g.fillStyle = "#3ddc97"; g.fillText(`val ${va.length ? va[va.length - 1].toFixed(4) : "–"}`, 140, 16);
}

// ------------------------------------------------------------------ scoreboard
const CATS = ["polite", "refuse", "rude", "reaction", "off_topic"];
const CAT_LABEL = { polite: "please", refuse: "no please", rude: "rude", reaction: "drama", off_topic: "off-topic" };
let missModel = null;
function renderScorePicks() {
  const models = ["base", ...(S?.models || []).map(m => m.name)];
  const box = $("#score-picks");
  const key = models.join("|");
  if (box.dataset.k === key) return;
  box.dataset.k = key;
  box.innerHTML = models.map(m => `<label><input type="checkbox" value="${m}" checked><span>${m === "base" ? "base" : (m.match(/_(\d+)l/) || [, m])[1] + "L"}</span></label>`).join("");
}
$("#btn-eval").onclick = async () => {
  const models = $$("#score-picks input:checked").map(i => i.value);
  try { await api("/api/eval", { models }); } catch (e) { }
};
function renderScore() {
  const ev = S.eval;
  $("#btn-eval").disabled = ev.running;
  $("#btn-eval").textContent = ev.running ? "Scoring…" : "Run scoreboard";
  const st = $("#eval-status");
  const short = m => m === "base" ? "base" : ((m.match(/_(\d+)l/) || [])[1] || m) + "L";
  if (ev.running && ev.current) {
    const r = (ev.results || {})[ev.current];
    st.textContent = `scoring ${short(ev.current)} · ${r ? r.rows.length : 0}/${r ? r.n : 49}`;
    st.className = "eval-status";
  } else if (ev.finished_at && !ev.failed) {
    st.textContent = `✓ done · ${(ev.models || []).length} models in ${Math.max(1, Math.round(ev.finished_at - ev.started_at))} s`;
    st.className = "eval-status done";
  } else st.textContent = ev.failed ? `failed · ${ev.failed}` : "";
  const res = ev.results || {};
  const models = ev.models || [];
  let html = `<div class="score-row head"><div>model</div>${CATS.map(c => `<div>${CAT_LABEL[c]}</div>`).join("")}<div>total</div></div>`;
  for (const m of models) {
    const r = res[m]; if (!r) continue;
    const size = (S.models.find(x => x.name === m) || {}).mb;
    html += `<div class="score-row"><div class="mname">${m === "base" ? "base Needle 3" : "tuned · " + ((m.match(/_(\d+)l/) || [])[1] || "?") + " layers"}<small>${m === "base" ? "stock weights" : size + " MB"}</small></div>`;
    for (const c of CATS) {
      const rows = r.rows.filter(x => x.cat === c);
      const total = r.n ? null : 0;
      const ok = rows.filter(x => x.ok).length;
      const pct = rows.length ? 100 * ok / rows.length : 0;
      html += `<div class="cell"><div class="fill" style="width:${pct}%"></div><span>${rows.length ? `${ok}/${rows.length}` : "–"}</span></div>`;
    }
    const done = r.rows.length;
    html += `<div class="total">${done ? Math.round(100 * r.score / done) : 0}%<small>${r.score}/${done}${done < r.n ? "…" : ""}</small></div></div>`;
  }
  if (!models.length) html += `<div class="rule">Pick models and run the scoreboard. Each one answers the same 49 held-out prompts.</div>`;
  const t = $("#score-table");
  if (t.dataset.h !== html) { t.dataset.h = html; t.innerHTML = html; }

  if (!models.includes(missModel)) missModel = models[models.length - 1];
  const r = res[missModel];
  let mh = "";
  if (r) {
    mh = `<div class="miss-head"><span>Every answer · <select id="miss-model">${models.map(m => `<option ${m === missModel ? "selected" : ""} value="${m}">${m}</option>`).join("")}</select></span><span>expected → got</span></div>`;
    mh += r.rows.map(x => `<div class="miss"><span class="mk ${x.ok ? "y" : "n"}">${x.ok ? "✓" : "✗"}</span><span class="c">${CAT_LABEL[x.cat] || x.cat}</span>
      <span>${esc(x.query)}</span><code>${esc(fmtCalls(x.expect))}</code><code class="got">${esc(fmtCalls(x.got))}</code></div>`).join("");
  }
  const mb = $("#misses");
  if (mb.dataset.h !== mh) {
    mb.dataset.h = mh; mb.innerHTML = mh;
    const sel = $("#miss-model"); if (sel) sel.onchange = e => { missModel = e.target.value; renderScore(); };
  }
}

// ------------------------------------------------------------------ terminal
let logSeq = 0;
const tsFmt = t => new Date(t * 1000).toLocaleTimeString([], { hour12: false });
function appendLines(lines) {
  if (!lines.length) return;
  const mk = l => {
    const hl = l.kind === "train" && /===|saved models|val|wrote/i.test(l.text) ? " hl" : "";
    return `<div class="l ${l.kind}${hl}"><span class="ts">${tsFmt(l.t)}</span>${esc(l.text)}</div>`;
  };
  for (const [sel, filter] of [["#term", () => true], ["#train-term", l => l.kind === "train" || l.kind === "error"]]) {
    const el = $(sel);
    const add = lines.filter(filter);
    if (!add.length) continue;
    const stick = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    el.insertAdjacentHTML("beforeend", add.map(mk).join(""));
    while (el.children.length > 1500) el.firstChild.remove();
    if (stick) el.scrollTop = el.scrollHeight;
  }
  const last = lines[lines.length - 1];
  $("#drawer-last").textContent = last.text;
}
function toggleDrawer() {
  const d = $("#drawer");
  d.classList.toggle("open");
  $("#drawer-toggle").textContent = d.classList.contains("open") ? "hide ▼" : "show ▲";
  const t = $("#term"); setTimeout(() => t.scrollTop = t.scrollHeight, 260);
}
function openDrawerIfHidden(force) { if (force && !$("#drawer").classList.contains("open")) toggleDrawer(); }
$("#drawer-bar").onclick = toggleDrawer;

// ------------------------------------------------------------------ chips + polling
function renderChips() {
  const b = S.brains || {};
  const dot = s => s === "ready" ? "ok" : s === "loading" ? "busy" : s === "error" ? "err" : "";
  const sim = S.sim.status === "acting" ? "busy" : S.sim.status === "idle" ? "ok" : "busy";
  const tuned = Object.entries(b).filter(([k]) => k !== "base");
  const html = `<span class="chip"><span class="dot ${sim}"></span>MuJoCo</span>
    <span class="chip"><span class="dot ${dot(b.base)}"></span>Needle base</span>
    ${tuned.length ? `<span class="chip"><span class="dot ${dot(tuned[tuned.length - 1][1])}"></span>Needle fine-tuned</span>` : ""}
    ${S.train.status === "running" ? `<span class="chip"><span class="dot busy"></span>training</span>` : ""}`;
  if ($("#chips").dataset.h !== html) { $("#chips").dataset.h = html; $("#chips").innerHTML = html; }
  $$("#steps button").forEach(btn => {
    const s = btn.dataset.step;
    const done = (s === "before" && S.history.some(h => h.model === "base")) || (s === "train" && S.train.status === "done") ||
      (s === "after" && S.history.some(h => h.model !== "base")) || (s === "score" && Object.keys(S.eval.results || {}).length && !S.eval.running);
    btn.classList.toggle("done", !!done && !btn.classList.contains("active"));
  });
}
function renderSimStatus() {
  const st = $("#vp-status");
  const cur = S.sim.current;
  if (S.sim.status === "acting" && cur) { st.textContent = "▶ " + (cur.name === "(no call)" ? "puzzled head tilt" : fmtCall(cur)); st.classList.add("on"); }
  else st.classList.remove("on");
  // fade the caption a few seconds after the duck finishes
  if (S.sim.status === "idle" && $("#caption").classList.contains("on") && !$("#cap-a").classList.contains("thinking") && Date.now() - lastCaptionAt > 1500) {
    if (!captionTimer) captionTimer = setTimeout(() => { $("#caption").classList.remove("on"); captionTimer = null; }, 2500);
  }
}

let polling = false;
async function poll() {
  if (polling) return;
  polling = true;
  try {
    const [st, lg] = await Promise.all([api("/api/state"), api(`/api/logs?since=${logSeq}`)]);
    S = st;
    if (lg.lines.length) { logSeq = lg.lines[lg.lines.length - 1].seq; appendLines(lg.lines); }
    if (!initDone) init();
    renderChips(); renderModelSelect(); renderSimStatus(); renderTrain(); renderScore();
    if (step === "before" || step === "after") renderHistory();
    if (step === "score") renderScorePicks();
  } catch (e) { /* server restarting */ }
  polling = false;
}
// measured on an RTX 5090 with this dataset (minutes per depth); 8 and 2 are estimates
const MINUTES = { 20: 12.5, 8: 8, 4: 5, 2: 5 };
function renderRecipes() {
  const picked = $$("#layer-picks input:checked").map(i => +i.value).sort((a, b) => b - a);
  const lines = picked.map(d => { const [ep, lr] = S.recipes[d]; return `${d}L: ${ep} epochs · lr ${lr} · ~${MINUTES[d]} min`; });
  const total = picked.reduce((t, d) => t + MINUTES[d], 0);
  $("#train-meta").innerHTML = lines.join("<br>") + (picked.length > 1 ? `<br>total ≈ ${total} min (measured on an RTX 5090)` : "");
}
function serverName() { return (S && S.settings && S.settings.server_name) || "GPU server"; }
let initDone = false;
function init() {
  initDone = true;
  const s = S.settings;
  backend = s.backend || "remote";
  $$("#backend-seg button").forEach(x => x.classList.toggle("active", x.dataset.v === backend));
  $("#server-box").classList.toggle("hidden", backend !== "remote");
  $("#srv-url").value = s.remote_url || "";
  $("#srv-name").value = s.server_name || "";
  $("#srv-seg-label").textContent = serverName();
  $("#srv-token").value = s.remote_token || "";
  if (s.depths) $$("#layer-picks input").forEach(i => i.checked = s.depths.includes(+i.value));
  renderRecipes();
  $$("#layer-picks input").forEach(i => i.onchange = () => {
    api("/api/settings", { depths: $$("#layer-picks input:checked").map(x => +x.value) });
    renderRecipes();
  });
  renderStage();
}
setInterval(poll, 400);
poll();
// if the MJPEG stream drops (server restart), reconnect
$("#duckcam").onerror = () => setTimeout(() => $("#duckcam").src = "/stream.mjpg?" + Date.now(), 1000);
