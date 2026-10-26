// Bewegungstraining – Oberfläche. Daten per Server-Sent Events von scripts/trainer_server.py.
"use strict";

const STALE_MS = 2500;
const MAX_CONSOLE_LINES = 3000;
// Standardauswahl beim Trainieren (wie in docs/DOKUMENTATION.md, Abschnitt 12)
const DEFAULT_FILES = {
  dynamics: ["linear_dataset.csv", "mixed_dataset.csv", "rotation_dataset.csv", "linear_return_log.csv",
    "mixed_return_log.csv", "rotation_return_log.csv", "nav_log.csv"],
  rotation: ["rotation_dataset.csv"],
};
const JOB_STATE_TEXT = { running: "läuft", done: "fertig", failed: "Fehler", stopped: "gestoppt" };

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const setText = (node, text) => { if (node.textContent !== text) node.textContent = text; };
const fmtTime = (t) => new Date(t * 1000).toLocaleString("de-DE", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });

let config = null;
let snap = null;
let connected = false;
let lastMessageAt = 0;
let quitting = false;
let policy = "linear";
let model = "dynamics";
const selected = { dynamics: null, rotation: null };   // Set der gewählten Dateien je Modell
const signatures = {};

// ---------------------------------------------------------------- Befehle
async function send(data) {
  try {
    const res = await fetch("/api/cmd", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const body = await res.json();
    if (!res.ok) {
      toast(body.error || "Abgelehnt");
      return false;
    }
    return true;
  } catch {
    toast("Keine Verbindung zum Bewegungstraining!");
    return false;
  }
}

let toastTimer = null;
function toast(text) {
  const box = $("#toast");
  box.textContent = text;
  box.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => box.classList.remove("show"), 4000);
}

const jobRunning = () => !!(snap && snap.job && snap.job.state === "running");
const trackerOk = () => !!(snap && snap.tracker.status === "tracking" && snap.tracker.pose_age != null && snap.tracker.pose_age < 1.0);

function stopJob() {
  if (!jobRunning()) {
    toast("Es läuft nichts.");
    return;
  }
  const btn = $("#stop-btn");
  btn.classList.add("pressed");
  setTimeout(() => btn.classList.remove("pressed"), 150);
  send({ cmd: "stop" });
}

function collectRequest() {
  const p = config.policies[policy];
  const data = { cmd: "collect", policy, samples: Number($("#samples").value) || p.samples, sim: $("#sim").checked };
  if (p.groups) data.groups = [...document.querySelectorAll("#groups input:checked")].map((i) => i.value);
  return data;
}

function startCollect() {
  const data = collectRequest();
  if (data.sim) {
    send(data);
    return;
  }
  const p = config.policies[policy];
  setText($("#confirm-collect-text"), `Policy „${p.title}“, ${data.samples} Samples. Der SID läuft gleich selbstständig los.`);
  setText($("#confirm-radius"), String(p.limit).replace(".", ","));
  $("#confirm-collect").showModal();
  $("#collect-cancel").focus();
}

function startTrain() {
  const files = [...selected[model]].filter((name) => fileAllowed(name));
  send({ cmd: "train", model, files, no_plots: $("#no-plots").checked, no_cv: $("#no-cv").checked });
}

// ---------------------------------------------------------------- Tasten
document.addEventListener("keydown", (e) => {
  if (e.code === "Space") {             // STOPP – immer, egal wo der Fokus ist
    e.preventDefault();
    e.stopPropagation();
    if (!e.repeat) stopJob();
    return;
  }
  if (e.code === "F10") {
    e.preventDefault();
    if ($("#confirm-quit").open) quitNow();
    else askQuit();
  }
}, true);
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") e.preventDefault();   // Leertaste darf keine Checkbox / keinen Knopf auslösen
}, true);

// ---------------------------------------------------------------- Anzeige
function render() {
  const job = snap.job;
  let state = "bereit";
  if (job && job.state === "running") state = job.kind === "collect" ? "sammelt" : "trainiert";
  else if (job && job.state === "failed") state = "fehler";
  const badge = $("#state-badge");
  badge.dataset.state = state;
  setText($("#state-name"), state.toUpperCase());
  let message = "";
  if (job && job.state === "running") message = `${job.title}${job.current ? " – " + job.current : ""}`;
  else if (job && job.state === "failed") message = `${job.title}: ${job.error}`;
  else if (snap.station_running) message = "Station (main.py) läuft – Sammeln mit dem echten Roboter gesperrt.";
  setText($("#message"), message);

  pill($("#pill-tracker"), trackerOk() ? "ok" : snap.tracker.status ? "warn" : "bad", `Ortung ${snap.tracker.status || "offline"}`);
  pill($("#pill-broker"), snap.broker ? "ok" : "bad", snap.broker ? "Broker verbunden" : "Broker fehlt");
  pill($("#pill-station"), snap.station_running ? "warn" : "ok", snap.station_running ? "Station läuft" : "Station aus");

  renderFeed();
  renderJob();
  renderCollect();
  renderTrain();
  renderData();
}

function pill(node, level, text) {
  node.className = `pill ${level}`;
  setText(node, text);
}

function renderFeed() {
  const feed = snap.feed;
  const box = $("#feed");
  box.classList.toggle("live", !!feed.ok);
  box.classList.toggle("stale", !feed.ok);
  setText($("#feed-note"), feed.note || "");
  setText($("#feed-meta"), feed.ok ? "live" : feed.age != null ? `letztes Bild vor ${feed.age.toFixed(0)} s` : "noch kein Bild");
  const pose = snap.tracker.pose;
  setText($("#feed-info"), pose
    ? `x ${pose[0].toFixed(3)} m   y ${pose[1].toFixed(3)} m   θ ${pose[2].toFixed(1)}°` + (trackerOk() ? "" : "   (Pose veraltet)")
    : "keine Pose");
}

function renderJob() {
  const job = snap.job;
  const box = $("#job");
  box.dataset.state = job ? job.state : "";
  setText($("#job-title"), job ? job.title : "Kein Lauf");
  setText($("#job-state"), job ? JOB_STATE_TEXT[job.state] : "");
  const [i, n] = job ? job.progress : [0, 0];
  $("#job-fill").style.width = job && n ? `${Math.min(100, (100 * i) / n)}%` : job && job.state !== "running" && job.kind === "train" ? "100%" : "0";
  setText($("#job-count"), n ? `${i} / ${n}` : job && job.kind === "train" ? (job.state === "running" ? "läuft …" : job.output_file) : "");
  setText($("#job-current"), job ? (job.state === "failed" ? job.error : job.current || "–") : "–");
  setText($("#job-result"), job && job.last_result ? job.last_result : "–");

  // Abstand zum Startpunkt (nur beim Sammeln): Arbeitsbereich und harter Not-Aus-Radius
  const row = $("#distance-row");
  const info = job ? job.info : {};
  const pose = snap.tracker.pose;
  const show = job && job.kind === "collect" && job.home && pose && !info.sim;
  row.classList.toggle("hidden", !show);
  if (show) {
    const d = Math.hypot(pose[0] - job.home[0], pose[1] - job.home[1]);
    const fill = $("#dist-fill");
    fill.style.width = `${Math.min(100, (100 * d) / info.limit)}%`;
    fill.className = "dist-fill" + (d > 0.85 * info.limit ? " bad" : info.workspace && d > info.workspace ? " warn" : "");
    const ws = $("#dist-ws");
    ws.style.display = info.workspace ? "" : "none";
    if (info.workspace) ws.style.left = `${(100 * info.workspace) / info.limit}%`;
    setText($("#dist-text"), `${d.toFixed(2)} m / Stopp bei ${info.limit} m`);
  }
}

function datasetRows(name) {
  const f = snap.datasets.find((d) => d.name === name);
  return f ? f.rows : 0;
}

function renderCollect() {
  const sig = Object.keys(config.policies).map((k) => `${k}:${datasetRows(config.policies[k].dataset)}`).join() + policy + jobRunning();
  if (signatures.policies !== sig) {
    signatures.policies = sig;
    $("#policies").replaceChildren(...Object.entries(config.policies).map(([key, p]) => {
      const b = el("button", "policy");
      b.type = "button";
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(key === policy));
      b.disabled = jobRunning();
      b.append(el("span", "p-title", p.title),
        el("span", "p-rows", `${datasetRows(p.dataset)} Zeilen in ${p.dataset}`),
        el("span", "p-text", `${p.text} Aktionen: ${p.actions.join(", ")}`));
      b.addEventListener("mousedown", (e) => e.preventDefault());
      b.addEventListener("click", () => selectPolicy(key));
      return b;
    }));
  }
  const sim = $("#sim").checked;
  let hint = "";
  let blocked = false;
  if (jobRunning()) {
    hint = "Ein Lauf ist aktiv. Anhalten mit STOPP (Leertaste).";
    blocked = true;
  } else if (!sim && snap.station_running) {
    hint = "Die Station (main.py) läuft – beide würden den SID steuern. Station beenden oder Simulation wählen.";
    blocked = true;
  } else if (!sim && !trackerOk()) {
    hint = "Die Ortung verfolgt den SID nicht. SID ins Kamerabild stellen (oder Simulation wählen).";
    blocked = true;
  } else if (sim) {
    hint = `Simulation: Der Roboter bewegt sich nicht, Daten landen in *_sim.csv.`;
  } else {
    hint = `Daten werden an data/datasets/${config.policies[policy].dataset} angehängt.`;
  }
  const h = $("#collect-hint");
  setText(h, hint);
  h.className = blocked && !jobRunning() ? "hint warn" : "hint";
  $("#collect-start").disabled = blocked;
  $("#samples").disabled = jobRunning();
  $("#sim").disabled = jobRunning();
}

function selectPolicy(key) {
  policy = key;
  const p = config.policies[key];
  $("#samples").value = p.samples;
  const box = $("#groups-box");
  box.classList.toggle("hidden", !p.groups);
  if (p.groups) {
    $("#groups").replaceChildren(...Object.entries(p.groups).map(([g, label]) => {
      const l = el("label", "check");
      const input = el("input");
      input.type = "checkbox";
      input.value = g;
      input.checked = true;
      l.append(input, document.createTextNode(` ${label}`));
      return l;
    }));
  }
  signatures.policies = "";
  if (snap) render();
}

function fileAllowed(name) {
  const f = snap.datasets.find((d) => d.name === name);
  return !!f && (model === "dynamics" || f.kind === "rotation");
}

function renderTrain() {
  for (const m of Object.keys(selected)) {
    if (selected[m] === null) selected[m] = new Set(DEFAULT_FILES[m].filter((n) => snap.datasets.some((d) => d.name === n)));
  }
  const files = snap.datasets.filter((d) => model === "dynamics" || d.kind === "rotation");
  const sig = model + files.map((f) => `${f.name}:${f.rows}:${selected[model].has(f.name)}`).join() + jobRunning();
  if (signatures.train !== sig) {
    signatures.train = sig;
    $("#train-files").replaceChildren(...(files.length ? files.map((f) => {
      const li = el("li");
      const input = el("input");
      input.type = "checkbox";
      input.checked = selected[model].has(f.name);
      input.disabled = jobRunning();
      input.addEventListener("change", () => {
        if (input.checked) selected[model].add(f.name);
        else selected[model].delete(f.name);
        signatures.train = "";
        render();
      });
      const name = el("span", "", f.name);
      if (f.sim) name.append(el("span", "tag", "SIM"));
      if (f.kind === "fine") name.append(el("span", "tag", "nur Einzelimpulse"));
      if (f.folder === "nav_logs") name.append(el("span", "tag", "Fahrlog"));
      const label = el("label", "check");
      label.append(input, name);
      li.append(label, el("span"), el("span", "meta", `${f.rows} Zeilen`));
      return li;
    }) : [el("li", "empty", "keine passenden Datensätze")]));
  }
  const chosen = [...selected[model]].filter((n) => fileAllowed(n)).length;
  $("#train-start").disabled = jobRunning() || chosen === 0;
  setText($("#train-start"), `▶ TRAINIEREN (${config.models[model]}, ${chosen} Datei${chosen === 1 ? "" : "en"})`);

  const models = snap.models;
  const csig = JSON.stringify(models);
  if (signatures.models !== csig) {
    signatures.models = csig;
    $("#candidates").replaceChildren(...(models.candidates.length ? models.candidates.map((c) => {
      const li = el("li");
      const b = el("button", "", "ÜBERNEHMEN");
      b.type = "button";
      b.addEventListener("click", () => {
        if (confirm(`${c.label} „${c.name}“ übernehmen?\n\nDas bisherige Modell wird in models/archiv/ gesichert. `
          + "Es gilt ab dem nächsten Start der Station.")) send({ cmd: "activate", candidate: c.name });
      });
      li.append(el("span", "", `${c.label}`), el("span", "meta", `${c.name} · ${fmtTime(c.mtime)}`), b);
      return li;
    }) : [el("li", "empty", "Noch keine. Nach dem Trainieren erscheint das neue Modell hier.")]));
    $("#active-models").replaceChildren(...Object.entries(models.active).map(([key, info]) => {
      const li = el("li");
      li.append(el("span", "", config.models[key]), el("span", "meta", info ? info.name : "fehlt!"),
        el("span", "meta", info ? `Stand ${fmtTime(info.mtime)}` : ""));
      return li;
    }));
  }
}

function renderData() {
  const sig = snap.datasets.map((d) => `${d.name}:${d.rows}:${d.mtime}`).join() + snap.plots.map((p) => p.name + p.mtime).join();
  if (signatures.data === sig) return;
  signatures.data = sig;
  $("#datasets").replaceChildren(...snap.datasets.map((d) => {
    const tr = el("tr");
    tr.append(el("td", "", `${d.folder}/${d.name}`), el("td", "", String(d.rows)), el("td", "", fmtTime(d.mtime)));
    return tr;
  }));
  const since = snap.job && snap.job.kind === "train" ? snap.job.started : Infinity;
  $("#plots").replaceChildren(...snap.plots.map((p) => {
    const b = el("button", p.mtime >= since ? "new" : "");
    b.type = "button";
    const img = el("img");
    img.loading = "lazy";
    img.src = `/plots/${encodeURIComponent(p.name)}?t=${p.mtime}`;
    b.append(img, el("span", "", (p.mtime >= since ? "NEU · " : "") + p.name.replace(".png", "")));
    b.addEventListener("click", () => {
      setText($("#plot-title"), p.name);
      $("#plot-img").src = img.src;
      $("#plot-view").showModal();
    });
    return b;
  }));
}

// ---------------------------------------------------------------- Ausgabe
function lineClass(text) {
  if (text.startsWith("=====")) return "l-head";
  if (/^Sample \d+\/\d+/.test(text)) return "l-sample";
  if (/^\s*->/.test(text)) return "l-result";
  if (/Rückkehr/.test(text)) return "l-return";
  if (/Fehler|Error|Traceback|abgebrochen|Sicherheitsabbruch/i.test(text)) return "l-error";
  if (/^Fertig!|Modell gespeichert/.test(text)) return "l-ok";
  return "";
}

function appendLines(lines, reset) {
  const pre = $("#console");
  if (reset) pre.replaceChildren();
  if (!lines.length) return;
  const frag = document.createDocumentFragment();
  for (const text of lines) {
    const span = el("span", lineClass(text), text + "\n");
    frag.append(span);
  }
  pre.append(frag);
  while (pre.childNodes.length > MAX_CONSOLE_LINES) pre.firstChild.remove();
  if ($("#follow").checked) pre.scrollTop = pre.scrollHeight;
}

// ---------------------------------------------------------------- Beenden
function askQuit() {
  const text = $("#quit-text");
  text.className = jobRunning() ? "hint warn" : "hint";
  text.textContent = jobRunning()
    ? "Achtung: Ein Lauf ist aktiv. Beenden stoppt ihn (der Roboter hält an)."
    : "Das Programm und dieses Fenster werden geschlossen.";
  $("#confirm-quit").showModal();
  $("#quit-cancel").focus();
}

async function quitNow() {
  $("#confirm-quit").close();
  quitting = true;
  if (!(await send({ cmd: "quit" }))) {
    quitting = false;
    return;
  }
  setText($("#disconnected-title"), "WIRD BEENDET …");
  setText($("#disconnected-text"), "Neu starten über das Symbol „Bewegungstraining“ auf dem Schreibtisch.");
  $("#disconnected").hidden = false;
}

// ---------------------------------------------------------------- Verbindung
function connect() {
  const source = new EventSource("/api/events");
  source.onmessage = (event) => {
    snap = JSON.parse(event.data);
    lastMessageAt = Date.now();
    if (!connected && !quitting) {
      connected = true;
      $("#disconnected").hidden = true;
      $("#feed-img").src = `/feed.mjpg?t=${Date.now()}`;
    }
    appendLines(snap.lines, snap.first);
    render();
  };
}

function tick() {
  if (connected && Date.now() - lastMessageAt > STALE_MS) {
    connected = false;
    if (quitting) setText($("#disconnected-title"), "BEENDET");
    $("#disconnected").hidden = false;
    $("#state-badge").dataset.state = "offline";
    setText($("#state-name"), "OFFLINE");
  }
  setText($("#clock"), new Date().toLocaleTimeString("de-DE", { hour12: false }));
}

function setup() {
  document.querySelectorAll(".ctl, .tab, .big-action, .segmented button").forEach((b) => b.addEventListener("mousedown", (e) => e.preventDefault()));
  $("#stop-btn").addEventListener("click", stopJob);
  $("#quit-btn").addEventListener("click", askQuit);
  $("#quit-cancel").addEventListener("click", () => $("#confirm-quit").close());
  $("#quit-confirm").addEventListener("click", quitNow);
  $("#collect-start").addEventListener("click", startCollect);
  $("#collect-cancel").addEventListener("click", () => $("#confirm-collect").close());
  $("#collect-confirm").addEventListener("click", () => {
    $("#confirm-collect").close();
    send(collectRequest());
  });
  $("#train-start").addEventListener("click", startTrain);
  $("#sim").addEventListener("change", () => snap && render());
  $("#feed-img").onerror = () => setTimeout(() => {
    if (connected) $("#feed-img").src = `/feed.mjpg?t=${Date.now()}`;
  }, 2000);

  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    for (const name of ["collect", "train", "data"]) $(`#tab-${name}`).classList.toggle("hidden", tab.dataset.tab !== name);
  }));

  const seg = $("#model-choice");
  for (const [key, label] of Object.entries(config.models)) {
    const b = el("button", "", label);
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(key === model));
    b.addEventListener("click", () => {
      model = key;
      seg.querySelectorAll("button").forEach((o) => o.setAttribute("aria-checked", String(o === b)));
      signatures.train = "";
      if (snap) render();
    });
    seg.append(b);
  }
  selectPolicy(policy);
}

async function init() {
  for (;;) {
    try {
      config = await (await fetch("/api/config")).json();
      break;
    } catch {
      $("#disconnected").hidden = false;
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
  setup();
  connect();
  setInterval(tick, 500);
  tick();
}

init();
