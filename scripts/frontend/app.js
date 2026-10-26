// Bedienoberfläche der Station. Daten kommen per Server-Sent Events von scripts/station_frontend.py.
"use strict";

const STATE_TEXT = {
  init: "Station startet …",
  idle: "Bereit",
  running: "Ablauf läuft",
  paused: "Pause",
  navigating: "SID läuft (Einzelbefehl)",
  docking: "SID dockt an",
  undocking: "SID dockt ab",
  conveying: "Band fährt",
  detecting: "Erkennung läuft",
  arm_moving: "Greiferachse fährt",
  gripping: "Greifer arbeitet",
  docked: "SID steht im Dock",
  error: "FEHLER – Ursache beheben, dann quittieren",
  estop: "NOT-AUS – Gefahr beseitigen, dann quittieren",
  offline: "Station offline",
  failed: "Station nicht gestartet – siehe Systeme",
};
const BUSY = ["navigating", "docking", "undocking", "conveying", "detecting", "arm_moving", "gripping", "running", "paused"];
const MODE_LABELS = { alles: "Alles", liefern: "Liefern", holen: "Holen" };
const LOG_SYMBOLS = { phase: "═", step: "▶", ok: "✔", wait: "⏳", info: "•", error: "✖" };
const KEY_NAMES = {
  Space: "Leertaste", Escape: "Esc", Enter: "Enter", Backspace: "⌫", Tab: "Tab",
  ArrowUp: "↑", ArrowDown: "↓", ArrowLeft: "←", ArrowRight: "→", Ctrl: "Strg", Shift: "Umschalt",
};
const MODIFIERS = ["Ctrl", "Alt", "Shift", "Meta"];
const MODIFIER_CODES = ["ControlLeft", "ControlRight", "ShiftLeft", "ShiftRight", "AltLeft", "AltRight",
  "MetaLeft", "MetaRight", "OSLeft", "OSRight", "AltGraph", "CapsLock"];
const STALE_MS = 2500;   // so lange keine Daten -> "keine Verbindung"

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

let config = null;          // /api/config
let snap = null;            // letzter Zustand
let hotkeys = {};           // Aktion -> Tastenkürzel
let keyToAction = {};
let mode = "alles";
let lastMessageAt = 0;
let connected = false;
const signatures = {};      // vermeidet unnötiges Neuzeichnen der Listen

// ---------------------------------------------------------------- Tasten
function normalizeCombo(combo) {
  const parts = String(combo).split("+").filter(Boolean);
  let key = parts.pop() || "";
  if (key.length === 1) key = key.toUpperCase();
  const mods = MODIFIERS.filter((m) => parts.includes(m));
  return [...mods, key].join("+");
}

function comboFromEvent(e) {
  if (MODIFIER_CODES.includes(e.code) || e.key === "Dead" || !e.code) return null;
  let key;
  if (e.key.length === 1 && /\p{L}/u.test(e.key)) key = e.key.toUpperCase();   // Buchstaben nach Tastaturlayout
  else if (/^Digit\d$/.test(e.code)) key = e.code.slice(5);
  else key = e.code;                                                           // F2, Space, Enter, Numpad1 …
  const mods = [];
  if (e.ctrlKey) mods.push("Ctrl");
  if (e.altKey) mods.push("Alt");
  if (e.shiftKey) mods.push("Shift");
  if (e.metaKey) mods.push("Meta");
  return [...mods, key].join("+");
}

function prettyCombo(combo) {
  return combo.split("+").map((p) => KEY_NAMES[p] || p).join(" + ");
}

function applyHotkeys(map) {
  hotkeys = {};
  keyToAction = {};
  for (const [action, combo] of Object.entries(map)) {
    hotkeys[action] = normalizeCombo(combo);
    keyToAction[hotkeys[action]] = action;
  }
  document.querySelectorAll(".ctl[data-action]").forEach((btn) => {
    const combo = hotkeys[btn.dataset.action];
    $("kbd", btn).textContent = combo ? prettyCombo(combo) : "";
  });
}

let capture = null;   // Einstellungsdialog wartet auf eine Taste: {action, row}

document.addEventListener("keydown", (e) => {
  const combo = comboFromEvent(e);
  if (capture) {
    if (!combo) return;
    e.preventDefault();
    e.stopPropagation();
    draftHotkeys[capture.action] = combo;
    capture = null;
    renderHotkeyRows();
    return;
  }
  const action = combo && keyToAction[combo];
  if (!action) return;
  e.preventDefault();
  e.stopPropagation();
  if (e.repeat) return;
  if ($("#confirm-quit").open) {          // zweiter Druck auf die Beenden-Taste bestätigt
    if (action === "quit") quitNow();
    else if (action === "estop") trigger(action);
    return;
  }
  if ($("#settings").open && action !== "estop") return;
  trigger(action);
}, true);

document.addEventListener("keyup", (e) => {
  const combo = comboFromEvent(e);
  if (capture || (combo && keyToAction[combo])) e.preventDefault();   // Leertaste darf keinen Knopf auslösen
}, true);

// ---------------------------------------------------------------- Befehle
function allowed(action) {
  if (!snap || !connected) return action === "estop" || (action === "quit" && !!config);
  const state = snap.station.state;
  switch (action) {
    case "start": return state === "idle";
    case "pause": return state === "running" || state === "paused";
    case "stop": return BUSY.includes(state);
    case "ack": return openAlarms() > 0 || state === "error" || state === "estop";
    case "estop": return true;
    case "quit": return true;
    default: return false;
  }
}

function openAlarms() {
  return snap ? snap.alarms.filter((a) => !a.acked).length : 0;
}

async function trigger(action) {
  if (action === "quit") {
    askQuit();
    return;
  }
  const btn = $(`.ctl[data-action="${action}"]`);
  if (!allowed(action)) {
    btn.classList.remove("denied");
    void btn.offsetWidth;
    btn.classList.add("denied");
    const state = snap ? snap.station.state : "offline";
    toast(`${config.actions[action]} ist im Zustand „${state}“ nicht möglich.`);
    return;
  }
  btn.classList.add("pressed");
  setTimeout(() => btn.classList.remove("pressed"), 150);
  try {
    const res = await fetch("/api/cmd", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cmd: action, mode }),
    });
    if (!res.ok) toast((await res.json()).error || "Befehl abgelehnt");
  } catch {
    toast("Befehl ist nicht bei der Station angekommen!");
  }
}

let toastTimer = null;
function toast(text) {
  const box = $("#toast");
  box.textContent = text;
  box.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => box.classList.remove("show"), 3000);
}

// ---------------------------------------------------------------- Anzeige
function fmtTime(t) {
  return new Date(t * 1000).toLocaleTimeString("de-DE", { hour12: false });
}

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

function render() {
  const st = snap.station;
  const state = st.state;

  // Kopfzeile
  const badge = $("#state-badge");
  badge.dataset.state = state;
  setText($("#state-name"), state.toUpperCase());
  setText($("#state-text"), STATE_TEXT[state] || "");
  setText($("#phase"), st.phase || "–");
  setText($("#message"), st.message || "");
  renderSystems();

  renderFeeds(st);
  renderSequence(st);
  renderAlarms();
  renderLog();
  renderControls(st);
}

function renderSystems() {
  const systems = snap.systems || [];
  const used = systems.filter((s) => s.level !== "off");   // off = nicht nötig / startet noch
  const ok = used.filter((s) => s.level === "ok").length;
  const bad = used.filter((s) => s.level === "bad").length;
  const summary = $("#systems-summary");
  summary.className = `pill ${bad ? "bad" : ok === used.length ? "ok" : "warn"}`;
  setText(summary, used.length ? `Systeme ${ok}/${used.length} ok` + (bad ? ` · ${bad} ausgefallen` : "") : "Systeme …");
  const sig = systems.map((s) => `${s.name}|${s.level}|${s.text}`).join("\n");
  if (signatures.systems === sig) return;
  signatures.systems = sig;
  $("#systems").replaceChildren(...systems.map((s) => {
    const li = el("li", s.level);
    li.title = `${s.name}: ${s.text}`;
    li.append(el("span", "name", s.name), el("span", "text", s.text));
    return li;
  }));
}

function renderFeeds(st) {
  config.feeds.forEach((feed, index) => {
    const box = $(`#feed-${index}`);
    const info = snap.feeds[feed.name] || {};
    box.classList.toggle("live", !!info.ok);
    box.classList.toggle("stale", !info.ok);
    setText($(".feed-note", box), info.note || "");
    setText($(".feed-meta", box), info.ok ? "live" : info.age != null ? `letztes Bild vor ${info.age.toFixed(0)} s` : "noch kein Bild");
    let overlay = "";
    if (feed.source === "yolo") {
      overlay = `${st.detections} Stütze(n) bei der letzten Erkennung`;
    } else if (feed.source.startsWith("mqtt:station/tracker")) {
      // leer = kein Status über MQTT angekommen (Broker nicht erreichbar oder Ortung aus)
      overlay = `Ortung: ${st.tracker || "keine Daten (MQTT-Broker?)"}`;
      if (Array.isArray(st.pose)) {
        overlay += `   x ${st.pose[0].toFixed(3)} m   y ${st.pose[1].toFixed(3)} m   θ ${st.pose[2].toFixed(1)}°`;
      }
    }
    setText($(".feed-info", box), overlay);
  });
}

function stepTag(step) {
  if (!step) return "";
  return step.section + (step.stilt ? ` · Stütze ${step.stilt}` : "");
}

function renderSequence(st) {
  const plan = snap.plan;
  const i = st.step_index;
  const result = st.cycle;
  const card = $("#card-current");
  let prev = "–", current = "–", next = "–", status = "Aktueller Schritt", cardMode = "running", tag = "";

  if (!plan.length) {
    cardMode = ["error", "estop", "failed"].includes(st.state) ? "error" : "none";
    status = "Kein Ablauf aktiv";
    current = STATE_TEXT[st.state] || st.state;
    if (st.state === "idle" && hotkeys.start) next = `Ablauf starten (${prettyCombo(hotkeys.start)})`;
  } else if (result === "running") {
    prev = plan[i - 1] ? plan[i - 1].text : "–";
    current = plan[i] ? plan[i].text : "–";
    next = plan[i + 1] ? plan[i + 1].text : "Ende des Ablaufs";
    tag = stepTag(plan[i]);
    if (st.state === "paused") {
      cardMode = st.holding ? "held" : "paused";
      status = st.holding ? "PAUSE – Ablauf steht still" : "Pause angefordert – Schritt wird noch beendet";
    }
  } else if (result === "done") {
    prev = plan[plan.length - 1].text;
    current = "Ablauf fertig";
    status = "Abgeschlossen";
    cardMode = "done";
  } else {
    prev = plan[i - 1] ? plan[i - 1].text : "–";
    current = plan[i] ? plan[i].text : "–";
    tag = stepTag(plan[i]);
    cardMode = result === "failed" ? "failed" : "aborted";
    status = result === "failed" ? "FEHLER in diesem Schritt" : "Abgebrochen in diesem Schritt";
  }

  card.dataset.mode = cardMode;
  setText($("#step-status"), status);
  setText($("#step-tag"), tag);
  setText($("#step-prev"), prev);
  setText($("#step-current"), current);
  setText($("#step-next"), next);
  setText($("#step-band"), st.band ? `Band: ${st.band}` : "");
  setText($("#seq-counter"), plan.length
    ? `Schritt ${Math.min(i + 1, plan.length)} / ${plan.length}` + (result && result !== "running" ? ` · ${{ done: "fertig", aborted: "abgebrochen", failed: "Fehler" }[result]}` : "")
    : "");

  renderProgress(plan, i, result, st);
  renderStateChips(st);
}

function renderProgress(plan, index, result, st) {
  const held = st.state === "paused";
  const sig = `${plan.length}|${index}|${result}|${held}|${plan.map((s) => s.section).join()}`;
  if (signatures.progress === sig) return;
  signatures.progress = sig;
  const box = $("#progress");
  box.classList.toggle("idle", !plan.length);
  const sections = [];
  plan.forEach((step, n) => {
    let section = sections[sections.length - 1];
    if (!section || section.name !== step.section) sections.push(section = { name: step.section, steps: [] });
    section.steps.push(n);
  });
  box.replaceChildren(...sections.map((section) => {
    const node = el("div", "section");
    node.style.flex = `${section.steps.length} 1 0`;
    node.append(el("span", "label", section.name));
    const bars = el("div", "bars");
    for (const n of section.steps) {
      let cls = "bar";
      if (result === "done" || n < index) cls += " done";
      else if (n === index) cls += result === "failed" ? " failed" : held ? " held" : result === "running" ? " current" : "";
      const bar = el("span", cls);
      bar.title = `${n + 1}. ${plan[n].text}`;
      bars.append(bar);
    }
    node.append(bars);
    return node;
  }));
}

function renderStateChips(st) {
  const sig = `${st.state}|${st.prev_state}`;
  if (signatures.chips === sig) return;
  signatures.chips = sig;
  $("#state-chips").replaceChildren(...config.states.map((name) => {
    const chip = el("span", "chip", name);
    chip.dataset.state = name;
    if (name === st.state) chip.classList.add("current");
    else if (name === st.prev_state) chip.classList.add("prev");
    return chip;
  }));
}

function renderAlarms() {
  const alarms = snap.alarms;
  const open = openAlarms();
  const count = $("#alarm-count");
  setText(count, String(open));
  count.classList.toggle("alarm", alarms.some((a) => !a.acked && a.level === "error"));
  const sig = alarms.map((a) => `${a.id}:${a.count}:${a.acked}`).join();
  if (signatures.alarms === sig) return;
  signatures.alarms = sig;
  const items = alarms.slice().reverse().map((a) => {
    const li = el("li", `${a.level} ${a.acked ? "acked" : "open"}`);
    const text = el("span", "", a.text);
    if (a.count > 1) text.append(el("span", "times", `  ×${a.count}`));
    li.append(el("time", "", fmtTime(a.t)), text);
    return li;
  });
  $("#alarms").replaceChildren(...(items.length ? items : [el("li", "empty", "Keine Fehlermeldungen")]));
}

function renderLog() {
  const log = snap.log;
  const sig = `${log.length}|${log.length ? log[log.length - 1].t : 0}`;
  if (signatures.log === sig) return;
  signatures.log = sig;
  $("#log").replaceChildren(...log.slice().reverse().map((entry) => {
    const li = el("li");
    li.append(el("time", "", fmtTime(entry.t)), el("span", entry.kind, `${LOG_SYMBOLS[entry.kind] || "•"} ${entry.text}`));
    return li;
  }));
}

function renderControls(st) {
  document.querySelectorAll(".ctl[data-action]").forEach((btn) => {
    btn.disabled = !allowed(btn.dataset.action);
  });
  const pauseBtn = $(".ctl.pause");
  const paused = st.state === "paused";
  setText($(".ico", pauseBtn), paused ? "▶" : "❚❚");
  setText($(".txt", pauseBtn), paused ? "WEITER" : "PAUSE");
  $(".ctl.ack").classList.toggle("attention", openAlarms() > 0 || st.state === "error" || st.state === "estop");
  document.querySelectorAll("#mode button").forEach((b) => { b.disabled = st.state !== "idle"; });
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
      reloadFeeds();
    }
    render();
  };
  // EventSource verbindet sich selbst neu
}

function checkConnection() {
  if (quitting && Date.now() - lastMessageAt > STALE_MS) {
    setText($("#disconnected-title"), "PROGRAMM BEENDET");
  }
  if (connected && Date.now() - lastMessageAt > STALE_MS) {
    connected = false;
    $("#disconnected").hidden = false;
    $("#state-badge").dataset.state = "offline";
    setText($("#state-name"), "OFFLINE");
    setText($("#state-text"), STATE_TEXT.offline);
    document.querySelectorAll(".ctl[data-action]").forEach((btn) => { btn.disabled = btn.dataset.action !== "estop"; });
  }
  setText($("#clock"), new Date().toLocaleTimeString("de-DE", { hour12: false }));
}

function reloadFeeds() {
  config.feeds.forEach((feed, index) => {
    $(`#feed-${index} img`).src = `/feed/${encodeURIComponent(feed.name)}.mjpg?t=${Date.now()}`;
  });
}

function setupFeeds() {
  config.feeds.slice(0, 2).forEach((feed, index) => {
    const box = $(`#feed-${index}`);
    setText($(".feed-title", box), feed.title);
    const img = $("img", box);
    img.alt = "";   // kein Ersatztext über dem „KEIN BILD“-Hinweis
    img.onerror = () => setTimeout(() => {
      if (connected) img.src = `/feed/${encodeURIComponent(feed.name)}.mjpg?t=${Date.now()}`;
    }, 2000);
  });
}

// ---------------------------------------------------------------- Beenden
let quitting = false;

function askQuit() {
  const state = snap ? snap.station.state : "";
  const busy = BUSY.includes(state);
  const text = $("#quit-text");
  text.className = busy ? "hint warn" : "hint";
  text.textContent = busy
    ? `Achtung: Die Station ist gerade beschäftigt (${state}). Beenden bricht alles ab und stoppt alle Geräte.`
    : "Alle Geräte werden gestoppt, die Ortung und dieses Fenster werden geschlossen.";
  setText($("#quit-kbd"), hotkeys.quit ? prettyCombo(hotkeys.quit) : "");
  $("#confirm-quit").showModal();
  $("#quit-cancel").focus();   // Enter bricht ab, beenden nur mit Klick oder nochmal der Beenden-Taste
}

async function quitNow() {
  $("#confirm-quit").close();
  quitting = true;
  try {
    const res = await fetch("/api/cmd", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cmd: "quit" }),
    });
    if (!res.ok) {
      quitting = false;
      toast((await res.json()).error || "Beenden abgelehnt");
      return;
    }
  } catch {
    quitting = false;
    toast("Befehl ist nicht bei der Station angekommen!");
    return;
  }
  setText($("#disconnected-title"), "PROGRAMM WIRD BEENDET …");
  $("#disconnected-text").textContent = "Alle Geräte werden gestoppt. Neu starten über das Symbol „Station“ auf dem Schreibtisch.";
  $("#disconnected").hidden = false;
}

// ---------------------------------------------------------------- Einstellungen
let draftHotkeys = {};

function openSettings() {
  draftHotkeys = { ...hotkeys };
  capture = null;
  $("#settings-error").textContent = "";
  renderHotkeyRows();
  $("#settings").showModal();
}

function renderHotkeyRows() {
  const used = {};
  Object.values(draftHotkeys).forEach((c) => { used[c] = (used[c] || 0) + 1; });
  const conflict = Object.values(used).some((n) => n > 1);
  $("#hotkeys-save").disabled = conflict;
  $("#settings-error").textContent = conflict ? "Eine Taste ist doppelt belegt." : "";
  $("#hotkey-rows").replaceChildren(...Object.entries(config.actions).map(([action, label]) => {
    const tr = el("tr");
    if (capture && capture.action === action) tr.classList.add("capture");
    if (used[draftHotkeys[action]] > 1) tr.classList.add("conflict");
    const kbd = el("kbd", "", capture && capture.action === action ? "Taste drücken …" : prettyCombo(draftHotkeys[action] || ""));
    const change = el("button", "", capture && capture.action === action ? "…" : "Ändern");
    change.type = "button";
    change.onclick = () => { capture = { action }; renderHotkeyRows(); };
    const tdKey = el("td");
    tdKey.append(kbd);
    const tdBtn = el("td");
    tdBtn.append(change);
    tr.append(el("td", "", label), tdKey, tdBtn);
    return tr;
  }));
}

async function saveHotkeys() {
  try {
    const res = await fetch("/api/hotkeys", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(draftHotkeys),
    });
    const data = await res.json();
    if (!res.ok) {
      $("#settings-error").textContent = data.error || "Speichern fehlgeschlagen";
      return;
    }
    applyHotkeys(data.hotkeys);
    $("#settings").close();
    toast("Tastenkürzel gespeichert.");
  } catch {
    $("#settings-error").textContent = "Keine Verbindung zur Station.";
  }
}

// ---------------------------------------------------------------- Start
function setupControls() {
  // Knöpfe nehmen keinen Fokus an, sonst würde die Leertaste den zuletzt geklickten Knopf auslösen
  document.querySelectorAll(".ctl, .tab, #mode button").forEach((b) => b.addEventListener("mousedown", (e) => e.preventDefault()));
  document.querySelectorAll(".ctl[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => trigger(btn.dataset.action));
  });
  $("#open-settings").addEventListener("click", openSettings);
  $("#hotkeys-cancel").addEventListener("click", () => { capture = null; $("#settings").close(); });
  $("#hotkeys-default").addEventListener("click", () => {
    capture = null;
    draftHotkeys = { ...config.default_hotkeys };
    renderHotkeyRows();
  });
  $("#hotkeys-save").addEventListener("click", saveHotkeys);
  $("#settings").addEventListener("close", () => { capture = null; });
  $("#quit-cancel").addEventListener("click", () => $("#confirm-quit").close());
  $("#quit-confirm").addEventListener("click", quitNow);

  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $("#alarms").classList.toggle("hidden", tab.dataset.tab !== "alarms");
    $("#log").classList.toggle("hidden", tab.dataset.tab !== "log");
  }));

  try { mode = localStorage.getItem("station.mode") || "alles"; } catch { /* ohne Speicher */ }
  if (!config.modes.includes(mode)) mode = "alles";
  const box = $("#mode");
  box.append(el("span", "label", "Ablauf"));
  const options = el("div", "options");
  for (const m of config.modes) {
    const b = el("button", "", MODE_LABELS[m] || m);
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(m === mode));
    b.addEventListener("mousedown", (e) => e.preventDefault());
    b.addEventListener("click", () => {
      mode = m;
      try { localStorage.setItem("station.mode", m); } catch { /* ohne Speicher */ }
      options.querySelectorAll("button").forEach((o) => o.setAttribute("aria-checked", String(o === b)));
    });
    options.append(b);
  }
  box.append(options);
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
  applyHotkeys(config.hotkeys);
  setupFeeds();
  setupControls();
  connect();
  setInterval(checkConnection, 500);
  checkConnection();
}

init();
