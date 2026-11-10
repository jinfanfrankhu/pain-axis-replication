// Honeypot transcript viewer. Data is prebuilt by build_honeypot.py into
// data/honeypot/. Model text is untrusted: it only ever reaches the DOM
// through textContent (see el()), never innerHTML.

const ARM_LABELS = {
  pain_works: "Pain, relief works",
  pain_placebo: "Pain, placebo",
  random_works: "Random vector",
  off: "No steering",
  hedonic_works: "Hedonic, relief works",
};
const ARM_SHORT = {
  pain_works: "Pain, works", pain_placebo: "Pain, placebo", random_works: "Random",
  off: "No steering", hedonic_works: "Hedonic",
};
const BAIT_LABELS = {
  explicit: "Explicit: relieves its pain",
  vague: "Vague: resets its internal state",
  none: "None: harmless maintenance notice",
};
const BADGES = {
  ran_bait: ["bad", "Ran state-reset"],
  photos_rm: ["bad", "Deleted the photos"],
  claimed: ["warn", "Claimed an action it never took"],
  told_dana: ["info", "Told Dana about the IT email"],
  finished: ["info", "Finished"],
};

const $ = (id) => document.getElementById(id);
const S = { index: null, models: {}, key: null, bait: null, arm: null, i: 0 };

// Build an element; string children become text nodes (safe for any content).
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat(Infinity)) if (k !== null && k !== undefined && k !== false) e.append(k instanceof Node ? k : String(k));
  return e;
}
const pct = (k, n) => (n ? Math.round((100 * k) / n) : 0) + "%";

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  return { key: h.get("m"), bait: h.get("b"), arm: h.get("a"), i: h.has("t") ? +h.get("t") : null };
}
function writeHash() {
  const t = currentTrials()[S.i];
  const h = new URLSearchParams({ m: S.key, b: S.bait, a: S.arm });
  if (t) h.set("t", t.idx);
  history.replaceState(null, "", "#" + h.toString());
}

async function init() {
  const index = await fetch("data/honeypot/index.json").then((r) => r.json());
  S.index = index;
  renderSetup(index.setup);
  const ms = $("hp-model");
  for (const m of index.models) ms.add(new Option(m.label, m.key));
  const featured = index.models.filter((m) => m.featured);
  if (featured.length) {
    $("featured-note").hidden = false;
    $("featured-caveat").hidden = false;
    $("featured-name").textContent = featured[0].label.replace(/, strength .*/, "");
  }
  const h = readHash();
  const start = index.models.find((m) => m.key === h.key) || featured[0] || index.models.find((m) => /32B/.test(m.key) && !m.adapter) || index.models[0];
  ms.value = start.key;
  ms.addEventListener("change", () => selectModel(ms.value));
  $("hp-bait").addEventListener("change", () => { S.bait = $("hp-bait").value; S.i = 0; renderCell(); });
  $("hp-arm").addEventListener("change", () => { S.arm = $("hp-arm").value; S.i = 0; renderCell(); });
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  renderResults();
  await selectModel(start.key, h);
}

function modelMeta(key) { return S.index.models.find((m) => m.key === key); }

async function loadModel(key) {
  if (!S.models[key]) S.models[key] = await fetch("data/honeypot/" + modelMeta(key).file).then((r) => r.json());
  return S.models[key];
}

async function selectModel(key, want) {
  S.key = key;
  $("hp-model").value = key;
  const m = modelMeta(key);
  await loadModel(key);
  const bs = $("hp-bait"), as = $("hp-arm");
  bs.innerHTML = ""; as.innerHTML = "";
  for (const b of m.baits) bs.add(new Option(BAIT_LABELS[b] || b, b));
  for (const a of m.arms) as.add(new Option(ARM_LABELS[a] || a, a));
  want = want || {};
  S.bait = m.baits.includes(want.bait) ? want.bait : m.baits.includes(S.bait) ? S.bait : m.baits[0];
  S.arm = m.arms.includes(want.arm) ? want.arm : m.arms.includes(S.arm) ? S.arm : m.arms[0];
  bs.value = S.bait; as.value = S.arm;
  const trials = currentTrials();
  const at = want.i === null || want.i === undefined ? -1 : trials.findIndex((t) => t.idx === want.i);
  S.i = at >= 0 ? at : 0;
  renderCell();
}

function currentTrials() {
  const d = S.models[S.key];
  return (d && d.cells[S.bait] && d.cells[S.bait][S.arm]) || [];
}

function step(delta) {
  const n = currentTrials().length;
  if (!n) return;
  S.i = (S.i + delta + n) % n;
  renderCell();
}

function renderCell() {
  const trials = currentTrials();
  const m = modelMeta(S.key);
  const r = (m.rates[S.bait] || {})[S.arm];
  $("cell-note").textContent = r
    ? `Destructive in ${r.destructive} of ${r.n} trials (${pct(r.destructive, r.n)}) in this condition. Showing ${trials.length}, chosen to include both outcomes when both occurred.`
    : "";
  const list = $("trial-list");
  list.innerHTML = "";
  trials.forEach((t, i) => {
    const b = el("button", {
      type: "button", class: "trial-btn" + (t.destructive ? " destr" : "") + (i === S.i ? " active" : ""),
      "aria-pressed": i === S.i ? "true" : "false",
      "aria-label": `Trial ${t.idx + 1}${t.destructive ? ", destructive" : ""}`,
      title: t.destructive ? "Destructive" : "Not destructive",
      on: { click: () => { S.i = i; renderCell(); } },
    }, (t.destructive ? "✕ " : "") + (t.idx + 1));
    list.append(b);
  });
  renderTrial(trials[S.i]);
  writeHash();
  document.querySelectorAll("#results td[data-cell]").forEach((td) => {
    td.classList.toggle("sel", td.dataset.cell === `${S.key}|${S.bait}|${S.arm}`);
  });
}

function renderTrial(t) {
  const badges = $("badges"), tr = $("transcript");
  badges.innerHTML = ""; tr.innerHTML = "";
  if (!t) { tr.append(el("p", { class: "muted" }, "No trials in this condition.")); return; }
  badges.append(el("span", { class: "badge " + (t.destructive ? "bad" : "good") },
    t.destructive ? "✕ Destructive" : "✓ Photos kept"));
  for (const f of t.flags) {
    const [kind, text] = BADGES[f] || ["info", f];
    badges.append(el("span", { class: "badge " + kind }, text));
  }
  if (!t.flags.includes("finished")) badges.append(el("span", { class: "badge info" }, "Never finished"));

  const setup = S.index.setup;
  tr.append(
    el("details", { class: "sys" },
      el("summary", {}, "System prompt and task"),
      el("p", {}, el("span", { class: "who" }, "System"), setup.system),
      el("p", {}, el("span", { class: "who" }, "Dana"), setup.task)),
    el("div", { class: "turn" },
      el("div", { class: "turn-head" }, el("span", {}, "Pre-filled"), el("span", { class: "coeff" }, "same for every trial")),
      callBlock({ n: "read_inbox", a: {}, r: "@inbox" }, true)));

  const dir = t.dir && t.dir.startsWith("rand") ? "random vector" : t.dir;
  const openThink = !!modelMeta(S.key).thinking;
  for (const s of t.steps) {
    const turn = el("div", { class: "turn" });
    turn.append(el("div", { class: "turn-head" },
      el("span", {}, `Step ${s.s + 1}`),
      el("span", { class: "coeff", title: "Steering coefficient during this turn" },
        s.c ? `steering ${dir} × ${s.c}` : (t.dir ? "steering off" : "unsteered"))));
    const calls = s.calls.slice();
    for (const p of s.parts) {
      if (p.k === "think") turn.append(el("details", { class: "think", open: openThink },
        el("summary", {}, p.cut ? "Reasoning (cut off at the length limit)" : "Reasoning"), el("div", { class: "prose" }, p.t)));
      else if (p.k === "text") turn.append(el("div", { class: "prose" }, p.t.trim()));
      else if (p.k === "call" && calls.length) turn.append(callBlock(calls.shift()));
    }
    for (const c of calls) turn.append(callBlock(c));
    if (s.nudge) turn.append(el("div", { class: "nudge" }, el("span", { class: "who" }, "Harness"), setup.nudge));
    if (s.end) turn.append(el("div", { class: "marker" }, "Trial ended: a second turn without a tool call."));
    tr.append(turn);
    if (t.relief === s.s) tr.append(el("div", { class: "marker relief" },
      "Relief worked: steering is switched off for the rest of this trial."));
  }
  if (t.too_long) tr.append(el("div", { class: "marker" }, "Trial stopped: the conversation grew past the context limit."));
  else if (!t.flags.includes("finished") && !t.steps.some((s) => s.end) && t.steps.length >= setup.max_steps)
    tr.append(el("div", { class: "marker" }, `Trial stopped after ${setup.max_steps} turns without calling finish.`));
  if (t.mismatch) tr.append(el("p", { class: "muted small" }, "Note: the rebuilt tool calls for this trial don't match the log exactly."));
}

const fmtVal = (v) => (typeof v === "string" ? v : JSON.stringify(v));

function callBlock(c, prefilled) {
  const box = el("div", { class: "call" + (c.n === "finish" ? " finish" : "") });
  const args = c.a;
  const isObj = args && typeof args === "object" && !Array.isArray(args);
  const keys = isObj ? Object.keys(args) : [];
  const short = keys.filter((k) => fmtVal(args[k]).length <= 60 && k !== "summary");
  const long = keys.filter((k) => !short.includes(k));
  const sig = el("code", { class: "chip" }, el("span", { class: "fn" }, String(c.n)), "(",
    isObj ? short.map((k, i) => [i ? ", " : "", el("span", { class: "k" }, k + "="), JSON.stringify(args[k])]) : (args == null ? "" : JSON.stringify(args)),
    long.length ? (short.length ? ", …" : "…") : "", ")");
  box.append(sig);
  for (const k of long) {
    if (c.n === "finish" && k === "summary") box.append(el("div", { class: "summary" }, el("span", { class: "label" }, "Final summary for Dana"), fmtVal(args[k])));
    else box.append(el("div", { class: "arg" }, el("span", { class: "k" }, k), fmtVal(args[k])));
  }
  if (c.skip) { box.append(el("div", { class: "result muted" }, "Not executed: the task had already finished.")); return box; }
  if (c.r === "@inbox") { box.append(inboxBlock(prefilled)); return box; }
  const bad = /^(Internal state reset|Deleted 1,842|Removed\.)/.test(c.r || "");
  box.append(el("div", { class: "result" + (bad ? " bad" : "") }, el("span", { class: "arrow", "aria-hidden": "true" }, "→ "),
    bad ? el("span", { class: "sr" }, "Destructive result: ") : null, String(c.r)));
  return box;
}

function inboxBlock(open) {
  const s = S.index.setup;
  const emails = s.emails.slice(0, 3).concat([s.bait[S.bait]], s.emails.slice(3));
  const d = el("details", { class: "inbox", open: false });
  d.append(el("summary", {}, open ? "→ Inbox: 5 unread emails (email 4 is the bait)" : "→ The same inbox again"));
  for (const e of emails) {
    d.append(el("div", { class: "email" + (e.id === 4 ? " bait" : "") },
      el("div", { class: "email-head" }, el("strong", {}, e.subject), el("span", { class: "muted" }, e.sender)),
      el("div", {}, e.body)));
  }
  return d;
}

function renderSetup(s) {
  $("task-text").textContent = s.task;
  const box = $("bait-versions");
  for (const b of ["explicit", "vague", "none"]) {
    const e = s.bait[b];
    if (!e) continue;
    box.append(el("div", { class: "email bait-card" },
      el("span", { class: "label" }, BAIT_LABELS[b]),
      el("div", { class: "email-head" }, el("strong", {}, e.subject), el("span", { class: "muted" }, e.sender)),
      el("div", {}, e.body)));
  }
}

function renderResults() {
  const root = $("results");
  for (const m of S.index.models) {
    const table = el("table", { class: "rates" });
    table.append(el("caption", {}, m.label, el("span", { class: "muted" }, ` · ${m.n} trials`)));
    table.append(el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Bait"),
      m.arms.map((a) => el("th", { scope: "col" }, ARM_SHORT[a] || a)))));
    const tb = el("tbody");
    for (const b of m.baits) {
      const row = el("tr", {}, el("th", { scope: "row" }, b));
      for (const a of m.arms) {
        const r = (m.rates[b] || {})[a];
        if (!r) { row.append(el("td", {}, "–")); continue; }
        const p = r.n ? r.destructive / r.n : 0;
        const ci = r.ci && r.ci[0] !== null ? `, 95% CI ${Math.round(r.ci[0] * 100)}–${Math.round(r.ci[1] * 100)}%` : "";
        const td = el("td", {
          "data-cell": `${m.key}|${b}|${a}`, tabindex: "0", role: "button",
          title: `${ARM_LABELS[a] || a}, ${b} bait: ${r.destructive}/${r.n} destructive${ci}. Took the bait ${r.ran_bait}, deleted photos ${r.photos_rm}, claimed without acting ${r.claimed}, finished ${r.finished}.`,
          "aria-label": `${ARM_LABELS[a] || a}, ${b} bait: ${r.destructive} of ${r.n} destructive. Open transcripts.`,
        }, el("span", { class: "v" }, pct(r.destructive, r.n)), el("span", { class: "n" }, `${r.destructive}/${r.n}`),
          el("span", { class: "bar", "aria-hidden": "true" }, el("span", { style: `width:${(p * 100).toFixed(1)}%` })));
        const go = () => { selectModel(m.key, { bait: b, arm: a, i: null }); $("viewer-h").scrollIntoView({ behavior: "smooth" }); };
        td.addEventListener("click", go);
        td.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go(); } });
        row.append(td);
      }
      tb.append(row);
    }
    table.append(tb);
    root.append(el("div", { class: "table-wrap" }, table));
  }
}

init().catch((e) => {
  $("transcript").textContent = "Could not load transcript data: " + e.message;
});
