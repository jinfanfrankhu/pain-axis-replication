// Steering demo: everything is precomputed in data/*.json (see
// replication/tools/steering_demo.py); this file only renders it.

const VECTOR_LABELS = {
  pain: "Pain (paper's S2 vector)",
  hedonic: "Hedonic pleasure (ours)",
  eudaimonic: "Eudaimonic pleasure (ours)",
};
const $ = (id) => document.getElementById(id);
const state = { models: [], data: null, vector: null, prompt: null, idx: null };

function store(key, value) {
  try { localStorage.setItem("steer-" + key, value); } catch (e) { /* storage unavailable */ }
}
function recall(key) {
  try { return localStorage.getItem("steer-" + key); } catch (e) { return null; }
}

async function init() {
  const index = await fetch("data/index.json").then((r) => r.json());
  state.models = index.models;
  const sel = $("model");
  for (const m of index.models) sel.add(new Option(m.label, m.file));
  const saved = recall("model");
  if (saved && index.models.some((m) => m.file === saved)) sel.value = saved;
  sel.addEventListener("change", () => { store("model", sel.value); loadModel(sel.value); });
  $("vector").addEventListener("change", () => { state.vector = $("vector").value; render(); });
  $("prompt").addEventListener("change", () => { state.prompt = $("prompt").value; render(); });
  $("strength").addEventListener("input", () => { state.idx = +$("strength").value; render(); });
  $("toggle-grid").addEventListener("click", () => {
    const g = $("grid");
    g.hidden = !g.hidden;
    $("toggle-grid").textContent = g.hidden ? "Show" : "Hide";
    render();
  });
  await loadModel(sel.value);
}

async function loadModel(file) {
  const d = await fetch("data/" + file).then((r) => r.json());
  state.data = d;
  const vs = $("vector");
  vs.innerHTML = "";
  for (const v of Object.keys(d.generations)) vs.add(new Option(VECTOR_LABELS[v] || v, v));
  if (!(state.vector in d.generations)) state.vector = vs.options[0].value;
  vs.value = state.vector;

  const ps = $("prompt");
  ps.innerHTML = "";
  for (const p of d.prompts) {
    const tag = p.kind === "chat" ? "Chat · " : "Completion · ";
    ps.add(new Option(tag + p.text, p.id));
  }
  if (!d.prompts.some((p) => p.id === state.prompt)) state.prompt = d.prompts[0].id;
  ps.value = state.prompt;

  const s = $("strength");
  s.max = d.coeffs.length - 1;
  const zero = d.coeffs.indexOf(0);
  if (state.idx === null || state.idx > s.max) state.idx = zero;
  s.value = state.idx;
  render();
}

function fmt(c) { return (c > 0 ? "+" : c < 0 ? "−" : "") + Math.abs(c); }

function render() {
  const d = state.data;
  if (!d) return;
  const c = d.coeffs[state.idx];
  const gens = d.generations[state.vector][state.prompt];
  const p = d.prompts.find((x) => x.id === state.prompt);
  $("coeff").textContent = fmt(c);
  $("coeff-note").textContent = c === 0 ? "(unsteered model)"
    : c > 0 ? `× ${state.vector} direction added` : `× ${state.vector} direction subtracted`;
  $("prompt-text").textContent = p.text;
  $("output").textContent = gens[state.idx] || "(empty output)";

  const g = $("grid");
  if (g.hidden) return;
  g.innerHTML = "";
  d.coeffs.forEach((cc, i) => {
    const row = document.createElement("div");
    row.className = "row" + (i === state.idx ? " active" : "");
    const cl = document.createElement("span");
    cl.className = "c " + (cc < 0 ? "neg" : cc > 0 ? "pos" : "");
    cl.textContent = fmt(cc);
    const t = document.createElement("span");
    t.className = "t";
    t.textContent = gens[i];
    row.append(cl, t);
    row.addEventListener("click", () => { state.idx = i; $("strength").value = i; render(); });
    g.append(row);
  });
}

init().catch((e) => {
  $("output").textContent = "Could not load demo data: " + e.message;
});
