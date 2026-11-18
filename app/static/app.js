'use strict';
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));

let CAT = null, MODEL = null, LAST = null, RIGS = [], AVATARS = new Map(), ARCH_SUPPORT = null;
let searchTimer, evalTimer, evalSeq = 0, VIEW = 'size', SRC = 'hf';

/* ------------------------------------------------------------------ utils */
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtInt = n => (n == null ? '—' : Math.round(n).toLocaleString());
const fmtGiB = g => (g == null ? '—' : (g >= 100 ? g.toFixed(0) : g.toFixed(g < 10 ? 2 : 1)) + ' GiB');
function fmtParams(p) {
  if (!p) return '—';
  if (p >= 1e12) return (p / 1e12).toFixed(2) + 'T';
  if (p >= 1e9) return (p / 1e9).toFixed(p / 1e9 >= 100 ? 0 : 1) + 'B';
  if (p >= 1e6) return (p / 1e6).toFixed(0) + 'M';
  return p.toLocaleString();
}
function fmtBytes(b) {
  if (!b) return '—';
  const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return b.toFixed(b < 10 && i > 1 ? 1 : 0) + ' ' + u[i];
}
const fmtCount = n => n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(0) + 'k' : String(n || 0);
function fmtTok(t) {
  if (!t || !isFinite(t)) return '—';
  return t >= 1000 ? Math.round(t).toLocaleString() : t.toFixed(t < 10 ? 1 : 0);
}
function fmtBig(n) {
  if (!n) return '0';
  if (n >= 1e6) return (n / 1e6).toFixed(2).replace(/\.?0+$/, '') + 'M';
  if (n >= 1e5) return Math.round(n / 1e3) + 'k';
  return n.toLocaleString();
}
// Findings carry a machine tag; say what it is about in words.
const TAG_WORDS = {
  kv: 'the KV cache', moe: 'mixture-of-experts', arch: 'the architecture',
  quant: 'quantization', parallel: 'parallelism', context: 'context length',
  vision: 'image input', access: 'repo access', attention: 'attention backends',
  kernel: 'kernels', driver: 'drivers', memory: 'memory', os: 'the operating system',
  vendor: 'the vendor', format: 'the weight format', cc: 'compute capability',
  offload: 'CPU offload', 'null': 'this setup',
};
function tagWords(tags) {
  const words = [...new Set(tags.map(t => TAG_WORDS[t] || t).filter(Boolean))];
  if (words.length <= 1) return words[0] || 'this setup';
  return words.slice(0, -1).join(', ') + ' and ' + words[words.length - 1];
}
const ago = iso => {
  if (!iso) return '';
  const d = (Date.now() - new Date(iso)) / 86400000;
  if (d < 1) return 'today';
  if (d < 30) return Math.round(d) + 'd ago';
  if (d < 365) return Math.round(d / 30) + 'mo ago';
  return Math.round(d / 365) + 'y ago';
};
// Rewrite the hash without losing the params already in it (view, open, scan).
function setHash(patch) {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of Object.entries(patch)) {
    if (v == null) p.delete(k); else p.set(k, v);
  }
  history.replaceState(null, '', '#' + p.toString().replace(/%2F/gi, '/'));
}

/* ------------------------------------------------------------------ byok */
// Personal HF/AA keys, kept in this browser only. Sent as headers so the
// server uses them for this request instead of its own (if any) - see
// app/main.py's _byok dependency. Never written anywhere but localStorage.
const KEY_HF = 'podfit_hf_token', KEY_AA = 'podfit_aa_key';
const getKey = k => { try { return localStorage.getItem(k) || ''; } catch (e) { return ''; } };
const setKey = (k, v) => { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch (e) { /* storage unavailable */ } };

async function api(path, opts) {
  opts = opts || {};
  const hfTok = getKey(KEY_HF), aaKey = getKey(KEY_AA);
  if (hfTok || aaKey) {
    opts = Object.assign({}, opts, { headers: Object.assign({}, opts.headers,
      hfTok ? { 'X-HF-Token': hfTok } : {}, aaKey ? { 'X-AA-Key': aaKey } : {}) });
  }
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* non-json */ }
    throw new Error(msg);
  }
  return r.json();
}
const ICON = {
  chev: '<svg class="chev" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--faint)" stroke-width="2"><path d="M9 6l6 6-6 6"/></svg>',
  down: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--faint)" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>',
  sep: '<span class="sep"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 6l6 6-6 6"/></svg></span>',
  check: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--ok)" stroke-width="2.2"><path d="M20 6L9 17l-5-5"/></svg>',
  info: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--faint)" stroke-width="1.9"><circle cx="12" cy="12" r="9"/><path d="M12 16v-5M12 8h.01"/></svg>',
  bolt: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="1.8"><path d="M13 2L3 14h8l-1 8 10-12h-8z"/></svg>',
  term: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="1.8"><path d="M4 17l6-6-6-6M12 19h8"/></svg>',
  swap: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="1.8"><path d="M17 2l4 4-4 4M3 6h18M7 22l-4-4 4-4M21 18H3"/></svg>',
  gpu: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="1.8"><rect x="4" y="4" width="16" height="16" rx="0"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>',
  search: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="1.9"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg>',
  cross: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--bad)" stroke-width="2.4" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>',
  tick: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--ok)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>',
  warn: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--warn)" stroke-width="2.2" stroke-linecap="round"><path d="M12 8v5M12 17h.01"/><path d="M10.3 3.9L2 18a2 2 0 001.7 3h16.6A2 2 0 0022 18L13.7 3.9a2 2 0 00-3.4 0z"/></svg>',
};

/* ------------------------------------------------------------------ logos */
const LOGO_HUES = [210, 265, 150, 25, 340, 190, 45, 300];
function logoFallback(org, cls) {
  const h = LOGO_HUES[[...org].reduce((a, c) => a + c.charCodeAt(0), 0) % LOGO_HUES.length];
  return `<span class="logo-fallback ${cls || ''}" style="background:hsl(${h} 55% 62%)">${esc(org.slice(0, 2))}</span>`;
}
function logo(id, cls) {
  const org = (id || '').split('/')[0] || '?';
  const url = AVATARS.get(org);
  if (!url) return logoFallback(org, cls);
  return `<img class="logo-sm ${cls || ''}" src="${esc(url)}" alt="${esc(org)}" loading="lazy"
    onerror="this.outerHTML=window.__lf('${esc(org)}','${esc(cls || '')}')">`;
}
window.__lf = (org, cls) => logoFallback(org, cls);
async function loadAvatars(ids, then) {
  const orgs = [...new Set(ids.map(i => (i || '').split('/')[0]).filter(o => o && !AVATARS.has(o)))];
  if (!orgs.length) { if (then) then(); return; }
  try {
    const r = await api('/api/avatars?orgs=' + encodeURIComponent(orgs.join(',')));
    Object.entries(r.avatars).forEach(([k, v]) => AVATARS.set(k, v));
  } catch (e) { orgs.forEach(o => AVATARS.set(o, null)); }
  if (then) then();
}

/* ------------------------------------------------------------------ boot */
function applyKeyStatus(cat) {
  const hasHf = !!getKey(KEY_HF), hasAa = !!getKey(KEY_AA);
  $('#hf-status').textContent = 'HF ' + (cat.hf_token ? '· token set' : '· anonymous');
  $('#hf-status').title = cat.hf_token ? 'Gated repos resolve.'
    : 'No HF token — gated repos will not resolve.';
  const aa = cat.bench && cat.bench.artificial_analysis.configured;
  $('#aa-status').textContent = 'AA ' + (aa ? '· on' : '· off');
  $('#aa-status').title = aa ? 'Artificial Analysis scores are available.'
    : 'No AA key — only the archived HF leaderboard supplies scores.';
  $('#key_hf_status').textContent = hasHf
    ? (cat.hf_token ? 'Your personal token is active.' : 'Saved, but not recognized — double-check it.')
    : (cat.hf_token ? 'No personal token saved — using the server’s token.'
                    : 'No token — gated repos will not resolve.');
  $('#key_aa_status').textContent = hasAa
    ? (aa ? 'Your personal key is active.' : 'Saved, but not recognized — double-check it.')
    : (aa ? 'No personal key saved — using the server’s key.'
          : 'No key — only the archived HF leaderboard supplies scores.');
}

(async function boot() {
  $('#key_hf').value = getKey(KEY_HF);
  $('#key_aa').value = getKey(KEY_AA);
  CAT = await api('/api/catalog');
  applyKeyStatus(CAT);

  fill('#pipeline', CAT.pipelines.map(p => [p.key, p.label]), '');
  fillGPUs();
  fill('#engine', CAT.engines.map(e => [e.key, e.label]), 'vllm');
  fill('#kv_dtype', CAT.kv_dtypes.map(k => [k.key, k.label]), 'auto');
  fill('#tp', [0, 1, 2, 4, 8, 16, 32, 64].map(n => [n, n ? String(n) : 'auto']), 0);
  fill('#pp', [0, 1, 2, 4, 8].map(n => [n, n ? String(n) : 'auto']), 0);
  FAMILIES = (await api('/api/gpu-families')).families;
  const at = await api('/api/attention');
  $('#attention').innerHTML = '<option value="auto">auto (engine picks)</option>' +
    at.backends.map(b => `<option value="${esc(b.key)}">${esc(b.label)}</option>`).join('');

  syncEngine();
  buildConcurrency();
  await loadRigs();
  wire();
  renderSetupBar();

  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get('src') === 'aa') {
    SRC = 'aa';
    $$('#src-toggle button').forEach(x => x.classList.toggle('on', x.dataset.src === 'aa'));
    $$('.filters')[0].classList.add('hidden');
    $('#q').placeholder = 'Filter Artificial Analysis models — qwen, glm, deepseek…';
  }
  search();

  if (p.get('model')) selectModel(p.get('model'));
  else openSetup(true);
  if (p.get('view')) switchView(p.get('view'));
})().catch(e => document.body.insertAdjacentHTML('afterbegin',
  `<div class="err" style="margin:16px">Failed to start: ${esc(e.message)}</div>`));

function fill(sel, pairs, val) {
  const el = $(sel);
  const keep = sel === '#pipeline' ? '<option value="">All task types</option>' : '';
  el.innerHTML = keep + pairs.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join('');
  if (val !== undefined) el.value = val;
}
function fillGPUs() {
  const groups = {};
  CAT.gpus.forEach(g => {
    const k = g.vendor === 'nvidia' ? 'NVIDIA ' + g.arch : g.vendor.toUpperCase() + ' ' + g.arch;
    (groups[k] = groups[k] || []).push(g);
  });
  $('#gpu').innerHTML = Object.entries(groups).map(([k, gs]) =>
    `<optgroup label="${esc(k)}">` + gs.map(g =>
      `<option value="${g.id}">${esc(g.name)}</option>`).join('') + '</optgroup>').join('');
  $('#gpu').value = 'l20';
}
function gpuOf(id) { return CAT.gpus.find(g => g.id === id); }
let FAMILIES = null;
// Mirrors gpuarch.family_for: newest family whose floor this card clears.
function familyKeyFor(g) {
  if (!FAMILIES || !g) return null;
  let best = null;
  for (const f of FAMILIES) {
    if (f.vendor !== g.vendor) continue;
    if (g.vendor === 'nvidia') { if (g.cc >= f.cc && (!best || f.cc > best.cc)) best = f; }
    else if (f.cc === g.cc) best = f;
  }
  return best ? best.key : null;
}

/* ------------------------------------------------------------------ wiring */
function wire() {
  $('#tabs').addEventListener('click', e => {
    const b = e.target.closest('.tab'); if (b && b.dataset.view) switchView(b.dataset.view);
  });
  $('#btn-setup').addEventListener('click', () => openSetup($('#setup-panel').classList.contains('hidden')));
  $('#setup-close').addEventListener('click', () => openSetup(false));
  $('#ph-open').addEventListener('click', () => openSetup(true));

  $('#btn-keys').addEventListener('click', () => $('#keys-panel').classList.toggle('hidden'));
  $('#keys-close').addEventListener('click', () => $('#keys-panel').classList.add('hidden'));
  $('#keys-save').addEventListener('click', async () => {
    setKey(KEY_HF, $('#key_hf').value.trim());
    setKey(KEY_AA, $('#key_aa').value.trim());
    $('#keys-save').textContent = 'Checking…';
    try {
      CAT = await api('/api/catalog');
      applyKeyStatus(CAT);
    } finally {
      $('#keys-save').textContent = 'Save keys';
    }
  });
  $('#keys-clear').addEventListener('click', async () => {
    setKey(KEY_HF, ''); setKey(KEY_AA, '');
    $('#key_hf').value = ''; $('#key_aa').value = '';
    CAT = await api('/api/catalog');
    applyKeyStatus(CAT);
  });

  $('#q').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(search, 320); });
  $('#src-toggle').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    SRC = b.dataset.src;
    $$('#src-toggle button').forEach(x => x.classList.toggle('on', x.dataset.src === SRC));
    $('#q').placeholder = SRC === 'aa'
      ? 'Filter Artificial Analysis models — qwen, glm, deepseek…'
      : 'Search HuggingFace — qwen3, deepseek, gemma…';
    $$('.filters')[0].classList.toggle('hidden', SRC === 'aa');
    search();
  });
  $('#pipeline').addEventListener('change', search);
  $('#sort').addEventListener('change', search);

  $('#gpu').addEventListener('change', () => { gpuChanged(); changed(); });
  $('#engine').addEventListener('change', () => { syncEngine(); changed(); });
  $('#weight_format').addEventListener('change', e => { e.target.dataset.touched = '1'; syncGguf(); changed(); });

  const rng = $('#context_len_range'), ctx = $('#context_len');
  rng.addEventListener('input', () => { ctx.value = 2 ** (+rng.value); ctx.dataset.touched = '1'; syncCtx(); changed(); });
  ctx.addEventListener('input', () => {
    ctx.dataset.touched = '1';
    rng.value = Math.min(22, Math.max(10, Math.round(Math.log2(Math.max(+ctx.value || 1024, 1024)))));
    syncCtx(); changed();
  });
  $('#concurrency').addEventListener('input', e => {
    const v = e.target.value.trim();
    if (v) { CONC = Math.max(1, +v || 1); $('#conc-label').textContent = CONC.toLocaleString();
             $('#conc-seg').querySelectorAll('button').forEach(b =>
               b.classList.toggle('on', +b.dataset.n === CONC)); changed(); }
  });
  $('#concurrency').addEventListener('blur', buildConcurrency);

  ['#gpus_per_node', '#nodes', '#host_ram_gb', '#os', '#interconnect', '#driver_version',
    '#vram_override_gb', '#kv_dtype', '#tp', '#pp', '#gpu_mem_util', '#enforce_eager',
    '#gpu_layers', '#images_per_request', '#gguf_file', '#attention', '#c_use', '#c_vram',
    '#c_bw', '#c_fp16', '#c_fp8', '#c_cc', '#c_nvlink', '#c_name'
  ].forEach(s => $(s).addEventListener('change', changed));
  $('#interconnect').addEventListener('change', e => e.target.dataset.touched = '1');
  ['#gpus_per_node', '#nodes'].forEach(s => $(s).addEventListener('input', gpuChanged));

  $('#rig-select').addEventListener('change', applyRig);
  $('#btn-save-rig').addEventListener('click', saveRig);
  $('#btn-del-rig').addEventListener('click', deleteRig);
  $('#btn-refresh').addEventListener('click', async () => {
    const n = (await api('/api/cache/clear', { method: 'POST' })).cleared;
    $('#btn-refresh').textContent = `Cleared ${n}`;
    setTimeout(() => $('#btn-refresh').textContent = 'Refresh data', 1600);
    if (MODEL) selectModel(MODEL.id, true);
    search();
  });

  // context input lives next to its slider
  ctx.classList.remove('hidden');
  ctx.style.cssText = 'width:110px;font-family:var(--mono);font-size:12px;padding:5px 8px;margin-top:6px';
  rng.parentElement.appendChild(ctx);

  gpuChanged(); syncCtx();
}

function changed() { renderSetupBar(); clearTimeout(evalTimer); evalTimer = setTimeout(runEval, 200); }
function syncCtx() { $('#ctx-label').textContent = (+$('#context_len').value || 0).toLocaleString(); }

let CONC = 1;                       // single source of truth for stream count
const CONC_PRESETS = [1, 8, 32, 128];

function buildConcurrency() {
  const inp = $('#concurrency');
  const preset = CONC_PRESETS.includes(CONC);
  const seg = $('#conc-seg');
  seg.innerHTML = CONC_PRESETS.map(n =>
    `<button data-n="${n}" class="${n === CONC && preset ? 'on' : ''}">${n}</button>`).join('');
  inp.classList.remove('hidden');
  inp.placeholder = 'custom';
  // Blank while a preset is active so it never reads as a fifth preset.
  if (document.activeElement !== inp) inp.value = preset ? '' : String(CONC);
  inp.style.cssText = 'width:84px;font-family:var(--mono);font-size:12.5px;padding:7px 8px;' +
    'text-align:center;border-style:dashed' +
    (preset ? '' : ';border-color:var(--acc-dim);background:var(--acc-bg);border-style:solid');
  seg.appendChild(inp);
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    CONC = +b.dataset.n; buildConcurrency(); changed();
  }));
  $('#conc-label').textContent = CONC.toLocaleString();
}
function setConcurrency(n) { CONC = Math.max(1, +n || 1); buildConcurrency(); }
function concurrencyValue() { return CONC; }

function syncEngine() {
  const eng = CAT.engines.find(e => e.key === $('#engine').value);
  const local = ['llamacpp', 'ollama'].includes(eng.key);
  const cur = $('#weight_format').value;
  const opts = CAT.weight_formats.filter(w => eng.weights.includes(w.key));
  fill('#weight_format', opts.map(w => [w.key, w.label]),
    opts.some(w => w.key === cur) ? cur : (local ? 'gguf_q4_k_m' : 'bf16'));
  $('#util-wrap').classList.toggle('hidden', local);
  $('#eager-wrap').classList.toggle('hidden', local);
  $('#layers-wrap').classList.toggle('hidden', !local);
  $('#tp').closest('.field').classList.toggle('hidden', local);
  $('#pp').closest('.field').classList.toggle('hidden', local);
  syncGguf();
}
function syncGguf() {
  const isG = $('#weight_format').value.startsWith('gguf');
  const files = (MODEL && MODEL.gguf_files) || [];
  $('#gguf-wrap').classList.toggle('hidden', !(isG && files.length));
  if (isG && files.length) {
    $('#gguf_file').innerHTML = '<option value="">— use the bits-per-weight estimate —</option>' +
      files.map(f => `<option value="${esc(f.path)}|${f.size}">${esc(f.path)} (${fmtBytes(f.size)})</option>`).join('');
  }
}
function gpuChanged() {
  const g = gpuOf($('#gpu').value);
  if (!g) return;
  $('#gpu-note').textContent = g.note || '';
  if (!$('#interconnect').dataset.touched) {
    $('#interconnect').value = g.vendor === 'apple' ? 'rdma' : (g.nvlink ? 'nvlink' : 'pcie');
  }
  const fam = (FAMILIES || []).find(f => f.key === familyKeyFor(g));
  const total = Math.max(1, +$('#gpus_per_node').value || 1) * Math.max(1, +$('#nodes').value || 1);
  const perUnit = (n, unit) => total > 1 ? `${n} × ${total} = ${(n * total).toLocaleString()}` : `${n} ${unit}`;
  const facts = [
    ['architecture', fam ? fam.name : g.arch],
    ['VRAM', g.vram_gb + ' GB'], ['bandwidth', g.bandwidth + ' GB/s'],
    ['compute cap', fam ? fam.sm : (g.cc > 0 ? 'sm_' + Math.round(g.cc * 10) : g.arch)],
    ['NVLink', g.nvlink ? g.nvlink + ' GB/s' : 'none'],
  ];
  if (g.cores) {
    facts.push(
      ['CPU cores', perUnit(g.cores.cpu, 'cores')],
      ['GPU cores', perUnit(g.cores.gpu, 'cores')],
      ['Neural Engine', perUnit(g.cores.npu, 'cores')],
    );
  }
  $('#gpu-facts').innerHTML = facts
    .map(([l, v]) => `<div><span>${esc(l)}</span><b${l === 'NVLink' && !g.nvlink ? ' style="color:var(--warn)"' : ''}>${esc(v)}</b></div>`).join('');
}

/* ------------------------------------------------------------------ views */
function switchView(v) {
  VIEW = v;
  setHash({ view: v === 'size' ? null : v });
  $$('.tab').forEach(t => t.classList.toggle('on', t.dataset.view === v));
  $$('.view').forEach(s => s.classList.add('hidden'));
  $('#view-' + v).classList.remove('hidden');
  $('#setupbar').classList.toggle('hidden', v === 'arch');
  if (v !== 'size') openSetup(false);
  if (v === 'recommend') renderRecommendView();
  if (v === 'arch') renderArchView();
}
function openSetup(on) {
  $('#setup-panel').classList.toggle('hidden', !on);
  $('#btn-setup-label').textContent = on ? 'Done' : 'Change setup';
  if (on && VIEW !== 'size') switchView('size');
}

/* ------------------------------------------------------------------ setup bar */
function renderSetupBar() {
  const g = $('#c_use').checked
    ? { name: $('#c_name').value || 'Custom GPU', vram_gb: +$('#c_vram').value || 0 }
    : gpuOf($('#gpu').value) || { name: '?', vram_gb: 0 };
  const n = +$('#gpus_per_node').value || 1, nodes = +$('#nodes').value || 1;
  const total = n * nodes;
  const eng = (CAT.engines.find(e => e.key === $('#engine').value) || {}).label || '';
  const fmt = ((CAT.weight_formats.find(w => w.key === $('#weight_format').value) || {}).label || '')
    .replace(/\s*\(.*\)$/, '');
  const ctx = +$('#context_len').value || 0, seqs = concurrencyValue();

  const model = MODEL
    ? `<div class="chip model" data-focus="q">${logo(MODEL.id, 'logo-xs')}
         <span class="mono">${esc(MODEL.id)}</span>
         <span class="sub">${fmtParams(MODEL.spec.params_total)}</span></div>`
    : `<div class="chip model" data-focus="q"><span class="dim">Choose a model</span></div>`;

  $('#setup-summary').innerHTML = `
    ${model}${ICON.sep}
    <div class="chip" data-focus="gpu">${ICON.gpu}<span>${total} × ${esc(g.name)}</span>${
      nodes > 1 ? `<span class="sub">${nodes} nodes</span>` : ''}</div>${ICON.sep}
    <div class="chip" data-focus="engine">${ICON.bolt}<span>${esc(eng)} · ${esc(fmt)}</span></div>
    <div class="chip" data-focus="context_len_range"><span class="dim">context</span>
      <span class="mono">${ctx.toLocaleString()}</span><span class="sub">× ${seqs}</span></div>`;

  $('#setup-summary').querySelectorAll('.chip').forEach(c => c.addEventListener('click', () => {
    openSetup(true);
    const el = $('#' + c.dataset.focus);
    if (el) setTimeout(() => { el.focus(); el.scrollIntoView({ block: 'center', behavior: 'smooth' }); }, 60);
  }));
}

/* ------------------------------------------------------------------ search */
async function search() {
  if (SRC === 'aa') return searchAA();
  const box = $('#results');
  box.innerHTML = '<div class="row"><span class="spin"></span>&nbsp;<span class="row-meta">searching…</span></div>';
  try {
    const { results } = await api('/api/models/search?' + new URLSearchParams({
      q: $('#q').value.trim(), pipeline: $('#pipeline').value, sort: $('#sort').value, limit: 40
    }));
    if (!results.length) { box.innerHTML = '<div class="row"><span class="row-meta">No models matched.</span></div>'; return; }
    box.innerHTML = results.map(m => `
      <div class="row${MODEL && MODEL.id === m.id ? ' on' : ''}" data-id="${esc(m.id)}">
        ${logo(m.id)}
        <div class="row-main">
          <div class="row-id">${esc(m.id)}</div>
          <div class="row-meta">
            ${m.params ? `<span class="tag blue">${fmtParams(m.params)}</span>` : ''}
            ${m.pipeline_tag ? `<span class="tag">${esc(m.pipeline_tag)}</span>` : ''}
            ${m.gguf ? '<span class="tag green">GGUF</span>' : ''}
            ${m.gated ? '<span class="tag red">gated</span>' : ''}
            <span>↓ ${fmtCount(m.downloads)}</span><span>${ago(m.lastModified)}</span>
          </div>
        </div>
      </div>`).join('');
    box.querySelectorAll('.row').forEach(r =>
      r.addEventListener('click', () => selectModel(r.dataset.id)));
    loadAvatars(results.map(m => m.id), () => box.querySelectorAll('.row').forEach(r => {
      const f = r.firstElementChild; if (f) f.outerHTML = logo(r.dataset.id);
    }));
  } catch (e) { box.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

async function searchAA() {
  const box = $('#results');
  box.innerHTML = '<div class="row"><span class="spin"></span>&nbsp;<span class="row-meta">ranking by intelligence…</span></div>';
  try {
    const { results } = await api('/api/aa/search?' + new URLSearchParams({
      q: $('#q').value.trim(), limit: 60 }));
    if (!results.length) { box.innerHTML = '<div class="row"><span class="row-meta">Nothing matched.</span></div>'; return; }
    // A picker should lead with what you can actually run. API-only models stay
    // visible as a count so the ranking is not silently edited.
    const open = results.filter(r => r.repo);
    const closed = results.filter(r => !r.repo);
    box.innerHTML = (open.length ? open.map(r => `
      <div class="aa-row pick${MODEL && MODEL.id === r.repo ? ' on' : ''}" data-id="${esc(r.repo)}">
        <div class="aa-score" style="color:var(--acc)">${r.score.toFixed(1)}</div>
        <div class="aa-main">
          <div class="aa-name">${esc(r.aa_name)}</div>
          <div class="aa-repo">${esc(r.repo)}</div>
        </div>
      </div>`).join('')
      : '<div class="row"><span class="row-meta">No open-weights models matched.</span></div>') +
      (closed.length ? `<div class="hint" style="padding:9px 10px;border-top:1px solid var(--line);margin-top:6px">
        ${closed.length} higher-ranked model${closed.length === 1 ? ' is' : 's are'} API-only —
        ${esc(closed.slice(0, 3).map(c => c.aa_name.split('(')[0].trim()).join(', '))}${
        closed.length > 3 ? '…' : ''} — with no weights to download.</div>` : '');
    box.querySelectorAll('.aa-row.pick').forEach(r =>
      r.addEventListener('click', () => selectModel(r.dataset.id)));
  } catch (e) { box.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

/* ------------------------------------------------------------------ model */
async function selectModel(id, refresh) {
  $$('#results .row').forEach(r => r.classList.toggle('on', r.dataset.id === id));
  setHash({ model: id });
  $('#model-col').innerHTML = '<div class="card"><span class="spin"></span> <span class="hint">reading config.json…</span></div>';
  try {
    MODEL = await api(`/api/models/detail?repo=${encodeURIComponent(id)}${refresh ? '&refresh=true' : ''}`);
  } catch (e) {
    MODEL = null;
    $('#model-col').innerHTML = `<div class="card"><div class="err">${esc(e.message)}</div></div>`;
    return;
  }
  autoFormat();
  syncGguf();
  const sp = MODEL.spec;
  if (sp.max_position_embeddings && !$('#context_len').dataset.touched) {
    const c = Math.min(sp.max_position_embeddings, 32768);
    $('#context_len').value = c;
    $('#context_len_range').value = Math.round(Math.log2(c));
    syncCtx();
  }
  renderSetupBar();
  loadAvatars([MODEL.id], () => { if (MODEL && MODEL.id === id) renderModelPanel(); });
  renderModelPanel();
  loadBench(MODEL.id, MODEL.base_model);
  openSetup(false);
  runEval();
}

function autoFormat() {
  if (!MODEL || $('#weight_format').dataset.touched) return;
  if (['llamacpp', 'ollama'].includes($('#engine').value)) return;
  const q = (MODEL.spec.quant_method || '').toLowerCase();
  const map = { fp8: 'fp8', fbgemm_fp8: 'fp8', mxfp4: 'mxfp4', nvfp4: 'nvfp4', awq: 'awq',
    gptq: 'gptq4', int4: 'gptq4', bitsandbytes: 'bnb4', bnb: 'bnb4' };
  const want = map[q] || ({ float16: 'fp16', bfloat16: 'bf16', float32: 'fp32' }[MODEL.spec.torch_dtype] || 'bf16');
  if ([...$('#weight_format').options].some(o => o.value === want)) $('#weight_format').value = want;
}

function renderModelPanel() {
  const m = MODEL; if (!m) return;
  const s = m.spec, a = m.arch_support;
  const specs = [
    ['parameters', fmtParams(s.params_total)],
    s.is_moe ? ['active per token', fmtParams(m.active_params)] : ['layers', fmtInt(s.num_layers)],
    s.is_moe ? ['layers', fmtInt(s.num_layers)] : ['hidden size', fmtInt(s.hidden_size)],
    ['attn / KV heads', `${s.num_attention_heads} / ${s.num_kv_heads}`],
    s.is_mla ? ['MLA latent rank', `${s.kv_lora_rank} + ${s.qk_rope_head_dim}`] : ['head dim', fmtInt(s.head_dim)],
    ['native context', fmtInt(s.max_position_embeddings)],
    s.sliding_window ? ['sliding window', `${fmtInt(s.sliding_window)}`] : null,
    ['repo size', fmtBytes(m.repo_bytes || m.weight_file_bytes)],
  ].filter(Boolean).slice(0, 8);
  if (specs.length % 2) specs.pop();

  const chip = (t, c) => `<span class="tag ${c || ''}">${esc(t)}</span>`;
  const st = k => ({ y: ['ok', 'supported'], p: ['warn', 'partial'], n: ['bad', 'no'] }[k] || ['faint', 'unknown']);

  $('#model-col').innerHTML = `
    <div class="card">
      <div class="mp-head">
        ${logo(m.id, 'logo-lg')}
        <div style="min-width:0">
          <a href="https://huggingface.co/${esc(m.id)}" target="_blank" rel="noopener">${esc(m.id.split('/').pop())}</a>
          <div class="mp-sub">${esc(m.id.split('/')[0])}${m.license ? ' · ' + esc(m.license) : ''} · ${ago(m.lastModified)}</div>
        </div>
      </div>
      <div class="row-meta" style="margin-top:12px">
        ${chip(a.family, 'blue')}
        ${s.is_moe ? chip('MoE', 'amber') : ''}${s.is_mla ? chip('MLA', 'amber') : ''}
        ${s.quant_method ? chip(s.quant_method.toUpperCase(), 'amber') : ''}
        ${s.has_vision ? chip('vision') : ''}${s.has_audio ? chip('audio') : ''}
        ${s.sliding_window ? chip('sliding window') : ''}
        ${m.gated ? chip('gated', 'red') : ''}
        ${m.has_gguf ? chip(m.gguf_files.length + ' GGUF', 'green') : ''}
      </div>
      <div class="specs">
        ${specs.map(([l, v]) => `<div class="spec"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join('')}
      </div>
      <div style="margin-top:14px;padding-top:14px;border-top:1px solid var(--line)">
        <div class="sect" style="margin-bottom:10px">How capable is it</div>
        <div id="bench-slot"><div class="iq-none"><span class="spin"></span> looking up…</div></div>
      </div>
    </div>

    <div class="card">
      <div class="sect" style="margin-bottom:12px">Runs on</div>
      ${[['vLLM', a.vllm], ['SGLang', a.sglang], ['llama.cpp', a.llamacpp]].map(([n, k]) => {
        const [c, t] = st(k);
        return `<div class="support-row"><span>${n}</span>
          <span class="st" style="color:var(--${c})"><span class="dot" style="background:var(--${c})"></span>${t}</span></div>`;
      }).join('')}
      ${a.note ? `<p class="gpu-note">${esc(a.note)}</p>` : ''}
    </div>`;
}

async function loadBench(id, base) {
  const slot = () => $('#bench-slot');
  if (!slot()) return;
  try {
    const r = await api('/api/bench?' + new URLSearchParams({ repo: id, base_model: base || '' }));
    if (!MODEL || MODEL.id !== id || !slot()) return;
    MODEL.bench = r.bench;
    slot().innerHTML = benchBlock(r.bench);
  } catch (e) { if (slot()) slot().innerHTML = `<div class="iq-none">Lookup failed: ${esc(e.message)}</div>`; }
}
function benchBlock(b) {
  if (!b || b.score == null) {
    return `<div class="iq-none">No published benchmark score. The Open LLM Leaderboard was archived
      in 2025${CAT.bench && CAT.bench.artificial_analysis.configured ? '' :
      ', and Artificial Analysis is not configured here'}.</div>`;
  }
  const pct = Math.max(2, Math.min(100, b.source === 'artificial-analysis' ? b.score : b.score * 2));
  return `<div style="display:flex;align-items:flex-end;gap:10px">
      <div class="iq-score">${b.score.toFixed(1)}</div>
      <div class="hint" style="padding-bottom:3px">/ 100</div></div>
    <div class="meter"><i style="width:${pct}%"></i></div>
    <div class="iq-src">${esc(b.source_label)}${b.name ? ` · matched <b>${esc(b.name)}</b>` : ''}${
      b.inherited_from ? ` · via <b>${esc(b.inherited_from)}</b>` : ''}</div>`;
}

/* ------------------------------------------------------------------ state */
function hardware() {
  const hw = {
    gpu_id: $('#gpu').value,
    gpus_per_node: +$('#gpus_per_node').value || 1,
    nodes: +$('#nodes').value || 1,
    host_ram_gb: +$('#host_ram_gb').value || 0,
    os: $('#os').value,
    interconnect: $('#interconnect').value,
    driver_version: $('#driver_version').value.trim(),
    vram_override_gb: +$('#vram_override_gb').value || null,
  };
  if ($('#c_use').checked) hw.custom_gpu = {
    name: $('#c_name').value || 'Custom GPU', vram_gb: +$('#c_vram').value || 0,
    bandwidth: +$('#c_bw').value || 0, fp16: +$('#c_fp16').value || 0,
    fp8: +$('#c_fp8').value || 0, cc: +$('#c_cc').value || 8.0,
    nvlink: +$('#c_nvlink').value || 0, vendor: 'nvidia', arch: 'custom',
  };
  return hw;
}
function concurrencyValue() {
  const raw = ($('#concurrency').value || '').trim();
  if (raw) return Math.max(1, +raw || 1);
  const on = $('#conc-seg') && $('#conc-seg').querySelector('button.on');
  return on ? +on.dataset.n : 1;
}

function runtime() {
  const g = ($('#gguf_file').value || '').split('|');
  const layers = $('#gpu_layers').value.trim();
  return {
    engine: $('#engine').value, weight_format: $('#weight_format').value,
    kv_dtype: $('#kv_dtype').value,
    context_len: Math.max(64, +$('#context_len').value || 8192),
    concurrency: concurrencyValue(),
    tp: +$('#tp').value || 0, pp: +$('#pp').value || 0,
    gpu_mem_util: Math.min(1, Math.max(0.1, +$('#gpu_mem_util').value || 0.9)),
    enforce_eager: $('#enforce_eager').checked, max_batched_tokens: 8192,
    gguf_size: g.length > 1 ? +g[1] : 0,
    gguf_path: g[0] ? g[0].split('/').pop() : '',
    gpu_layers: (layers === '' || layers === 'auto') ? null : (+layers || 0),
    images_per_request: +$('#images_per_request').value || 0,
    attention: $('#attention').value,
  };
}
function applyPatch(p) {
  if (p.hardware) for (const [k, v] of Object.entries(p.hardware)) {
    const el = $('#' + k); if (el) el.value = v;
  }
  if (p.runtime) for (const [k, v] of Object.entries(p.runtime)) {
    if (k === 'concurrency') continue;            // handled below via setConcurrency
    const el = $('#' + k); if (!el) continue;
    el.value = v;
    if (k === 'weight_format' || k === 'context_len') el.dataset.touched = '1';
  }
  if (p.runtime && p.runtime.context_len) {
    $('#context_len_range').value = Math.round(Math.log2(Math.max(p.runtime.context_len, 1024)));
    syncCtx();
  }
  if (p.runtime && p.runtime.concurrency) setConcurrency(p.runtime.concurrency);
  gpuChanged(); renderSetupBar(); runEval();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ------------------------------------------------------------------ evaluate */
async function runEval() {
  if (!MODEL) return;
  const seq = ++evalSeq;
  try {
    const res = await api('/api/evaluate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo: MODEL.id, model: MODEL, hardware: hardware(), runtime: runtime() }),
    });
    if (seq !== evalSeq) return;
    LAST = res; renderAnswer(res);
    loadArchSupport(seq);
  } catch (e) {
    if (seq !== evalSeq) return;
    $('#answer-col').innerHTML = `<div class="card"><div class="err">${esc(e.message)}</div></div>`;
  }
}

async function loadArchSupport(seq) {
  const slot = () => $('#b-fam');
  if (!slot() || !MODEL) return;
  try {
    const r = await api('/api/arch-support', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo: MODEL.id, model: MODEL, runtime: runtime() }),
    });
    if (seq !== evalSeq || !slot()) return;
    ARCH_SUPPORT = r.families;
    slot().innerHTML = familyMatrix(r.families);
    const mine = r.families.find(f => f.key === (LAST.gpu.family_key));
    const v = $('#fam-row-value');
    if (v && mine) v.innerHTML = `<span class="dot" style="background:var(--${
      { yes: 'ok', degraded: 'warn', no: 'bad' }[mine.status] || 'faint'})"></span>${esc(mine.name)} — ${
      { yes: 'supported', degraded: 'runs with limits', no: 'not supported' }[mine.status] || 'unknown'}`;
  } catch (e) {
    if (slot()) slot().innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

function familyMatrix(fams) {
  const mineKey = LAST && LAST.gpu ? LAST.gpu.family_key : null;
  const label = { yes: 'supported', degraded: 'limits', no: 'no', unknown: '?' };
  return `<p class="hint" style="margin:0 0 12px">Whether the architecture itself can run this
      model, with memory taken out of the question — a family marked supported can still be
      too small for it.</p>
    ${fams.map(f => `
      <div class="fam ${f.status}${f.key === mineKey ? ' mine' : ''}">
        <div class="fam-body">
          <div class="fam-name">${esc(f.name)}
            <span class="hint mono">${esc(f.sm)}</span>
            ${f.key === mineKey ? '<span class="you">yours</span>' : ''}</div>
          <div class="fam-cards">${esc(f.cards)}</div>
          ${f.reasons.length ? `<div class="fam-why">${esc(f.reasons.join(' · '))}</div>` : ''}
          ${f.caveats && f.caveats.length ? `<div class="fam-why">${esc(f.caveats.join(' · '))}</div>` : ''}
          ${f.key === mineKey ? `<div class="fam-caps">
            ${f.yes.map(x => `<span class="cap y">${esc(x)}</span>`).join('')}
            ${f.no.map(x => `<span class="cap n">no ${esc(x)}</span>`).join('')}</div>
            <div class="fam-why">${esc(f.note)}</div>` : ''}
        </div>
        <span class="fam-st ${f.status}">${label[f.status] || f.status}</span>
      </div>`).join('')}`;
}

function renderAnswer(r) {
  const m = r.memory, p = r.perf, local = m.mode === 'local';
  const want = runtime();
  const mark = { yes: ICON.tick, tight: ICON.warn, no: ICON.cross }[r.verdict];

  const segs = [
    ['w', m.weights_gib, 'Weights'],
    ['ov', m.overhead_gib, local ? 'Reserve' : 'Framework'],
    ['act', m.activations_gib, local ? 'Compute buffer' : 'Activations'],
    ['cg', m.cudagraph_gib, 'CUDA graphs'],
    ['kv', m.kv_needed_gib, 'KV cache'],
  ].filter(s => s[1] > 0.005);
  const required = segs.reduce((a, s) => a + s[1], 0);
  const over = required > m.budget_gib + 0.01;

  let bar;
  if (over) {
    const budgetPct = 100 * m.budget_gib / required;
    bar = `<div class="overbar">
      <div class="zone" style="left:${budgetPct}%"></div>
      <div class="mark" style="left:${budgetPct}%"></div>
      <div class="lbl-l" style="left:${budgetPct}%">${fmtGiB(m.budget_gib)} budget</div>
      <div class="lbl-r" style="left:${budgetPct}%">${fmtGiB(required - m.budget_gib)} over</div>
      <div class="bar">${segs.map(([c, v]) => `<div class="seg ${c}" style="width:${100 * v / required}%"></div>`).join('')}</div>
    </div>`;
  } else {
    const total = m.vram_gib;
    const free = Math.max(m.budget_gib - required, 0);
    const reserve = Math.max(total - m.budget_gib, 0);
    bar = `<div class="bar">
      ${segs.map(([c, v]) => `<div class="seg ${c}" style="width:${100 * v / total}%"></div>`).join('')}
      <div class="seg free" style="width:${100 * free / total}%"></div>
      ${reserve > 0.01 ? `<div class="seg res" style="width:${100 * reserve / total}%"></div>` : ''}
    </div>`;
    segs.push(['free', free, 'Free']);
  }

  const hl = [];
  hl.push(['max context', fmtBig(m.max_context_single), 'one stream',
    m.max_context_single >= want.context_len ? '' : 'bad']);
  hl.push(['streams', fmtInt(m.max_concurrent_at_ctx), `at ${fmtBig(want.context_len)} context`,
    m.max_concurrent_at_ctx >= want.concurrency ? '' : 'bad']);
  if (local) hl.push(['GPU layers', `${m.gpu_layers}/${m.total_layers}`,
    m.gpu_layers >= m.total_layers ? 'fully offloaded' : 'partial offload',
    m.gpu_layers >= m.total_layers ? '' : 'warn']);
  if (p.available) {
    hl.push(['decode', fmtTok(p.decode_tps_single) + '<small> tok/s</small>', 'single stream', '']);
    hl.push(['first token', (p.ttft_s < 1 ? Math.round(p.ttft_s * 1000) + '<small> ms</small>'
      : p.ttft_s.toFixed(1) + '<small> s</small>'), 'at full context', '']);
  }

  const by = { blocker: [], warn: [], info: [] };
  r.findings.forEach(f => by[f.level].push(f));
  const shown = [...by.blocker, ...by.warn];

  const sug = r.suggestions || [];
  const par = local ? `${r.parallel.total_gpus} × ${esc(r.gpu.name)}`
    : `TP${r.parallel.tp}${r.parallel.pp > 1 ? ` × PP${r.parallel.pp}` : ''} on ${r.parallel.gpus_used} × ${esc(r.gpu.name)}`;

  $('#answer-col').innerHTML = `
    <div class="verdict v-${r.verdict}">
      <div class="v-head">
        <div class="v-mark">${mark}</div>
        <div style="min-width:0;flex:1">
          <h2>${esc(r.headline)}</h2>
          <div class="v-sub">${par}${r.gpu.family ? ` (${esc(r.gpu.family)})` : ''} · ${fmtGiB(m.weights_gib)} of weights per card${
            over ? '' : ` · ${fmtGiB(Math.max(m.budget_gib - required, 0))} spare`}</div>
        </div>
        ${over
          ? `<div class="v-right"><div class="v-over">${fmtGiB(required - m.budget_gib)}</div>
             <div class="v-over-lbl">over budget, per GPU</div></div>`
          : `<div class="v-par"><b>${r.weights.bpw.toFixed(2)} bits/weight</b>${
              r.parallel.auto ? '<span>auto-sized</span>' : ''}</div>`}
      </div>

      <div class="headline">
        ${hl.map(([l, v, s, c]) => `<div class="hl ${c || ''}"><b>${v}</b><span>${esc(l)} — ${esc(s)}</span></div>`).join('')}
      </div>

      <div class="section">
        <div class="sec-head">
          <h3>${over ? 'What it needs vs. what this GPU has' : (local ? 'Memory across ' + r.parallel.total_gpus + ' GPUs' : 'Memory per GPU')}</h3>
          <span class="meta">${over ? fmtGiB(required) + ' required' :
            `${fmtGiB(m.vram_gib)} card · ${fmtGiB(m.budget_gib)} budget at ${Math.round(m.util * 100)}%`}</span>
        </div>
        ${bar}
        <div class="legend">
          ${segs.map(([c, v, l]) => `<div><i class="seg ${c}"></i><span>${esc(l)}</span><b>${fmtGiB(v)}</b></div>`).join('')}
        </div>
        ${local && m.weights_cpu_gib > 0.01 ? `<p class="gpu-note" style="color:var(--warn)">Plus
          ${fmtGiB(m.weights_cpu_gib)} of weights and ${fmtGiB(m.kv_cpu_gib)} of KV in system RAM.</p>` : ''}
      </div>
    </div>

    ${sug.length ? `<div class="card">
      <div class="sec-head"><h3>What would make it fit</h3>
        <span class="meta">each option re-sized against your hardware</span></div>
      ${sug.map((s, i) => `
        <div class="fix ${i === 0 ? 'best' : ''}">
          ${ICON.check}
          <div class="fix-body">
            <div class="fix-title">${esc(s.title)}</div>
            <div class="fix-detail">TP${s.tp}${s.pp > 1 ? ` × PP${s.pp}` : ''} on ${s.gpus} GPU${s.gpus > 1 ? 's' : ''} ·
              <span class="mono">${fmtGiB(s.weights_gib)}</span> of weights per card ·
              <span class="mono">${fmtGiB(s.kv_available_gib)}</span> for KV · up to
              <span class="mono">${fmtInt(s.max_context)}</span> tokens</div>
            ${s.note ? `<div class="fix-note">${esc(s.note)}</div>` : ''}
          </div>
          <button class="${i === 0 ? 'primary' : 'ghost'}" data-fix="${i}">Apply</button>
        </div>`).join('')}
    </div>` : ''}

    <div class="card">
      <div class="sec-head"><h3>Worth knowing</h3>
        <span class="meta">${by.blocker.length} blocker${by.blocker.length === 1 ? '' : 's'} ·
          ${by.warn.length} warning${by.warn.length === 1 ? '' : 's'} · ${by.info.length} note${by.info.length === 1 ? '' : 's'}</span></div>
      ${shown.length ? shown.map(f => `
        <div class="finding f-${f.level}">
          <div><div class="f-title">${esc(f.title)}</div><div class="f-detail">${esc(f.detail)}</div></div>
        </div>`).join('') : '<p class="hint">Nothing blocking or surprising here.</p>'}
      ${by.info.length ? `<div class="fold" id="fold-info">
        <span class="lbl"><span class="dot info"></span>${by.info.length} note${by.info.length === 1 ? '' : 's'} on
          ${esc(tagWords(by.info.map(f => f.tag)))}</span>${ICON.down}</div>
        <div class="hidden" id="info-body" style="margin-top:8px">
          ${by.info.map(f => `<div class="finding"><div><div class="f-title">${esc(f.title)}</div>
            <div class="f-detail">${esc(f.detail)}</div></div></div>`).join('')}
        </div>` : ''}
    </div>

    <div class="rows" id="drows">
      <div class="drow" data-body="b-attn">
        ${ICON.bolt}<div class="t">Attention backend</div>
        <div class="v">${r.attention && r.attention.chosen
          ? `<span class="dot" style="background:var(--${attnTone(r)})"></span>${esc(r.attention.chosen.label)}`
          : '<span class="dot" style="background:var(--bad)"></span>none available'}</div>${ICON.chev}
      </div>
      <div class="dbody pad hidden" id="b-attn">${attentionBlock(r.attention)}</div>

      <div class="drow" data-body="b-fam">
        ${ICON.gpu}<div class="t">GPU architecture</div>
        <div class="v" id="fam-row-value"><span class="spin"></span></div>${ICON.chev}
      </div>
      <div class="dbody pad hidden" id="b-fam"><span class="spin"></span> <span class="hint">checking architectures…</span></div>

      <div class="drow" data-body="b-cmd">
        ${ICON.term}<div class="t">Launch command</div>
        <div class="v mono" style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.command.split('\n')[0])}</div>
        <button class="ghost sm" id="btn-copy">Copy</button>${ICON.chev}
      </div>
      <div class="dbody pad hidden" id="b-cmd"><pre class="cmd">${esc(r.command)}</pre></div>

      <div class="drow" data-body="b-cmp">
        ${ICON.swap}<div class="t">Compare against other GPUs</div>
        <div class="v">9 cards</div>${ICON.chev}
      </div>
      <div class="dbody pad hidden" id="b-cmp"><div id="compare-out"><button class="ghost" id="btn-compare">Run comparison</button></div></div>
    </div>`;

  $$('#answer-col [data-fix]').forEach(b =>
    b.addEventListener('click', () => applyPatch(sug[+b.dataset.fix].patch)));
  const fold = $('#fold-info');
  if (fold) fold.addEventListener('click', () => {
    const body = $('#info-body');
    body.classList.toggle('hidden');
    fold.style.borderStyle = body.classList.contains('hidden') ? 'dashed' : 'solid';
  });
  $$('#drows .drow').forEach(row => row.addEventListener('click', e => {
    if (e.target.closest('button')) return;
    const body = $('#' + row.dataset.body);
    body.classList.toggle('hidden');
    row.classList.toggle('open', !body.classList.contains('hidden'));
  }));
  // #open=fam|cmd|cmp deep-links straight to one of the disclosure rows
  const openRow = new URLSearchParams(location.hash.slice(1)).get('open');
  if (openRow) {
    const row = $(`#drows .drow[data-body="b-${openRow}"]`);
    if (row) { $('#b-' + openRow).classList.remove('hidden'); row.classList.add('open'); }
  }
  $('#btn-copy').addEventListener('click', e => {
    e.stopPropagation();
    navigator.clipboard.writeText(r.command);
    e.target.textContent = 'Copied'; setTimeout(() => e.target.textContent = 'Copy', 1400);
  });
  $('#btn-compare').addEventListener('click', runCompare);
}

// Green only when the chosen backend is the good path and nothing warns about it.
function attnTone(r) {
  const c = r.attention && r.attention.chosen;
  if (!c) return 'bad';
  const warned = r.findings.some(f => f.tag === 'attention' && f.level !== 'info');
  return (c.status === 'ok' && !warned) ? 'ok' : 'warn';
}

function attentionBlock(a) {
  if (!a) return '<p class="hint">not evaluated</p>';
  const n = a.needs;
  const needs = [n.mla ? 'multi-head latent attention' : null, n.swa ? 'sliding window' : null,
    n.fp8_kv ? 'FP8/INT8 KV cache' : null, n.head_dim ? `head_dim ${n.head_dim}` : null].filter(Boolean);
  const mk = { ok: '●', warn: '◐', no: '○' };
  return `<div class="needs"><span class="hint">This model needs:</span>
      ${needs.map(x => `<span class="tag">${esc(x)}</span>`).join('') || '<span class="tag">standard attention</span>'}</div>
    ${a.candidates.map(c => `
      <div class="attn ${c.status}${a.chosen && a.chosen.key === c.key ? ' chosen' : ''}">
        <span style="color:var(--${c.status === 'ok' ? 'ok' : c.status === 'warn' ? 'warn' : 'faint'})">${mk[c.status]}</span>
        <div style="min-width:0">
          <div class="attn-name">${esc(c.label)}${a.chosen && a.chosen.key === c.key ? ' <span class="tag blue">selected</span>' : ''}</div>
          <div class="attn-why">${esc(c.reason || c.note)}</div>
        </div>
      </div>`).join('')}`;
}

async function runCompare() {
  const out = $('#compare-out');
  out.innerHTML = '<span class="spin"></span> <span class="hint">sizing across GPUs…</span>';
  const ids = ['l20', 'l40s', 'h20', 'h100-sxm', 'h200-sxm', 'a100-80', 'b200', 'rtx-4090', 'mi300x'];
  const cur = $('#gpu').value;
  if (!ids.includes(cur)) ids.unshift(cur);
  try {
    const { results } = await api('/api/compare?gpu_ids=' + ids.join(','), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo: MODEL.id, model: MODEL, hardware: hardware(), runtime: runtime() }),
    });
    out.innerHTML = `<table><thead><tr><th>GPU</th><th>Fits</th><th class="mono">TP×PP</th>
        <th class="mono">Weights</th><th class="mono">KV free</th><th class="mono">Max ctx</th><th class="mono">Decode</th></tr></thead><tbody>
      ${results.map(x => `<tr${x.gpu_id === cur ? ' style="background:var(--acc-bg)"' : ''}>
        <td>${esc(x.name)}</td>
        <td><span class="v-chip ${x.verdict}">${x.verdict === 'yes' ? 'fits' : x.verdict === 'tight' ? 'tight' : 'no'}</span></td>
        <td class="mono">${x.parallel.tp}×${x.parallel.pp}</td>
        <td class="mono">${fmtGiB(x.memory.weights_gib)}</td>
        <td class="mono">${fmtGiB(Math.max(x.memory.kv_available_gib, 0))}</td>
        <td class="mono">${fmtInt(x.memory.max_context_single)}</td>
        <td class="mono">${x.perf.available ? fmtTok(x.perf.decode_tps_single) : '—'}</td></tr>`).join('')}</tbody></table>`;
  } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

/* ------------------------------------------------------------------ rigs */
async function loadRigs() {
  RIGS = (await api('/api/rigs')).rigs;
  const sel = $('#rig-select'), keep = sel.value;
  sel.innerHTML = '<option value="">— saved rigs —</option>' +
    RIGS.map(r => `<option value="${esc(r.id)}">${esc(r.name)}${r.builtin ? ' (detected)' : ''}</option>`).join('');
  if (RIGS.some(r => r.id === keep)) sel.value = keep;
  else if (RIGS.length && !keep) { sel.value = RIGS[0].id; applyRig(); }
}
function applyRig() {
  const r = RIGS.find(x => x.id === $('#rig-select').value);
  if (!r) return;
  if (r.gpu_id) $('#gpu').value = r.gpu_id;
  $('#gpus_per_node').value = r.gpus_per_node || 1;
  $('#nodes').value = r.nodes || 1;
  $('#host_ram_gb').value = r.host_ram_gb || 0;
  $('#os').value = r.os || 'linux';
  $('#interconnect').value = r.interconnect || 'pcie';
  $('#driver_version').value = r.driver_version || '';
  $('#vram_override_gb').value = r.vram_override_gb || '';
  if (r.custom_gpu) {
    $('#c_use').checked = true; $('#c_name').value = r.custom_gpu.name || '';
    $('#c_vram').value = r.custom_gpu.vram_gb || ''; $('#c_bw').value = r.custom_gpu.bandwidth || '';
    $('#c_fp16').value = r.custom_gpu.fp16 || ''; $('#c_fp8').value = r.custom_gpu.fp8 || '';
    $('#c_cc').value = r.custom_gpu.cc || ''; $('#c_nvlink').value = r.custom_gpu.nvlink || 0;
  } else $('#c_use').checked = false;
  gpuChanged(); changed();
}
async function saveRig() {
  const existing = RIGS.find(r => r.id === $('#rig-select').value);
  const g = gpuOf($('#gpu').value) || { name: 'GPU' };
  const name = prompt('Name this rig', existing && !existing.builtin ? existing.name
    : `${$('#gpus_per_node').value}× ${g.name}`);
  if (!name) return;
  const reuse = existing && !existing.builtin && existing.name === name;
  await api(reuse ? `/api/rigs/${existing.id}` : '/api/rigs', {
    method: reuse ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, ...hardware() }),
  });
  await loadRigs();
}
async function deleteRig() {
  const r = RIGS.find(x => x.id === $('#rig-select').value);
  if (!r || !confirm(`Delete rig "${r.name}"?`)) return;
  await api(`/api/rigs/${r.id}`, { method: 'DELETE' });
  $('#rig-select').value = '';
  await loadRigs();
}

/* ------------------------------------------------------------------ recommend view */
function renderRecommendView() {
  const g = gpuOf($('#gpu').value) || { name: 'GPU' };
  const total = (+$('#gpus_per_node').value || 1) * (+$('#nodes').value || 1);
  $('#view-recommend').innerHTML = `
    <div class="view-head">
      <div>
        <h2>Best models for ${total} × ${esc(g.name)}</h2>
        <p>Sized against your hardware at ${fmtBig(+$('#context_len').value || 8192)} context,
          strongest first.</p>
      </div>
      <div class="view-controls">
        <select id="rec-pipe">${CAT.pipelines.slice(0, 6).map(p =>
          `<option value="${esc(p.key)}">${esc(p.label)}</option>`).join('')}</select>
        <select id="rec-sort">
          <option value="aa">Artificial Analysis — by intelligence</option>
          <option value="trendingScore">Trending on the Hub</option>
          <option value="downloads">Most downloaded</option>
          <option value="likes">Most liked</option></select>
        <select id="rec-pool">
          <option value="20">20 models</option><option value="30" selected>30 models</option>
          <option value="50">50 models</option></select>
        <button class="primary" id="rec-go">Scan</button>
      </div>
    </div>
    <div class="banner" id="rec-banner"></div>
    <div id="rec-out"><button class="primary" id="rec-start">Scan the Hub</button></div>`;
  const syncRecSrc = () => {
    const aa = $('#rec-sort').value === 'aa';
    $('#rec-pipe').classList.toggle('hidden', aa);
    $('#rec-pool').title = aa ? 'How many open-weights models to take from the AA ranking' : '';
    $('#rec-banner').innerHTML = ICON.info + '<span>' + (aa
      ? 'Ranked by Artificial Analysis\u2019 intelligence index, strongest first, then filtered to models that actually publish weights. API-only models are excluded — there is nothing to download.'
      : 'Ranked from a HuggingFace popularity list, which is a proxy for attention rather than capability. Scores come from Artificial Analysis and the archived HF leaderboard; models with no published score sit below the ranked ones rather than being guessed at.')
      + '</span>';
  };
  $('#rec-sort').addEventListener('change', syncRecSrc);
  syncRecSrc();
  $('#rec-go').addEventListener('click', runRecommend);
  $('#rec-start').addEventListener('click', runRecommend);
  if (new URLSearchParams(location.hash.slice(1)).get('scan')) runRecommend();
}

async function runRecommend() {
  const out = $('#rec-out');
  const n = +$('#rec-pool').value || 30;
  const aaSrc = $('#rec-sort').value === 'aa';
  out.innerHTML = `<div class="banner"><span class="spin"></span><span>${aaSrc
    ? `Walking the Artificial Analysis ranking for the ${n} strongest models with open weights, then sizing each one`
    : `Reading ${n} model configs from the Hub and looking up scores`}.
    The first run takes a while; results are cached.</span></div>`;
  try {
    const r = await api('/api/recommend', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hardware: hardware(), runtime: runtime(),
        pipeline: $('#rec-pipe').value,
        sort: aaSrc ? 'downloads' : $('#rec-sort').value,
        source: aaSrc ? 'aa' : 'hf',
        pool: n, min_params: aaSrc ? 0 : 1 }),
    });
    const rows = r.results;
    const scored = rows.filter(x => x.bench && x.bench.score != null);
    const best = Math.max(...scored.map(x => x.bench.score), 1);
    const fits = rows.filter(x => x.verdict !== 'no');
    const nope = rows.filter(x => x.verdict === 'no');

    const row = x => {
      const b = x.bench, sc = b && b.score != null ? b.score : null;
      return `<tr class="rowlink" data-id="${esc(x.id)}"${x.verdict === 'no' ? ' style="opacity:.62"' : ''}>
        <td style="width:150px">${sc == null ? '<span class="hint">no data</span>'
          : `<b class="mono" style="font-size:15px">${sc.toFixed(1)}</b>
             <div class="meter" style="margin-top:5px"><i style="width:${Math.max(3, 100 * sc / best)}%"></i></div>`}</td>
        <td><div class="mono" style="display:flex;align-items:center;gap:9px">${logo(x.id)}<span
            style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(x.id)}</span></div>
          <div class="row-meta" style="margin-left:31px">${esc(x.family || '')}
            ${b && b.inherited_from ? `<span class="tag">via ${esc(b.inherited_from.split('/').pop())}</span>` : ''}
            ${x.gated ? '<span class="tag red">gated</span>' : ''}
            ${x.verdict === 'no' && x.blockers.length ? `<span style="color:var(--bad)">${esc(x.blockers[0])}</span>` : ''}</div></td>
        <td class="mono">${fmtParams(x.params)}${x.active_params && x.active_params !== x.params
          ? `<div class="hint">${fmtParams(x.active_params)} active</div>` : ''}</td>
        <td><span class="v-chip ${x.verdict}">${x.verdict === 'yes' ? 'fits' : x.verdict === 'tight' ? 'tight' : 'no'}</span></td>
        <td class="mono">${x.parallel.tp}×${x.parallel.pp}</td>
        <td class="mono">${x.verdict === 'no' ? '—' : fmtInt(x.max_context)}</td>
        <td class="mono">${x.verdict === 'no' ? '—' : fmtTok(x.decode_tps)}</td></tr>`;
    };

    out.innerHTML = `
      ${r.source === 'aa' ? `<p class="hint" style="margin:0 0 12px">The
        ${r.scanned} strongest <b>open-weights</b> models on Artificial Analysis, sized against your
        hardware. Reaching them meant walking ${r.aa.considered} of ${r.aa.total_ranked} ranked
        entries — ${r.aa.no_open_weights} of those are API-only models with no weights to download.</p>`
      : `<p class="hint" style="margin:0 0 12px">Sized ${r.scanned} of ${r.shortlisted} shortlisted ·
        ${scored.length} have a published score${r.skipped.gated.length
          ? ` · ${r.skipped.gated.length} gated skipped` : ''}${
        (r.skipped.too_large || []).length ? ` · <span title="${esc(r.skipped.too_large.join(', '))}">${
          r.skipped.too_large.length} too large at ${esc(r.format_label)}</span>` : ''}${
        (r.skipped.gguf_only || []).length ? ` · <span title="${esc(r.skipped.gguf_only.join(', '))}">${
          r.skipped.gguf_only.length} GGUF-only — switch the engine to llama.cpp to include them</span>` : ''}${
        (r.skipped.fetch_failed || []).length ? ` · <span title="${esc(r.skipped.fetch_failed.join(', '))}">${
          r.skipped.fetch_failed.length} could not be read from the Hub</span>` : ''}.
        <br>Ranked from the <b>${esc(($('#rec-sort').selectedOptions[0] || {}).text || '')}</b> list —
        a model outside it will not appear here even if it fits your hardware.</p>`}
      ${!r.bench_available && !scored.length ? `<div class="err" style="margin-bottom:12px">The benchmark
        source did not answer for this scan (it throttles bursts). Sizing is still valid — scan again in a moment.</div>` : ''}
      ${nope.length >= fits.length && nope.length ? `<div class="banner">${ICON.info}<span>Only
        ${fits.length} of these fit at <b>${esc(r.format_label)}</b>. Quantizing — FP8 or AWQ in
        the setup panel — usually brings several more within reach.</span></div>` : ''}
      <div class="rows">
        <table><thead><tr><th>Score</th><th>Model</th><th class="mono">Size</th><th>Fits</th>
          <th class="mono">TP×PP</th><th class="mono">Max context</th><th class="mono">Decode</th></tr></thead>
          <tbody>${fits.map(row).join('')}</tbody></table>
        ${nope.length ? `<div class="subhead">Stronger, but needs more hardware</div>
          <table><tbody>${nope.map(row).join('')}</tbody></table>` : ''}
      </div>`;

    out.querySelectorAll('.rowlink').forEach(tr => tr.addEventListener('click', () => {
      switchView('size'); selectModel(tr.dataset.id);
    }));
    loadAvatars(rows.map(x => x.id), () => out.querySelectorAll('.rowlink').forEach(tr => {
      const c = tr.querySelector('td:nth-child(2) .mono');
      if (c) c.innerHTML = logo(tr.dataset.id) +
        `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(tr.dataset.id)}</span>`;
    }));
  } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

/* ------------------------------------------------------------------ arch view */
async function renderArchView() {
  const v = $('#view-arch');
  v.innerHTML = '<p class="hint"><span class="spin"></span> loading…</p>';
  const [{ rows }, fam] = await Promise.all([api('/api/arch'), api('/api/gpu-families')]);
  const cell = k => `<span class="status-${k === '?' ? 'p' : k}">${
    { y: '● yes', p: '◐ partial', n: '○ no', '?': '· unknown' }[k]}</span>`;
  v.innerHTML = `
    <div class="view-head">
      <div><h2>GPU architectures</h2>
        <p>What each family can do. Compute capability is what kernels gate on;
          the model table below says which families each architecture can serve.</p></div>
    </div>
    <div class="rows" style="margin-bottom:28px"><table><thead><tr>
      <th>Family</th><th class="mono">Target</th><th>Cards</th><th>Supports</th><th>Lacks</th>
      </tr></thead><tbody>
      ${fam.families.map(f => `<tr>
        <td><b>${esc(f.name)}</b></td>
        <td class="mono">${esc(f.sm)}</td>
        <td class="hint">${esc(f.cards)}</td>
        <td>${f.yes.map(x => `<span class="cap y">${esc(x)}</span>`).join(' ')}</td>
        <td>${f.no.length ? f.no.map(x => `<span class="cap n">${esc(x)}</span>`).join(' ')
          : '<span class="hint">—</span>'}</td></tr>`).join('')}
      </tbody></table></div>

    <div class="view-head">
      <div><h2>Model families</h2>
        <p>${rows.length} model families. <b>min sm</b> is the lowest CUDA compute capability whose
          kernels cover the family; traits drive the memory model.</p></div>
      <div class="view-controls"><input type="search" id="arch-q" placeholder="Filter…" style="width:240px"></div>
    </div>
    <div class="rows"><table id="arch-table"><thead><tr>
      <th>Family</th><th class="mono">model_type</th><th>vLLM</th><th>SGLang</th><th>llama.cpp</th>
      <th class="mono">min sm</th><th>Traits</th></tr></thead><tbody>
      ${rows.map(r => `<tr data-k="${esc((r.family + ' ' + r.model_type + ' ' + r.traits.join(' ')).toLowerCase())}">
        <td>${esc(r.family)}${r.note ? ` <span class="tag" title="${esc(r.note)}">i</span>` : ''}</td>
        <td class="mono">${esc(r.model_type)}</td>
        <td>${cell(r.vllm)}</td><td>${cell(r.sglang)}</td><td>${cell(r.llamacpp)}</td>
        <td class="mono">${r.min_cc.toFixed(1)}</td>
        <td>${r.traits.map(t => `<span class="tag">${esc(t)}</span>`).join(' ')}</td></tr>`).join('')}
    </tbody></table></div>`;
  $('#arch-q').addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    $$('#arch-table tbody tr').forEach(tr => tr.style.display = tr.dataset.k.includes(q) ? '' : 'none');
  });
}
