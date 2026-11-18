'use strict';
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));

let CAT = null, MODEL = null, LAST = null, RIGS = [], AVATARS = new Map(), ARCH_SUPPORT = null;
let searchTimer, evalTimer, evalSeq = 0, VIEW = 'size', SRC = 'hf';

/* ------------------------------------------------------------------ language */
// Arabic is a first-class UI mode. Technical identifiers (model IDs, engine
// names, CLI flags and CUDA/Metal terms) intentionally remain untouched.
const LANG_KEY = 'podfit_language';
let LANG = (() => { try { return localStorage.getItem(LANG_KEY) || 'en'; } catch (e) { return 'en'; } })();
const AR = {
  'Size a model': 'حجم نموذج', 'Best for my hardware': 'الأفضل لجهازي',
  'Compatibility': 'التوافق', 'Docs': 'الوثائق', 'API keys': 'مفاتيح API',
  'Refresh data': 'تحديث البيانات', 'API Keys': 'مفاتيح API', 'Done': 'تم',
  'Personal access token': 'رمز وصول شخصي', 'API key': 'مفتاح API',
  'Save keys': 'حفظ المفاتيح', 'Clear keys': 'مسح المفاتيح', 'Setup': 'الإعداد',
  'Change setup': 'تغيير الإعداد', 'Model': 'النموذج', 'Hardware': 'العتاد',
  'Runtime': 'بيئة التشغيل', 'HuggingFace': 'HuggingFace',
  'Artificial Analysis': 'Artificial Analysis', 'All task types': 'كل أنواع المهام',
  'Trending': 'الأكثر رواجاً', 'Most downloaded': 'الأكثر تحميلاً',
  'Most liked': 'الأكثر إعجاباً', 'Recently updated': 'حُدِّث مؤخراً', 'Newest': 'الأحدث',
  '— saved rigs —': '— الأجهزة المحفوظة —', 'Save': 'حفظ', 'Delete': 'حذف',
  'GPU': 'وحدة GPU', 'How many': 'العدد', 'Nodes': 'العُقد',
  'Interconnect, host RAM, OS, driver, custom GPU': 'الربط، ذاكرة المضيف، النظام، التعريف، GPU مخصص',
  'Interconnect': 'الربط البيني', 'PCIe only': 'PCIe فقط', 'Ethernet': 'إيثرنت',
  'Host RAM (GB)': 'ذاكرة RAM للمضيف (GB)', 'Operating system': 'نظام التشغيل',
  'NVIDIA driver': 'تعريف NVIDIA', 'VRAM override (GB)': 'تجاوز VRAM (GB)',
  'usable, if different from spec': 'المتاح، إن اختلف عن المواصفة', 'Name': 'الاسم',
  'VRAM (GB)': 'VRAM (GB)', 'Bandwidth (GB/s)': 'عرض النطاق (GB/s)',
  'FP16 dense TFLOPS': 'TFLOPS كثيفة FP16', 'FP8 dense TFLOPS': 'TFLOPS كثيفة FP8',
  'Compute capability': 'قدرة الحوسبة', 'NVLink (GB/s, 0 = none)': 'NVLink (GB/s، 0 = لا يوجد)',
  'Use this custom GPU': 'استخدم وحدة GPU المخصصة هذه', 'Engine': 'المحرك',
  'Weights': 'الأوزان', 'GGUF file': 'ملف GGUF',
  'Advanced — parallelism, KV dtype, attention, utilization': 'متقدم — التوازي، نوع KV، الانتباه، الاستفادة',
  'Tensor parallel': 'التوازي الموترّي', 'Pipeline parallel': 'التوازي الخطي',
  'KV cache dtype': 'نوع بيانات ذاكرة KV', 'Attention backend': 'خلفية الانتباه',
  'gpu_memory_utilization': 'استخدام ذاكرة GPU', 'GPU layers (-ngl)': 'طبقات GPU (-ngl)',
  'Images per request': 'الصور لكل طلب', 'Use CUDA graphs / do not enforce eager': 'استخدم CUDA graphs / لا تفرض eager',
  'Pick a model to size it': 'اختر نموذجاً لحسابه', 'Choose a model': 'اختر نموذجاً',
  '> awaiting input_': '> بانتظار الإدخال_', 'Search HuggingFace — qwen3, deepseek, gemma…': 'ابحث في HuggingFace — qwen3، deepseek، gemma…',
  'Filter Artificial Analysis models — qwen, glm, deepseek…': 'رشّح نماذج Artificial Analysis — qwen، glm، deepseek…',
  'optional': 'اختياري', 'auto': 'تلقائي', 'custom': 'مخصص', 'Filter…': 'تصفية…'
};
const i18n = {
  searching: 'جارٍ البحث…', noModels: 'لا توجد نماذج مطابقة.', ranking: 'جارٍ الترتيب حسب الذكاء…',
  nothing: 'لا توجد نتائج مطابقة.', reading: 'جارٍ قراءة config.json…', failed: 'فشل البدء: ',
  refresh: 'تحديث البيانات', cleared: n => `تم مسح ${n}`, checking: 'جارٍ التحقق…',
  saveKeys: 'حفظ المفاتيح', copied: 'تم النسخ', copy: 'نسخ', apply: 'تطبيق',
  scan: 'فحص', scanHub: 'فحص Hub', loading: 'جارٍ التحميل…', compare: 'تشغيل المقارنة',
  allTaskTypes: 'كل أنواع المهام', useEstimate: '— استخدم تقدير البتات لكل وزن —',
  enginePicks: 'تلقائي (يختار المحرك)', savedRigs: '— الأجهزة المحفوظة —',
};
const isAr = () => LANG === 'ar';
const tx = (english, arabic) => isAr() ? (arabic || AR[english] || english) : english;
function applyLanguage() {
  const root = document.documentElement;
  root.lang = isAr() ? 'ar' : 'en';
  root.dir = isAr() ? 'rtl' : 'ltr';
  document.body.classList.toggle('lang-ar', isAr());
  document.title = isAr() ? 'بودفِت — هل يعمل هذا النموذج على عتادي؟' : 'podfit — will this model run on my GPUs?';
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach(node => {
    const p = node.parentElement;
    if (!p || ['SCRIPT', 'STYLE', 'CODE', 'PRE'].includes(p.tagName)) return;
    const raw = node.nodeValue;
    const key = raw.trim();
    if (!key || !AR[key]) return;
    if (!node.__podfitEn) node.__podfitEn = raw;
    node.nodeValue = raw.replace(key, isAr() ? AR[key] : node.__podfitEn.trim());
  });
  document.querySelectorAll('[placeholder],[title]').forEach(el => {
    ['placeholder', 'title'].forEach(attr => {
      const value = el.getAttribute(attr);
      if (!value) return;
      if (!el.dataset['en' + attr]) el.dataset['en' + attr] = value;
      const original = el.dataset['en' + attr];
      el.setAttribute(attr, isAr() ? (AR[original] || original) : original);
    });
  });
  const btn = $('#btn-language');
  if (btn) { btn.textContent = isAr() ? 'English' : 'العربية'; btn.title = isAr() ? 'Switch to English' : 'التبديل إلى العربية'; }
  const staticCopy = {
    '#answer-placeholder .ph-sub': [
      'podfit reads the real config.json and safetensors parameter counts from HuggingFace, then computes weights, KV cache, activations and framework overhead against the hardware you describe.',
      'يقرأ podfit ملف config.json الحقيقي وعدد معاملات safetensors من HuggingFace، ثم يحسب الأوزان وذاكرة KV والأنشطة وزيادة الإطار مقابل العتاد الذي تصفه.'
    ],
    '#answer-placeholder .prompt-line': ['> awaiting input_', '> بانتظار الإدخال_'],
    '#keys-panel .setup-grid > .card:nth-child(1) .gpu-note': [
      `Used from your browser only, instead of this server's key. Stored in this browser's local storage on this device — never saved server-side. <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Get a token ↗</a>`,
      'يُستخدم من متصفحك فقط بدلاً من مفتاح هذا الخادم. يُحفظ في التخزين المحلي لهذا المتصفح ولا يُحفظ في الخادم. <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">احصل على رمز ↗</a>'
    ],
    '#keys-panel .setup-grid > .card:nth-child(2) .gpu-note': [
      `Used from your browser only, instead of this server's key. Stored in this browser's local storage on this device — never saved server-side. <a href="https://artificialanalysis.ai/" target="_blank" rel="noopener">Get a key ↗</a>`,
      'يُستخدم من متصفحك فقط بدلاً من مفتاح هذا الخادم. يُحفظ في التخزين المحلي لهذا المتصفح ولا يُحفظ في الخادم. <a href="https://artificialanalysis.ai/" target="_blank" rel="noopener">احصل على مفتاح ↗</a>'
    ]
  };
  Object.entries(staticCopy).forEach(([sel, copies]) => {
    const el = $(sel); if (el) el.innerHTML = isAr() ? copies[1] : copies[0];
  });
}

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
  $('#hf-status').textContent = 'HF ' + (cat.hf_token ? tx('· token set', '· تم ضبط الرمز') : tx('· anonymous', '· مجهول'));
  $('#hf-status').title = cat.hf_token ? tx('Gated repos resolve.', 'يمكن الوصول إلى المستودعات المقيدة.')
    : tx('No HF token — gated repos will not resolve.', 'لا يوجد رمز HF — لن تُحل المستودعات المقيدة.');
  const aa = cat.bench && cat.bench.artificial_analysis.configured;
  $('#aa-status').textContent = 'AA ' + (aa ? tx('· on', '· نشط') : tx('· off', '· متوقف'));
  $('#aa-status').title = aa ? tx('Artificial Analysis scores are available.', 'درجات Artificial Analysis متاحة.')
    : tx('No AA key — only the archived HF leaderboard supplies scores.', 'لا يوجد مفتاح AA — يوفّر جدول HF المؤرشف الدرجات فقط.');
  $('#key_hf_status').textContent = hasHf
    ? (cat.hf_token ? tx('Your personal token is active.', 'رمزك الشخصي نشط.') : tx('Saved, but not recognized — double-check it.', 'حُفظ، لكنه غير معروف — راجعه.'))
    : (cat.hf_token ? tx('No personal token saved — using the server’s token.', 'لا يوجد رمز شخصي محفوظ — يُستخدم رمز الخادم.')
                    : tx('No token — gated repos will not resolve.', 'لا يوجد رمز — لن تُحل المستودعات المقيدة.'));
  $('#key_aa_status').textContent = hasAa
    ? (aa ? tx('Your personal key is active.', 'مفتاحك الشخصي نشط.') : tx('Saved, but not recognized — double-check it.', 'حُفظ، لكنه غير معروف — راجعه.'))
    : (aa ? tx('No personal key saved — using the server’s key.', 'لا يوجد مفتاح شخصي محفوظ — يُستخدم مفتاح الخادم.')
          : tx('No key — only the archived HF leaderboard supplies scores.', 'لا يوجد مفتاح — يوفّر جدول HF المؤرشف الدرجات فقط.'));
}

(async function boot() {
  applyLanguage();
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
  $('#attention').innerHTML = `<option value="auto">${tx('auto (engine picks)', i18n.enginePicks)}</option>` +
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
    $('#q').placeholder = tx('Filter Artificial Analysis models — qwen, glm, deepseek…');
  }
  search();

  if (p.get('model')) selectModel(p.get('model'));
  else openSetup(true);
  if (p.get('view')) switchView(p.get('view'));
})().catch(e => document.body.insertAdjacentHTML('afterbegin',
  `<div class="err" style="margin:16px">Failed to start: ${esc(e.message)}</div>`));

function fill(sel, pairs, val) {
  const el = $(sel);
  const keep = sel === '#pipeline' ? `<option value="">${tx('All task types', i18n.allTaskTypes)}</option>` : '';
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
  $('#btn-language').addEventListener('click', () => {
    LANG = isAr() ? 'en' : 'ar';
    try { localStorage.setItem(LANG_KEY, LANG); } catch (e) { /* storage unavailable */ }
    location.reload();
  });
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
    $('#keys-save').textContent = tx('Checking…', i18n.checking);
    try {
      CAT = await api('/api/catalog');
      applyKeyStatus(CAT);
    } finally {
      $('#keys-save').textContent = tx('Save keys', i18n.saveKeys);
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
      ? tx('Filter Artificial Analysis models — qwen, glm, deepseek…')
      : tx('Search HuggingFace — qwen3, deepseek, gemma…');
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
    setTimeout(() => $('#btn-refresh').textContent = tx('Refresh data'), 1600);
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
    $('#gguf_file').innerHTML = `<option value="">${tx('— use the bits-per-weight estimate —', i18n.useEstimate)}</option>` +
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
  $('#setupbar').classList.toggle('hidden', v === 'arch' || v === 'docs');
  if (v !== 'size') openSetup(false);
  if (v === 'recommend') renderRecommendView();
  if (v === 'arch') renderArchView();
  if (v === 'docs' && !$('#view-docs').dataset.built) renderDocsView();
}
function openSetup(on) {
  $('#setup-panel').classList.toggle('hidden', !on);
  $('#btn-setup-label').textContent = on ? tx('Done') : tx('Change setup');
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
  box.innerHTML = `<div class="row"><span class="spin"></span>&nbsp;<span class="row-meta">${tx('searching…', i18n.searching)}</span></div>`;
  try {
    const { results } = await api('/api/models/search?' + new URLSearchParams({
      q: $('#q').value.trim(), pipeline: $('#pipeline').value, sort: $('#sort').value, limit: 40
    }));
    if (!results.length) { box.innerHTML = `<div class="row"><span class="row-meta">${tx('No models matched.', i18n.noModels)}</span></div>`; return; }
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
  box.innerHTML = `<div class="row"><span class="spin"></span>&nbsp;<span class="row-meta">${tx('ranking by intelligence…', i18n.ranking)}</span></div>`;
  try {
    const { results } = await api('/api/aa/search?' + new URLSearchParams({
      q: $('#q').value.trim(), limit: 60 }));
    if (!results.length) { box.innerHTML = `<div class="row"><span class="row-meta">${tx('Nothing matched.', i18n.nothing)}</span></div>`; return; }
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
  $('#model-col').innerHTML = `<div class="card"><span class="spin"></span> <span class="hint">${tx('reading config.json…', i18n.reading)}</span></div>`;
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
    ['w', m.weights_gib, tx('Weights', 'الأوزان')],
    ['ov', m.overhead_gib, local ? tx('Reserve', 'المساحة المحجوزة') : tx('Framework', 'الإطار')],
    ['act', m.activations_gib, local ? tx('Compute buffer', 'ذاكرة الحوسبة') : tx('Activations', 'الأنشطة')],
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
      <div class="lbl-l" style="left:${budgetPct}%">${fmtGiB(m.budget_gib)} ${tx('budget', 'الميزانية')}</div>
      <div class="lbl-r" style="left:${budgetPct}%">${fmtGiB(required - m.budget_gib)} ${tx('over', 'زيادة')}</div>
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
    segs.push(['free', free, tx('Free', 'متاح')]);
  }

  const hl = [];
  hl.push([tx('max context', 'أقصى سياق'), fmtBig(m.max_context_single), tx('one stream', 'تدفق واحد'),
    m.max_context_single >= want.context_len ? '' : 'bad']);
  hl.push([tx('streams', 'التدفقات'), fmtInt(m.max_concurrent_at_ctx), `${tx('at', 'عند')} ${fmtBig(want.context_len)} ${tx('context', 'سياق')}`,
    m.max_concurrent_at_ctx >= want.concurrency ? '' : 'bad']);
  if (local) hl.push([tx('GPU layers', 'طبقات GPU'), `${m.gpu_layers}/${m.total_layers}`,
    m.gpu_layers >= m.total_layers ? tx('fully offloaded', 'نقل كامل') : tx('partial offload', 'نقل جزئي'),
    m.gpu_layers >= m.total_layers ? '' : 'warn']);
  if (p.available) {
    hl.push([tx('decode', 'التوليد'), fmtTok(p.decode_tps_single) + '<small> tok/s</small>', tx('single stream', 'تدفق واحد'), '']);
    hl.push([tx('first token', 'أول رمز'), (p.ttft_s < 1 ? Math.round(p.ttft_s * 1000) + '<small> ms</small>'
      : p.ttft_s.toFixed(1) + '<small> s</small>'), tx('at full context', 'عند السياق الكامل'), '']);
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
             <div class="v-over-lbl">${tx('over budget, per GPU', 'فوق الميزانية، لكل GPU')}</div></div>`
          : `<div class="v-par"><b>${r.weights.bpw.toFixed(2)} bits/weight</b>${
              r.parallel.auto ? `<span>${tx('auto-sized', 'بحجم تلقائي')}</span>` : ''}</div>`}
      </div>

      <div class="headline">
        ${hl.map(([l, v, s, c]) => `<div class="hl ${c || ''}"><b>${v}</b><span>${esc(l)} — ${esc(s)}</span></div>`).join('')}
      </div>

      <div class="section">
        <div class="sec-head">
          <h3>${over ? tx('What it needs vs. what this GPU has', 'ما يحتاجه مقابل ما لدى GPU') : (local ? `${tx('Memory across', 'الذاكرة عبر')} ${r.parallel.total_gpus} GPU` : tx('Memory per GPU', 'الذاكرة لكل GPU'))}</h3>
          <span class="meta">${over ? fmtGiB(required) + ' ' + tx('required', 'مطلوب') :
            `${fmtGiB(m.vram_gib)} ${tx('card', 'بطاقة')} · ${fmtGiB(m.budget_gib)} ${tx('budget', 'ميزانية')} ${tx('at', 'عند')} ${Math.round(m.util * 100)}%`}</span>
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
      <div class="sec-head"><h3>${tx('What would make it fit', 'ما الذي يجعله مناسباً')}</h3>
        <span class="meta">${tx('each option re-sized against your hardware', 'أُعيد حساب كل خيار مقابل عتادك')}</span></div>
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
          <button class="${i === 0 ? 'primary' : 'ghost'}" data-fix="${i}">${tx('Apply', i18n.apply)}</button>
        </div>`).join('')}
    </div>` : ''}

    <div class="card">
      <div class="sec-head"><h3>${tx('Worth knowing', 'معلومات مهمة')}</h3>
        <span class="meta">${by.blocker.length} ${tx('blocker', 'مانع')} ·
          ${by.warn.length} ${tx('warning', 'تحذير')} · ${by.info.length} ${tx('note', 'ملاحظة')}</span></div>
      ${shown.length ? shown.map(f => `
        <div class="finding f-${f.level}">
          <div><div class="f-title">${esc(f.title)}</div><div class="f-detail">${esc(f.detail)}</div></div>
        </div>`).join('') : `<p class="hint">${tx('Nothing blocking or surprising here.', 'لا توجد موانع أو مفاجآت هنا.')}</p>`}
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
        ${ICON.bolt}<div class="t">${tx('Attention backend', 'خلفية الانتباه')}</div>
        <div class="v">${r.attention && r.attention.chosen
          ? `<span class="dot" style="background:var(--${attnTone(r)})"></span>${esc(r.attention.chosen.label)}`
          : `<span class="dot" style="background:var(--bad)"></span>${tx('none available', 'لا يتوفر شيء')}`}</div>${ICON.chev}
      </div>
      <div class="dbody pad hidden" id="b-attn">${attentionBlock(r.attention)}</div>

      <div class="drow" data-body="b-fam">
        ${ICON.gpu}<div class="t">${tx('GPU architecture', 'معمارية GPU')}</div>
        <div class="v" id="fam-row-value"><span class="spin"></span></div>${ICON.chev}
      </div>
      <div class="dbody pad hidden" id="b-fam"><span class="spin"></span> <span class="hint">${tx('checking architectures…', 'جارٍ فحص المعماريات…')}</span></div>

      <div class="drow" data-body="b-cmd">
        ${ICON.term}<div class="t">${tx('Launch command', 'أمر التشغيل')}</div>
        <div class="v mono" style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.command.split('\n')[0])}</div>
        <button class="ghost sm" id="btn-copy">${tx('Copy', i18n.copy)}</button>${ICON.chev}
      </div>
      <div class="dbody pad hidden" id="b-cmd"><pre class="cmd">${esc(r.command)}</pre></div>

      <div class="drow" data-body="b-cmp">
        ${ICON.swap}<div class="t">${tx('Compare against other GPUs', 'قارن مع وحدات GPU أخرى')}</div>
        <div class="v">9 ${tx('cards', 'بطاقات')}</div>${ICON.chev}
      </div>
      <div class="dbody pad hidden" id="b-cmp"><div id="compare-out"><button class="ghost" id="btn-compare">${tx('Run comparison', i18n.compare)}</button></div></div>
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
    e.target.textContent = tx('Copied', i18n.copied); setTimeout(() => e.target.textContent = tx('Copy', i18n.copy), 1400);
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
  if (!a) return `<p class="hint">${tx('not evaluated', 'لم يُقيَّم')}</p>`;
  const n = a.needs;
  const needs = [n.mla ? 'multi-head latent attention' : null, n.swa ? 'sliding window' : null,
    n.fp8_kv ? 'FP8/INT8 KV cache' : null, n.head_dim ? `head_dim ${n.head_dim}` : null].filter(Boolean);
  const mk = { ok: '●', warn: '◐', no: '○' };
  return `<div class="needs"><span class="hint">${tx('This model needs:', 'يحتاج هذا النموذج إلى:')}</span>
      ${needs.map(x => `<span class="tag">${esc(x)}</span>`).join('') || `<span class="tag">${tx('standard attention', 'انتباه قياسي')}</span>`}</div>
    ${a.candidates.map(c => `
      <div class="attn ${c.status}${a.chosen && a.chosen.key === c.key ? ' chosen' : ''}">
        <span style="color:var(--${c.status === 'ok' ? 'ok' : c.status === 'warn' ? 'warn' : 'faint'})">${mk[c.status]}</span>
        <div style="min-width:0">
          <div class="attn-name">${esc(c.label)}${a.chosen && a.chosen.key === c.key ? ` <span class="tag blue">${tx('selected', 'محدد')}</span>` : ''}</div>
          <div class="attn-why">${esc(c.reason || c.note)}</div>
        </div>
      </div>`).join('')}`;
}

async function runCompare() {
  const out = $('#compare-out');
  out.innerHTML = `<span class="spin"></span> <span class="hint">${tx('sizing across GPUs…', 'جارٍ الحساب عبر وحدات GPU…')}</span>`;
  const ids = ['l20', 'l40s', 'h20', 'h100-sxm', 'h200-sxm', 'a100-80', 'b200', 'rtx-4090', 'mi300x'];
  const cur = $('#gpu').value;
  if (!ids.includes(cur)) ids.unshift(cur);
  try {
    const { results } = await api('/api/compare?gpu_ids=' + ids.join(','), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo: MODEL.id, model: MODEL, hardware: hardware(), runtime: runtime() }),
    });
    out.innerHTML = `<table><thead><tr><th>GPU</th><th>${tx('Fits', 'الملاءمة')}</th><th class="mono">TP×PP</th>
        <th class="mono">${tx('Weights', 'الأوزان')}</th><th class="mono">KV ${tx('free', 'متاح')}</th><th class="mono">${tx('Max ctx', 'أقصى سياق')}</th><th class="mono">${tx('Decode', 'التوليد')}</th></tr></thead><tbody>
      ${results.map(x => `<tr${x.gpu_id === cur ? ' style="background:var(--acc-bg)"' : ''}>
        <td>${esc(x.name)}</td>
        <td><span class="v-chip ${x.verdict}">${x.verdict === 'yes' ? tx('fits', 'مناسب') : x.verdict === 'tight' ? tx('tight', 'محدود') : tx('no', 'لا')}</span></td>
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
  sel.innerHTML = `<option value="">${tx('— saved rigs —', i18n.savedRigs)}</option>` +
    RIGS.map(r => `<option value="${esc(r.id)}">${esc(r.name)}${r.builtin ? ` (${tx('detected', 'مكتشف')})` : ''}</option>`).join('');
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
  const name = prompt(tx('Name this rig', 'سمِّ هذا الجهاز'), existing && !existing.builtin ? existing.name
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
  if (!r || !confirm(isAr() ? `حذف الجهاز "${r.name}"؟` : `Delete rig "${r.name}"?`)) return;
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
        <h2>${tx('Best models for', 'أفضل النماذج لـ')} ${total} × ${esc(g.name)}</h2>
        <p>${tx('Sized against your hardware at', 'حُسبت مقابل عتادك عند')} ${fmtBig(+$('#context_len').value || 8192)} ${tx('context, strongest first.', 'سياق، الأقوى أولاً.')}</p>
      </div>
      <div class="view-controls">
        <select id="rec-pipe">${CAT.pipelines.slice(0, 6).map(p =>
          `<option value="${esc(p.key)}">${esc(p.label)}</option>`).join('')}</select>
        <select id="rec-sort">
          <option value="aa">Artificial Analysis — ${tx('by intelligence', 'حسب الذكاء')}</option>
          <option value="trendingScore">${tx('Trending on the Hub', 'الأكثر رواجاً على Hub')}</option>
          <option value="downloads">${tx('Most downloaded', 'الأكثر تحميلاً')}</option>
          <option value="likes">${tx('Most liked', 'الأكثر إعجاباً')}</option></select>
        <select id="rec-pool">
          <option value="20">20 ${tx('models', 'نموذجاً')}</option><option value="30" selected>30 ${tx('models', 'نموذجاً')}</option>
          <option value="50">50 ${tx('models', 'نموذجاً')}</option></select>
        <button class="primary" id="rec-go">${tx('Scan', i18n.scan)}</button>
      </div>
    </div>
    <div class="banner" id="rec-banner"></div>
    <div id="rec-out"><button class="primary" id="rec-start">${tx('Scan the Hub', i18n.scanHub)}</button></div>`;
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
  v.innerHTML = `<p class="hint"><span class="spin"></span> ${tx('loading…', i18n.loading)}</p>`;
  const [{ rows }, fam] = await Promise.all([api('/api/arch'), api('/api/gpu-families')]);
  const cell = k => `<span class="status-${k === '?' ? 'p' : k}">${
    { y: '● yes', p: '◐ partial', n: '○ no', '?': '· unknown' }[k]}</span>`;
  v.innerHTML = `
    <div class="view-head">
      <div><h2>${tx('GPU architectures', 'معماريات GPU')}</h2>
        <p>${tx('What each family can do. Compute capability is what kernels gate on; the model table below says which families each architecture can serve.', 'ما يمكن لكل عائلة فعله. قدرة الحوسبة هي ما تعتمد عليه kernels؛ ويبين الجدول أدناه العائلات التي يمكن لكل معمارية خدمتها.')}</p></div>
    </div>
    <div class="rows" style="margin-bottom:28px"><table><thead><tr>
      <th>${tx('Family', 'العائلة')}</th><th class="mono">${tx('Target', 'الهدف')}</th><th>${tx('Cards', 'البطاقات')}</th><th>${tx('Supports', 'يدعم')}</th><th>${tx('Lacks', 'لا يدعم')}</th>
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
      <div><h2>${tx('Model families', 'عائلات النماذج')}</h2>
        <p>${rows.length} ${tx('model families.', 'عائلة نموذج.')} <b>min sm</b> ${tx('is the lowest CUDA compute capability whose kernels cover the family; traits drive the memory model.', 'هي أقل قدرة CUDA تغطي kernels الخاصة بالعائلة؛ والخصائص تقود نموذج الذاكرة.')}</p></div>
      <div class="view-controls"><input type="search" id="arch-q" placeholder="${tx('Filter…')}" style="width:240px"></div>
    </div>
    <div class="rows"><table id="arch-table"><thead><tr>
      <th>${tx('Family', 'العائلة')}</th><th class="mono">model_type</th><th>vLLM</th><th>SGLang</th><th>llama.cpp</th>
      <th class="mono">min sm</th><th>${tx('Traits', 'الخصائص')}</th></tr></thead><tbody>
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

/* ------------------------------------------------------------------ docs */
function renderDocsArabicView() {
  $('#view-docs').dataset.built = '1';
  $('#view-docs').innerHTML = `
    <div class="view-head"><div><h2>الوثائق</h2>
      <p>تعريف بالمشروع، مرجع العتاد، مزودو السحابة، المحركات، وطريقة المساهمة.</p></div></div>
    <div class="docs-layout">
      <nav class="docs-nav">
        <a href="#docs-about">عن المشروع</a>
        <a href="#docs-hw">مرجع العتاد</a>
        <a href="#docs-interconnects" class="sub">الربط البيني</a>
        <a href="#docs-apple" class="sub">Apple Silicon</a>
        <a href="#docs-nvidia" class="sub">NVIDIA و L20</a>
        <a href="#docs-providers">مزودو السحابة</a>
        <a href="#docs-aws" class="sub">AWS</a>
        <a href="#docs-azure" class="sub">Azure</a>
        <a href="#docs-gcp" class="sub">Google Cloud</a>
        <a href="#docs-alibaba" class="sub">Alibaba Cloud</a>
        <a href="#docs-engines">محركات LLM</a>
        <a href="#docs-vllm" class="sub">vLLM</a>
        <a href="#docs-sglang" class="sub">SGLang</a>
        <a href="#docs-llamacpp" class="sub">llama.cpp</a>
        <a href="#docs-ollama" class="sub">Ollama</a>
        <a href="#docs-contributing">المساهمة</a>
      </nav>
      <article class="docs-content">
        <section id="docs-about">
          <h3>عن المشروع</h3>
          <p>يجيب <b>podfit</b> عن سؤال واحد: <b>هل يعمل هذا النموذج على عتادي، وبأي أداء؟</b>
          اختر أي نموذج من HuggingFace، صف وحدات GPU لديك، اختر المحرك وكمّ الأوزان، ثم يحسب podfit
          الأوزان وذاكرة KV والأنشطة وCUDA graphs والزيادة الخاصة بالإطار، ويعرض الملاءمة والسياق والتوازي وسرعة تقريبية وأمر التشغيل.</p>
          <p>بناه عبدالرحمن الميمان، مهندس ذكاء اصطناعي وسحابة.</p>
          <div class="docs-links">
            <a class="ghost sm" href="https://www.abdulrahmanalmyman.dev/" target="_blank" rel="noopener">الموقع ↗</a>
            <a class="ghost sm" href="https://www.linkedin.com/in/abdulrahmanalmyman/" target="_blank" rel="noopener">LinkedIn ↗</a>
            <a class="ghost sm" href="https://x.com/PYTHON01100100" target="_blank" rel="noopener">X ↗</a>
            <a class="ghost sm" href="https://github.com/PYTHON01100100" target="_blank" rel="noopener">GitHub ↗</a>
            <a class="ghost sm" href="https://github.com/PYTHON01100100/PodFit" target="_blank" rel="noopener">مستودع المشروع ↗</a>
            <a class="primary sm" href="https://buymeacoffee.com/AbdulrahmanAlmyman" target="_blank" rel="noopener">ادعم المشروع ↗</a>
          </div>
        </section>

        <section id="docs-hw"><h3>مرجع العتاد</h3>
          <p>تجمع الآلة الحاسبة سعة الذاكرة وعرض النطاق وقدرة الحوسبة ومتطلبات المحرك والربط البيني. الذاكرة تجيب عن «هل يناسب؟»، أما عرض النطاق فيقيد سرعة التوليد، وقدرة الحوسبة تقيد kernels المتاحة.</p>
        </section>
        <section id="docs-interconnects"><h4>NVLink و RDMA و PCIe و Ethernet</h4>
          <p><b>NVLink / NVSwitch</b> هو أسرع ربط خاص من NVIDIA للوحدات ضمن الخادم، وهو الخيار المثالي للتوازي الموترّي TP. يصل H100/H200 SXM إلى 900 GB/s، وتصل Blackwell SXM إلى 1.8 TB/s.</p>
          <p><b>RDMA</b> عبر InfiniBand أو RoCE مع GPUDirect RDMA هو المسار المعتاد بين الخوادم. يظل أبطأ بكثير من NVLink، لذلك يُفضّل TP داخل الخادم وPP بين الخوادم. في مجموعة Mac Studio يعرض podfit خيار <b>RDMA (Thunderbolt 5)</b>؛ تحتاج إلى llama.cpp RPC أو MLX distributed، ولا تتحد الذاكرة تلقائياً.</p>
          <p><b>PCIe</b> مناسب لعدد محدود من البطاقات، بينما Ethernet العادي يصلح أكثر لـ Pipeline Parallel ولا يصلح عملياً لـ TP عبر الخوادم.</p>
        </section>
        <section id="docs-apple"><h4>Apple Silicon والذاكرة الموحّدة</h4>
          <p>تستخدم Apple Silicon ذاكرة موحدة تشاركها CPU وGPU وNeural Engine. يحتسب podfit نحو 75% من الذاكرة المعلنة كمساحة متاحة بأمان للنموذج كي يترك حيزاً للنظام والتطبيقات.</p>
          <p>تعمل أجهزة Max وUltra مع <b>llama.cpp (Metal)</b> أو <b>MLX</b>، لا مع vLLM أو SGLang. يضيف Ultra عرض نطاق وسعة أكبر؛ وUltraFusion لا يجعل عدة أجهزة Mac Studio جهازاً واحداً—التوزيع يحتاج runtime وشبكة فعلية.</p>
        </section>
        <section id="docs-nvidia"><h4>NVIDIA و L20</h4>
          <p>رقم compute capability هو بوابة kernels الحقيقية. يحتاج FlashAttention-2 إلى sm80، ويحتاج FP8 KV إلى sm89، وتحتاج MLA إلى Hopper أو Blackwell لتحقيق أفضل مسار.</p>
          <p><b>L20</b> بطاقة Ada بسعة 48GB وعرض نطاق 864 GB/s، صممت للاستدلال والامتثال الإقليمي. تصلح للنماذج المكمّمة وخدمات الدفعات المتوسطة، لكن ليس لديها NVLink؛ لذا يقل تحجيم TP فوق PCIe. لا تخلطها مع H20: H20 Hopper مختلفة تماماً.</p>
        </section>

        <section id="docs-providers"><h3>مزودو السحابة</h3><p>تختلف المنطقة والحصة والأسعار من وقت لآخر. استخدم أسماء عائلات الآلات أدناه كنقطة بداية ثم تحقق من المنطقة المختارة وقيود السعة والتسعير قبل التشغيل.</p></section>
        <section id="docs-aws"><h4>AWS</h4>
          <p><b>P5</b> يقدم H100 SXM (عادة 8 وحدات لكل خادم). <b>P5e</b> و<b>P5en</b> يقدمان H200، بينما <b>P6-B200</b> هو جيل Blackwell. AWS خيار قوي عندما تحتاج شبكة managed وشبكات EFA ومجموعات تدريب كبيرة؛ لكن الحصة والسعة الإقليمية هما القيد المعتاد.</p>
        </section>
        <section id="docs-azure"><h4>Microsoft Azure</h4>
          <p>تغطي <b>ND H100 v5</b> و<b>ND H200 v5</b> Hopper. ويظهر Blackwell في عائلة <b>ND GB200 v6</b>. يناسب Azure البيئات المرتبطة بـ Entra ID وAzure ML وPrivate Link، وغالباً تحتاج طلب حصة GPU قبل إنشاء العقدة.</p>
        </section>
        <section id="docs-gcp"><h4>Google Cloud</h4>
          <p><b>A3 Mega</b> (<code>a3-megagpu-8g</code>) يوفر H100، و<b>A3 Ultra</b> (<code>a3-ultragpu-8g</code>) يوفر H200، و<b>A4</b> يقدم B200. استخدم reservation أو capacity block عندما تحتاج بدءاً مؤكداً لمجموعة كبيرة.</p>
        </section>
        <section id="docs-alibaba"><h4>Alibaba Cloud</h4>
          <p>العائلات الأكثر صلة: <b>GN7i</b> لـ A10، و<b>GN8is</b>/<b>GN8v</b> لـ L20 مع خيارات 1 و2 و4 و8 GPU، وخط <b>gx8sf / L20X</b> الذي يركز على الذاكرة والشبكات السريعة. تختلف الإتاحة حسب المنطقة وقد يتطلب بعض الأنواع التواصل مع المبيعات. تحقق من صفحة ECS الخاصة بمنطقتك قبل اعتماد الاسم أو السعة.</p>
        </section>

        <section id="docs-engines"><h3>محركات LLM</h3><p>يعرض podfit فقط المسارات التي يمكن للمحرك والعتاد تشغيلها فعلاً. اختر المحرك أولاً لأن نوع الأوزان ونظام التشغيل والـ kernels تختلف باختلافه.</p></section>
        <section id="docs-vllm"><h4>vLLM</h4><p>Linux فقط. مناسب لخدمات CUDA/ROCm عالية الإنتاجية ويدعم fp16 وbf16 وfp32 وfp8 وint8 وAWQ وGPTQ وbitsandbytes وMXFP4 وNVFP4. يستخدم paged attention والتوازي الموترّي والخطي للخدمات الكبيرة.</p></section>
        <section id="docs-sglang"><h4>SGLang</h4><p>Linux فقط. محرك خدمة سريع يدعم fp16 وbf16 وfp8 وint8 وAWQ وGPTQ وMXFP4 وNVFP4. خيار جيد عند الحاجة إلى runtime برمجي وتحسينات توجيه/ذاكرة موجهة للاستدلال.</p></section>
        <section id="docs-llamacpp"><h4>llama.cpp</h4><p>يعمل على Linux وWindows وmacOS. يدعم GGUF وfp16، وهو المسار الأساسي للأجهزة المحلية وApple Silicon عبر Metal. يمكنه توزيع الطبقات عبر أجهزة متعددة أو عبر RPC، لكن السرعة تعتمد على الربط الفعلي بين الأجهزة.</p></section>
        <section id="docs-ollama"><h4>Ollama</h4><p>يعمل على Linux وWindows وmacOS، ويستخدم GGUF فقط. مناسب لتجربة النماذج محلياً بواجهة بسيطة، بينما llama.cpp يمنحك تحكماً أدق في ملفات GGUF وطبقات GPU وأوامر التشغيل.</p></section>

        <section id="docs-contributing"><h3>المساهمة</h3>
          <p>المشروع مفتوح المصدر تحت GPL-3.0. يمكنك اقتراح GPU أو مزود سحابي أو دعم عائلة نموذج أو إصلاح حسابات الذاكرة. افتح issue يحتوي على روابط لمواصفات رسمية وطريقة إعادة الإنتاج.</p>
          <div class="docs-links"><a class="ghost sm" href="https://github.com/PYTHON01100100/PodFit/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener">دليل المساهمة الكامل ↗</a><a class="ghost sm" href="https://github.com/PYTHON01100100/PodFit/issues/new" target="_blank" rel="noopener">فتح issue ↗</a></div>
          <pre>py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --reload --port 8510</pre>
        </section>
        <p class="docs-foot">podfit مرخص تحت GPL-3.0. راجع <code>LICENSE</code> في المستودع.</p>
      </article>
    </div>`;
}
function renderDocsView() {
  if (isAr()) { renderDocsArabicView(); return; }
  $('#view-docs').dataset.built = '1';
  $('#view-docs').innerHTML = `
    <div class="view-head"><div><h2>Docs</h2>
      <p>About the project, hardware reference, and how to contribute.</p></div></div>
    <div class="docs-layout">
      <nav class="docs-nav">
        <a href="#docs-about">About</a>
        <a href="#docs-hw">Hardware reference</a>
        <a href="#docs-interconnects" class="sub">Interconnects</a>
        <a href="#docs-apple" class="sub">Apple Silicon</a>
        <a href="#docs-nvidia" class="sub">NVIDIA &amp; L20</a>
        <a href="#docs-cloud" class="sub">Cloud landscape</a>
        <a href="#docs-providers">Cloud providers</a>
        <a href="#docs-aws" class="sub">AWS</a>
        <a href="#docs-azure" class="sub">Azure</a>
        <a href="#docs-gcp" class="sub">GCP</a>
        <a href="#docs-alibaba" class="sub">Alibaba Cloud</a>
        <a href="#docs-engines">LLM engines</a>
        <a href="#docs-vllm" class="sub">vLLM</a>
        <a href="#docs-sglang" class="sub">SGLang</a>
        <a href="#docs-llamacpp" class="sub">llama.cpp</a>
        <a href="#docs-ollama" class="sub">Ollama</a>
        <a href="#docs-contributing">Contributing</a>
      </nav>
      <article class="docs-content">

        <section id="docs-about">
          <h3>About</h3>
          <p>podfit answers one question: <b>can this model run on my hardware, and how
            well?</b> Point it at any model on HuggingFace, describe the GPUs you have, choose
            an engine and quantization, and it computes the real memory breakdown — weights, KV
            cache, activations, CUDA graphs, framework overhead — tells you what fits, how much
            context and concurrency you get, roughly how fast it will run, and the exact command
            to launch it. Nothing is invented: every number traces back to the Hub's real
            <code>config.json</code>/safetensors metadata, a benchmark provider, or a vendor spec
            sheet, labelled as an estimate when it has to be one.</p>
          <p>Built by Abdulrahman Almyman, an AI/cloud engineer.</p>
          <div class="docs-links">
            <a class="ghost sm" href="https://www.abdulrahmanalmyman.dev/" target="_blank" rel="noopener">Website ↗</a>
            <a class="ghost sm" href="https://www.linkedin.com/in/abdulrahmanalmyman/" target="_blank" rel="noopener">LinkedIn ↗</a>
            <a class="ghost sm" href="https://x.com/PYTHON01100100" target="_blank" rel="noopener">X ↗</a>
            <a class="ghost sm" href="https://github.com/PYTHON01100100" target="_blank" rel="noopener">GitHub ↗</a>
            <a class="ghost sm" href="https://github.com/PYTHON01100100/PodFit" target="_blank" rel="noopener">Project repo ↗</a>
            <a class="ghost sm" href="https://buymeacoffee.com/AbdulrahmanAlmyman" target="_blank" rel="noopener">Buy me a coffee ↗</a>
          </div>
        </section>

        <section id="docs-hw">
          <h3>Hardware reference</h3>
        </section>

        <section id="docs-interconnects">
          <h4>Interconnects: NVLink, RDMA, PCIe, Ethernet</h4>
          <p>What links the GPUs together changes how far tensor parallelism (TP) can scale, and
            podfit's <b>Interconnect</b> field (Setup → Hardware → the advanced disclosure) drives
            the warnings and blockers it raises about that.</p>
          <ul>
            <li><b>NVLink / NVSwitch</b> — NVIDIA's proprietary GPU-to-GPU fabric, the fastest
              option and the one TP really wants. Bandwidth depends on the SKU: 900 GB/s on
              H100/H200 SXM, 1.8 TB/s on Blackwell SXM (B100/B200/GB200). Many PCIe-form-factor
              cards (L40S, L4, RTX 6000 Ada, the RTX 50-series) ship with none at all — TP on
              those falls back to PCIe and degrades hard past a few GPUs.</li>
            <li><b>RDMA (InfiniBand / RoCE, with GPUDirect RDMA)</b> — the standard way to extend
              TP <i>across nodes</i>: direct GPU-to-GPU memory access over the network, bypassing
              the CPU. Needs 200 Gb/s+ to be worth it; below that, prefer pipeline parallelism
              (PP) across nodes instead, since PP only ships one activation tensor per stage
              boundary rather than an all-reduce every layer.</li>
            <li><b>RDMA over Thunderbolt 5 (Apple Silicon)</b> — a newer entry in the same
              dropdown, for clustering multiple Mac Studios rather than NVIDIA GPUs. Apple's own
              MLX research uses exactly this transport for distributed inference across
              machines.</li>
            <li><b>PCIe only / Ethernet</b> — no dedicated GPU fabric. PCIe-only is fine inside
              one box for PP; plain Ethernet is viable for PP across nodes but not for TP, which
              podfit flags as a blocker if you try it.</li>
          </ul>
        </section>

        <section id="docs-apple">
          <h4>Apple Silicon: unified memory, UltraFusion, and clustering</h4>
          <p>podfit treats Apple Silicon as its own vendor family ("Metal", a synthetic
            compute-capability sentinel), gated to llama.cpp and MLX only — vLLM and SGLang are
            CUDA/ROCm-only and will never run on it.</p>
          <ul>
            <li><b>Unified memory.</b> CPU, GPU and Neural Engine share one memory pool; by
              convention (and by default in every Apple catalog entry) ~75% of it is treated as
              GPU-addressable, matching what llama.cpp/MLX actually get to use.</li>
            <li><b>Max vs. Ultra.</b> A Max chip is a single die. An Ultra chip fuses two Max dies
              together over Apple's UltraFusion interposer — roughly double the CPU/GPU/Neural
              Engine cores and double the memory bandwidth of the Max it's built from, not a
              multi-GPU setup in the NVLink sense. M5 Ultra is the first Ultra built this way with
              per-core Neural Accelerators (matrix-multiply units) in its GPU, which speed up
              prefill far more than decode — decode stays bandwidth-bound either way.</li>
            <li><b>Clustering multiple Macs.</b> For models too large for even a single 512 GB
              M3/M5 Ultra, llama.cpp's RPC backend or MLX's distributed runtime can pool several
              machines over Thunderbolt 5 (RDMA) or plain Ethernet. Set <b>nodes</b> &gt; 1 in
              podfit to model this: memory pools across machines (layer-split, same as multi-GPU
              llama.cpp), but the speed figures still assume each layer's compute stays within
              <i>one</i> chip's own bandwidth — the inter-machine transfer time itself isn't
              simulated, which is exactly what the finding podfit raises in this situation
              ("N separate machines, not one box") says.</li>
          </ul>
        </section>

        <section id="docs-nvidia">
          <h4>NVIDIA: compute capability, architecture families, and the L20</h4>
          <p>Every NVIDIA kernel-availability gate in podfit (FlashAttention-2, FP8, sparse-MLA,
            and so on) keys off <b>CUDA compute capability</b> (<code>sm_XX</code>), not the
            marketing name. Roughly, oldest to newest: Volta (<code>sm_70</code>) → Turing
            (<code>sm_75</code>) → Ampere (<code>sm_80</code>/<code>sm_86</code>) → Ada
            (<code>sm_89</code>) → Hopper (<code>sm_90</code>) → Blackwell (<code>sm_100</code>
            datacenter / <code>sm_120</code> RTX-class). NVLink availability is a function of SKU
            and form factor, not architecture alone — SXM parts get it, most PCIe and all
            consumer cards don't.</p>
          <p><b>The L20</b> is an Ada-generation (AD102) card built specifically for the China
            market after US export controls cut off H100/A100-class chips there: 48 GB GDDR6 at
            864 GB/s, FP8-capable (<code>sm_89</code>) but roughly a quarter of an H100's dense
            tensor FLOPS, and <b>no NVLink</b> — tensor-parallel traffic crosses PCIe Gen4 x16, so
            TP &gt; 4 scales poorly on it. It has compliance-driven siblings in the catalog with
            the same story: H20/H20-141 (Hopper-class HBM bandwidth, but tensor FLOPS cut to a
            fraction of H100's), H800 (an H100 with NVLink capped at 400 GB/s instead of 900),
            A800 (an A100 with NVLink capped at 400 GB/s). podfit sizes all of these against their
            real, cut-down numbers — not the uncapped chip's spec sheet.</p>
        </section>

        <section id="docs-cloud">
          <h4>Renting GPUs: the cloud landscape</h4>
          <ul>
            <li><b>Hyperscalers</b> (AWS, Azure, GCP, Oracle Cloud) — the broadest catalog and the
              strongest reliability/compliance guarantees, generally the most expensive
              on-demand. Worth it when you need the rest of their ecosystem alongside the GPU, not
              just the GPU.</li>
            <li><b>GPU-specialist "neoclouds"</b> (Lambda, CoreWeave, Nebius, Crusoe, Voltage
              Park, and similar) — a narrower catalog focused on current NVIDIA silicon, usually
              meaningfully cheaper than hyperscalers for the same card, and often first to get new
              SKUs (H200, B200, GB200) in volume.</li>
            <li><b>Marketplaces / spot platforms</b> (RunPod, Vast.ai, Paperspace) — the cheapest
              on-demand pricing and the most variable reliability and availability. A good fit for
              a short sizing or benchmarking run, a risky one for production serving.</li>
            <li><b>Regional / compliance-driven clouds</b> (Alibaba Cloud, Tencent Cloud, Huawei
              Cloud) — the primary source for the export-compliant chips in podfit's catalog (H20,
              H800, L20), since those SKUs largely don't exist outside China-market
              deployments.</li>
            <li><b>Bare metal / colo</b> — full control and the best $/GPU-hour at scale, but you
              own the power, cooling, networking, and NVLink/InfiniBand fabric design yourself.
              It's also the only path to an Apple Silicon cluster — no cloud provider rents Mac
              Studios by the hour at any real scale.</li>
          </ul>
        </section>

        <section id="docs-providers">
          <h3>Cloud providers</h3>
          <p>Instance-family names for the four clouds most people size against, so you know
            what to search for once podfit tells you what fits. Naming and availability vary by
            region — check each provider's own docs before committing to one.</p>
        </section>

        <section id="docs-aws">
          <h4>AWS</h4>
          <p><b>P5</b> (H100 SXM, 8 per node), <b>P5e</b>/<b>P5en</b> (H200 SXM), and the newer
            <b>P6-B200</b> (Blackwell) cover the hyperscaler's current NVIDIA lineup, plus
            Capacity Blocks for reserving GPU time in advance rather than competing for on-demand
            capacity. AWS also has its own silicon (Trainium/Inferentia) outside podfit's
            NVIDIA/AMD/Apple catalog — relevant if you're training rather than serving with
            vLLM/SGLang/llama.cpp. Worth it when you need the rest of the AWS ecosystem (IAM,
            VPC, S3, SageMaker) alongside the GPU.</p>
        </section>

        <section id="docs-azure">
          <h4>Azure</h4>
          <p><b>ND H100 v5</b> and <b>ND H200 v5</b> cover Hopper; Blackwell ships as
            <b>ND GB200 v6</b> — a Grace-CPU-plus-Blackwell configuration with four B200-class
            GPUs per VM rather than a standalone B200 instance, which matters when you're
            estimating per-GPU cost. Microsoft's tight partnership with OpenAI tends to put new
            NVIDIA silicon on Azure early and in volume.</p>
        </section>

        <section id="docs-gcp">
          <h4>GCP</h4>
          <p><b>A3 Mega</b> (<code>a3-megagpu-8g</code>, H100), <b>A3 Ultra</b>
            (<code>a3-ultragpu-8g</code>, H200), and <b>A4</b> (<code>a4-highgpu-8g</code>, B200)
            are the current NVIDIA families. GCP also offers its own TPU silicon as an
            alternative to GPU-based serving entirely — outside podfit's scope, since the app
            sizes against the real GPUs vLLM/SGLang/llama.cpp actually run on, not custom
            accelerators with their own software stack.</p>
        </section>

        <section id="docs-alibaba">
          <h4>Alibaba Cloud</h4>
          <p>The most directly relevant provider to podfit's catalog: <b>GN8is</b>/<b>GN8v</b>
            instances run the <b>L20</b> in 1/2/4/8-GPU configurations — the exact card podfit
            models with its real, cut-down numbers (see the NVIDIA &amp; L20 section above).
            <b>GN7i</b> runs the A10 (inference-optimized). A separate higher-end line
            (<code>gx8sf</code>/"L20X") offers 141 GB with NVLink and 3.2 Tbps RDMA for
            large-scale training and inference. Alibaba Cloud, alongside Tencent and Huawei Cloud,
            is also the primary place you'll actually find the other export-compliant chips in
            podfit's catalog — H20 and H800 — since those SKUs exist specifically for this market.
            Some instance families are region-restricted (overseas-only, or mainland-only) —
            check availability for your account before planning around one.</p>
        </section>

        <section id="docs-engines">
          <h3>LLM engines</h3>
          <p>The four engines podfit can size against, with exactly the OS and weight-format
            support it models them with (<code>app/catalog.py</code>'s <code>ENGINES</code>) —
            not a generic engine comparison, but what actually constrains the dropdown.</p>
        </section>

        <section id="docs-vllm">
          <h4>vLLM</h4>
          <p>Linux only. Loads fp16/bf16/fp32/fp8/int8/AWQ/GPTQ (4 &amp; 8-bit)/bitsandbytes
            NF4/MXFP4/NVFP4. podfit's default "server" engine: PagedAttention, continuous
            batching, tensor/pipeline parallelism over NCCL. This is the engine podfit's
            server-style memory model (<code>gpu_memory_utilization</code> budget, CUDA graph
            reservation, KV cache sizing) is built around.</p>
        </section>

        <section id="docs-sglang">
          <h4>SGLang</h4>
          <p>Linux only. Loads fp16/bf16/fp8/int8/AWQ/GPTQ/MXFP4/NVFP4 — no fp32, no
            bitsandbytes, unlike vLLM. RadixAttention gives it an edge on prefix-heavy workloads
            (shared system prompts, few-shot examples). podfit sizes it with the same
            server-style memory model as vLLM.</p>
        </section>

        <section id="docs-llamacpp">
          <h4>llama.cpp</h4>
          <p>Linux, Windows, and macOS. Loads GGUF quantizations (Q8_0 down to Q2_K, IQ3/IQ4
            variants) plus fp16. The only engine with partial GPU offload (some layers on GPU,
            the rest on CPU/system RAM) and the only one Apple Silicon can use, via its Metal
            backend. podfit models it with a layer-split memory calculation, not tensor-parallel
            — see the Apple Silicon section above for what that means for multi-GPU/multi-machine
            setups.</p>
        </section>

        <section id="docs-ollama">
          <h4>Ollama</h4>
          <p>Linux, Windows, and macOS. GGUF-only — no fp16 option, unlike llama.cpp itself.
            It's a llama.cpp wrapper with a friendlier CLI/API and model-pull workflow; podfit
            sizes it identically to llama.cpp under the hood. MLX is a real alternative for
            Apple Silicon clustering (see Apple Silicon above) but isn't a selectable engine in
            podfit yet — there's no MLX entry in <code>ENGINES</code>.</p>
        </section>

        <section id="docs-contributing">
          <h3>Contributing</h3>
          <p>podfit is a small, single-maintainer project — contributions are welcome, but keep
            changes focused and in line with what's already here. The full guide lives in
            <code>CONTRIBUTING.md</code> in the repo; the short version:</p>
          <h4>Before you open a PR</h4>
          <ul>
            <li><b>Bug reports</b> — open an issue with the model repo, GPU, engine, and runtime
              settings that produced the wrong result, plus what you expected.</li>
            <li><b>New hardware</b> — GPU catalog entries need a real source (a vendor spec sheet,
              an official announcement, or a well-corroborated benchmark). Don't guess at
              bandwidth or TFLOPS; label an estimate as one.</li>
            <li><b>New model architectures</b> — need the engine's actual kernel requirements, not
              assumptions from the model card.</li>
            <li><b>Features</b> — open an issue first for anything beyond a bug fix.</li>
          </ul>
          <h4>Dev setup</h4>
          <pre>python -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.txt
PODFIT_DB=./data/podfit.db PODFIT_SEED=./data/detected_hardware.json \\
  uvicorn app.main:app --reload --port 8510</pre>
          <p>No build step for the frontend — it's plain HTML/CSS/JS, edit and reload.</p>
          <h4>Code conventions</h4>
          <ul>
            <li>No comments unless they explain a non-obvious <i>why</i> — a hidden constraint, a
              workaround, an invariant that isn't clear from the code itself.</li>
            <li>Dense, technical prose everywhere user-facing text appears (findings, docstrings)
              — no filler, no generic error-message tone.</li>
            <li>"Nothing is invented" is the project's core promise. Any number the app shows
              should trace back to a real source or be clearly labeled as an estimate and why.</li>
            <li>Keep PRs scoped to one change.</li>
          </ul>
          <div class="docs-links">
            <a class="ghost sm" href="https://github.com/PYTHON01100100/PodFit/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener">Full contributing guide ↗</a>
            <a class="ghost sm" href="https://github.com/PYTHON01100100/PodFit/issues" target="_blank" rel="noopener">Open an issue ↗</a>
          </div>
          <p class="docs-foot">podfit is licensed under GPL-3.0. See <code>LICENSE</code> in the
            repo.</p>
        </section>

      </article>
    </div>`;
}
