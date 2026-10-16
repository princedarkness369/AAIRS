
/* ==========================================================================
   AAIRS — frontend logic
   ==========================================================================
   HOW TO CONNECT YOUR MODEL
   --------------------------------------------------------------------------
   Set API_BASE_URL below to your backend's base URL. This frontend expects
   three endpoints (adjust the fetch calls further down if your routes or
   payload shapes differ):

     GET  {API_BASE_URL}/stats
          -> { "total_documents": 50, "vocabulary_size": 812 }

     POST {API_BASE_URL}/search      body: { "query": "...", "top_k": 10 }
          -> { "results": [
                 { "doc_id": "AI_01", "category": "AI", "category_ar": "...",
                   "similarity": 0.91, "transcript": "...", "audio_file": "AI_01.mp3" },
                 ...
               ] }

     GET  {API_BASE_URL}/index-sample
          -> [ { "term": "الذكاء", "doc_ids": ["AI_01","AI_07"] }, ... ]

     POST {API_BASE_URL}/evaluate    body: { "query": "...", "category": "AI" }
          -> { "precision": 0.87, "recall": 0.7, "f1": 0.78 }

   If API_BASE_URL is left as null, the page runs in DEMO MODE: it loads
   data/transcripts.json and computes everything (tokenizing, TF-IDF,
   cosine similarity, evaluation) locally in the browser, purely so the
   interface has something real to show while your backend isn't wired up.
   Demo mode is not meant to replace your actual IR pipeline.
   ========================================================================== */

const API_BASE_URL = "http://127.0.0.1:8000/api";
const AUDIO_BASE_PATH = "audio/"; // folder containing the mp3 files, used in demo mode

const CATEGORY_LABELS = {
  AI: "الذكاء الاصطناعي",
  IR: "استرجاع المعلومات",
  HEALTHY: "الصحة",
  EDU: "التعليم",
  SPORT: "الرياضة",
};

// ---------------------------------------------------------------------------
// Demo-mode Arabic IR pipeline (mirrors the phases in the project document)
// ---------------------------------------------------------------------------

const STOPWORDS = new Set([
  "في", "من", "إلى", "على", "عن", "أن", "إن", "كان", "هذا", "هذه", "ذلك", "التي", "الذي",
  "و", "ثم", "أو", "لا", "ما", "لم", "لن", "كل", "بعض", "حتى", "مع", "عند", "قد", "لقد",
  "هو", "هي", "هم", "نحن", "أنت", "أنتم", "كما", "بين", "أيضا", "حيث", "لكن", "إذا",
  "كانت", "يكون", "له", "لها", "إلا", "بل", "قبل", "بعد", "غير",
]);

function normalizeArabic(text) {
  return text
    .replace(/[\u064B-\u0652\u0670\u0640]/g, "")
    .replace(/[إأآا]/g, "ا")
    .replace(/ى/g, "ي")
    .replace(/ة/g, "ه")
    .replace(/ؤ/g, "و")
    .replace(/ئ/g, "ي")
    .replace(/[^\w\s\u0600-\u06FF]/g, " ")
    .replace(/\d+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(text) {
  return normalizeArabic(text)
    .split(" ")
    .filter((t) => t && !STOPWORDS.has(t));
}

let demoCorpus = null;

async function loadDemoCorpus() {
  if (demoCorpus) return demoCorpus;
  const res = await fetch("data/transcripts.json");
  const raw = await res.json();
  demoCorpus = raw.map((doc) => ({ ...doc, tokens: tokenize(doc.transcript) }));
  return demoCorpus;
}

function buildTfidf(corpus) {
  const df = new Map();
  corpus.forEach((doc) => {
    new Set(doc.tokens).forEach((term) => df.set(term, (df.get(term) || 0) + 1));
  });
  const n = corpus.length;
  const idf = new Map();
  df.forEach((freq, term) => idf.set(term, Math.log((1 + n) / (1 + freq)) + 1));

  const vectors = new Map();
  corpus.forEach((doc) => {
    const tf = new Map();
    doc.tokens.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    const total = doc.tokens.length || 1;
    const vec = new Map();
    tf.forEach((count, term) => vec.set(term, (count / total) * (idf.get(term) || 0)));
    let norm = 0;
    vec.forEach((v) => (norm += v * v));
    vectors.set(doc.doc_id, { vec, norm: Math.sqrt(norm) || 1 });
  });

  return { idf, vectors };
}

function vectorizeQuery(tokens, idf) {
  const tf = new Map();
  tokens.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
  const total = tokens.length || 1;
  const vec = new Map();
  tf.forEach((count, term) => vec.set(term, (count / total) * (idf.get(term) || 0)));
  let norm = 0;
  vec.forEach((v) => (norm += v * v));
  return { vec, norm: Math.sqrt(norm) || 1 };
}

function cosineSimilarity(a, b) {
  let dot = 0;
  a.vec.forEach((val, term) => {
    if (b.vec.has(term)) dot += val * b.vec.get(term);
  });
  if (a.norm === 0 || b.norm === 0) return 0;
  return dot / (a.norm * b.norm);
}

async function demoSearch(query, topK = 10) {
  const corpus = await loadDemoCorpus();
  const { idf, vectors } = buildTfidf(corpus);
  const qTokens = tokenize(query);
  const qVector = vectorizeQuery(qTokens, idf);

  const scored = corpus.map((doc) => ({
    doc,
    score: cosineSimilarity(qVector, vectors.get(doc.doc_id)),
  }));
  scored.sort((a, b) => b.score - a.score);

  return scored
    .filter((s) => s.score > 0)
    .slice(0, topK)
    .map((s) => ({
      doc_id: s.doc.doc_id,
      category: s.doc.category,
      category_ar: s.doc.category_ar || CATEGORY_LABELS[s.doc.category],
      similarity: Math.round(s.score * 10000) / 10000,
      transcript: s.doc.transcript,
      audio_file: s.doc.audio_file,
    }));
}

async function demoStats() {
  const corpus = await loadDemoCorpus();
  const vocab = new Set();
  corpus.forEach((d) => d.tokens.forEach((t) => vocab.add(t)));
  return { total_documents: corpus.length, vocabulary_size: vocab.size };
}

async function demoIndexSample(limit = 16) {
  const corpus = await loadDemoCorpus();
  const index = new Map();
  corpus.forEach((doc) => {
    new Set(doc.tokens).forEach((term) => {
      if (!index.has(term)) index.set(term, []);
      index.get(term).push(doc.doc_id);
    });
  });
  return [...index.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, limit)
    .map(([term, doc_ids]) => ({ term, doc_ids }));
}

async function demoEvaluate(query, category) {
  const corpus = await loadDemoCorpus();
  const totalRelevant = corpus.filter((d) => d.category === category).length;
  const results = await demoSearch(query, 10);
  const relevantRetrieved = results.filter((r) => r.category === category).length;
  const retrieved = results.length;
  const precision = retrieved ? relevantRetrieved / retrieved : 0;
  const recall = totalRelevant ? relevantRetrieved / totalRelevant : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return {
    precision: Math.round(precision * 1000) / 1000,
    recall: Math.round(recall * 1000) / 1000,
    f1: Math.round(f1 * 1000) / 1000,
  };
}

// ---------------------------------------------------------------------------
// API wrapper — routes to your backend if API_BASE_URL is set, else demo mode
// ---------------------------------------------------------------------------

async function apiSearch(query, topK = 10) {
  if (!API_BASE_URL) return demoSearch(query, topK);
  const res = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  const data = await res.json();
  return data.results || [];
}

async function apiStats() {
  if (!API_BASE_URL) return demoStats();
  const res = await fetch(`${API_BASE_URL}/stats`);
  return res.json();
}

async function apiIndexSample() {
  if (!API_BASE_URL) return demoIndexSample();
  const res = await fetch(`${API_BASE_URL}/index-sample`);
  return res.json();
}

async function apiEvaluate(query, category) {
  if (!API_BASE_URL) return demoEvaluate(query, category);
  const res = await fetch(`${API_BASE_URL}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, category }),
  });
  return res.json();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function buildWaveform() {
  const el = document.getElementById("waveform");
  const bars = 42;
  let html = "";
  for (let i = 0; i < bars; i++) {
    const h = 12 + Math.round(Math.abs(Math.sin(i * 0.4)) * 40 + Math.random() * 8);
    const delay = (i * 0.03).toFixed(2);
    html += `<span style="height:${h}px;animation:wave 1.6s ease-in-out ${delay}s infinite"></span>`;
  }
  el.innerHTML = html;

  const styleTag = document.createElement("style");
  styleTag.textContent = `
    @keyframes wave {
      0%, 100% { transform: scaleY(0.6); opacity: 0.7; }
      50% { transform: scaleY(1); opacity: 1; }
    }
  `;
  document.head.appendChild(styleTag);
}

function renderStats(stats) {
  document.querySelectorAll("[data-stat]").forEach((el) => {
    const key = el.getAttribute("data-stat");
    if (stats[key] !== undefined) el.textContent = stats[key];
  });
}

function renderIndexSample(items) {
  const el = document.getElementById("index-table");
  if (!items.length) {
    el.innerHTML = `<p class="empty-state">لا توجد بيانات فهرس بعد.</p>`;
    return;
  }
  el.innerHTML = items
    .map(
      (item) => `
      <div class="index-term">
        <span class="index-term__word">${escapeHtml(item.term)}</span>
        <span class="index-term__count">${item.doc_ids.length}</span>
      </div>`
    )
    .join("");
}

function renderResults(query, results) {
  const section = document.getElementById("results-section");
  const list = document.getElementById("results-list");
  const meta = document.getElementById("results-meta");
  const empty = document.getElementById("empty-state");

  section.hidden = false;
  meta.textContent = `الاستعلام: «${query}» — ${results.length} نتيجة`;

  if (!results.length) {
    list.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;

  list.innerHTML = results
    .map((r, i) => {
      const pct = Math.round(r.similarity * 100);
      const catLabel = r.category_ar || CATEGORY_LABELS[r.category] || r.category;
      const rawAudio = (r.audio_file || `${r.doc_id}.wav`).replace(/\.mp3$/i, ".wav");
      const audioSrc = /^https?:\/\//.test(rawAudio) ? rawAudio : `audio/${rawAudio.split('/').pop()}`;
      return `
        <article class="result-card">
          <span class="result-card__rank">${i + 1}</span>
          <div class="result-card__body">
            <div class="result-card__meta">
              <span class="result-card__id">${escapeHtml(r.doc_id)}</span>
              <span class="category-chip" data-cat="${escapeHtml(r.category)}">${escapeHtml(catLabel)}</span>
            </div>
            <p class="result-card__transcript">${escapeHtml(r.transcript)}</p>
          </div>
          <div class="result-card__right">
            <div class="similarity">
              <span class="similarity__value">${r.similarity.toFixed(4)}</span>
              <div class="similarity__bar"><div class="similarity__bar-fill" style="width:${pct}%"></div></div>
            </div>
            <audio controls preload="none" src="${audioSrc}"></audio>
          </div>
        </article>`;
    })
    .join("");
}

function renderMetrics(metrics) {
  document.getElementById("metrics").hidden = false;
  document.getElementById("metric-precision").textContent = metrics.precision ?? "—";
  document.getElementById("metric-recall").textContent = metrics.recall ?? "—";
  document.getElementById("metric-f1").textContent = metrics.f1 ?? "—";
}

function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  buildWaveform();

  if (!API_BASE_URL) {
    document.getElementById("demo-flag").hidden = false;
  }

  try {
    const stats = await apiStats();
    renderStats(stats);
  } catch (e) {
    console.warn("تعذر تحميل إحصاءات المجموعة:", e);
  }

  try {
    const sample = await apiIndexSample();
    renderIndexSample(sample);
  } catch (e) {
    console.warn("تعذر تحميل عينة الفهرس المعكوس:", e);
  }

  document.getElementById("search-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const query = document.getElementById("query-input").value.trim();
    if (!query) return;
    try {
      const results = await apiSearch(query, 10);
      renderResults(query, results);
      document.getElementById("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      console.error(e);
      renderResults(query, []);
    }
  });

  document.getElementById("evaluate-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const query = document.getElementById("eval-query").value.trim();
    const category = document.getElementById("eval-category").value;
    if (!query) return;
    try {
      const metrics = await apiEvaluate(query, category);
      renderMetrics(metrics);
    } catch (e) {
      console.error(e);
    }
  });
});
