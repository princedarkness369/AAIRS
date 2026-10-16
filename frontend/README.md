# AAIRS — Arabic Audio Information Retrieval System (Frontend)

A static frontend for the project described in your document: a search
interface over Arabic audio transcripts, showing ranked results with
category, cosine-similarity score, transcript, and audio playback — plus
a pipeline overview, an inverted-index preview, and a Precision/Recall/F1
evaluation panel.

This is **frontend only** — no server code. It currently runs in a local
**demo mode** (client-side JS recomputes tokenization → TF-IDF → cosine
similarity → ranking over `data/transcripts.json`) so the page is fully
functional to look at and click through before your backend is wired up.

## Files

```
frontend/
├── index.html          the whole page
├── css/style.css        design system (colors, type, layout)
├── js/main.js            search logic + rendering + API connector
├── data/transcripts.json sample Arabic transcripts (demo mode only)
└── audio/                 put matching .mp3 files here for demo playback
```

## Connect it to your model

Open `js/main.js` and set:

```js
const API_BASE_URL = "https://your-backend-url/api";
```

Once set, the frontend stops using the local demo pipeline and calls your
backend instead. Your backend needs to expose:

| Method | Endpoint         | Request body                          | Response shape |
|--------|------------------|----------------------------------------|-----------------|
| GET    | `/stats`         | —                                      | `{ "total_documents": 50, "vocabulary_size": 812 }` |
| POST   | `/search`        | `{ "query": "...", "top_k": 10 }`     | `{ "results": [ { "doc_id", "category", "category_ar", "similarity", "transcript", "audio_file" }, ... ] }` |
| GET    | `/index-sample`  | —                                      | `[ { "term": "الذكاء", "doc_ids": ["AI_01","AI_07"] }, ... ]` |
| POST   | `/evaluate`      | `{ "query": "...", "category": "AI" }` | `{ "precision": 0.87, "recall": 0.7, "f1": 0.78 }` |

If your endpoint names, routes, or field names differ, adjust the four
`api*` functions near the top of `js/main.js` (`apiSearch`, `apiStats`,
`apiIndexSample`, `apiEvaluate`) — everything else (rendering, layout,
events) stays the same either way.

`audio_file` can be either a filename (resolved against `AUDIO_BASE_PATH`
in `main.js`, currently `audio/`) or a full `http(s)://` URL — the
frontend detects which and plays it directly.

## Running it locally

Because the page fetches `data/transcripts.json`, open it through a local
server rather than double-clicking the file (browsers block `fetch` on
`file://`):

```bash
cd frontend
python3 -m http.server 8080
# then open http://localhost:8080
```

## Swapping in your real 50-document dataset

Replace `data/transcripts.json` with your Speech-to-Text output, one
entry per audio file:

```json
{
  "doc_id": "AI_01",
  "category": "AI",
  "category_ar": "الذكاء الاصطناعي",
  "audio_file": "AI_01.mp3",
  "transcript": "يحاكي الذكاء الاصطناعي القدرات الذهنية للبشر."
}
```

and drop the matching `.mp3` files into `audio/`. This keeps demo mode
useful as a fallback/staging view even after your backend exists.
