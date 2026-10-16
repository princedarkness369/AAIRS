# AAIRS — Arabic Audio Information Retrieval System
نظام استرجاع المعلومات الصوتية العربية

An end-to-end Arabic Audio Information Retrieval (IR) System that processes Arabic speech/audio records, transcribes them using speech recognition (Whisper), normalizes Arabic text, builds an inverted index, and performs ranking and retrieval using TF-IDF and Cosine Similarity. The system includes an interactive web interface with audio playback, retrieval scoring, index inspection, and performance evaluation metrics.

---

## 🚀 Features

- **Audio Speech-to-Text (STT):** Automatic transcription of Arabic audio using OpenAI's Whisper model.
- **Arabic Text Normalization:** Comprehensive preprocessing including diacritics removal (Tashkeel), Tatweel stripping, letter normalization (Alef, Yaa, Taa Marbuta), and stopword filtering.
- **Vector Space Model (VSM) & Ranking:** TF-IDF representation with Cosine Similarity ranking for accurate query matching.
- **Inverted Index Engine:** Fast index generation mapping vocabulary terms to documents.
- **Evaluation Module:** Computes Precision, Recall, and F1-score across document categories.
- **Interactive Web Interface:** Modern UI for searching, filtering, audio playback, inspecting inverted index terms, and testing evaluation metrics.

---

## 📁 Project Structure

```text
AAIRS/
├── backend_files/
│   ├── main.py                  # FastAPI backend server & IR pipeline
│   ├── model_definition.py      # Neural network / encoder architecture
│   ├── actigraphy_encoder.pth   # Pre-trained encoder weights
│   └── actigraphy_index.pkl     # Pre-computed index
├── frontend/
│   ├── index.html               # Main user interface
│   ├── css/style.css            # Stylesheet and layout
│   ├── js/main.js               # Frontend search and API interactions
│   ├── data/transcripts.json    # Transcribed dataset
│   └── audio/                   # Processed WAV audio files for playback
├── audios/
│   ├── ai.audio/                # Raw audio files (Artificial Intelligence)
│   ├── edu.audio/               # Raw audio files (Education)
│   ├── health.audio/            # Raw audio files (Healthcare)
│   ├── ir.audio/                # Raw audio files (Information Retrieval)
│   ├── sport.audio/             # Raw audio files (Sports & Fitness)
│   ├── organize_and_copy.py     # Script to organize & rename audio files
│   └── reco_audio.py            # STT matching & alignment script
├── metadata.csv                 # Dataset metadata and annotations
├── .gitignore                   # Git ignore configurations
└── README.md                    # Project documentation
```

---

## 🛠️ Getting Started

### 1. Prerequisites

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) (for audio processing with Whisper)

### 2. Backend Setup

Install the required Python libraries:

```bash
pip install fastapi uvicorn openai-whisper torch scikit-learn numpy pydantic python-multipart
```

Run the backend server:

```bash
cd backend_files
python -m uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`).

### 3. Frontend Setup

Run the frontend using any static file server:

```bash
cd frontend
python -m http.server 8080
```

Open your browser at `http://localhost:8080`.

---

## 📊 Dataset Categories

The dataset includes Arabic audio recordings categorized into:
- **AI (الذكاء الاصطناعي):** Artificial Intelligence and Deep Learning concepts.
- **IR (استرجاع المعلومات):** Information Retrieval, indexing, and search engines.
- **HEALTHY (الصحة والتغذية):** Health, nutrition, and wellness.
- **EDU (التعليم):** Educational methodologies and e-learning.
- **SPORT (الرياضة واللياقة):** Exercise, physical training, and fitness.

---

## 📄 License

This project is developed for Information Retrieval academic coursework and research.
