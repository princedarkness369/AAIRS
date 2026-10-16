import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import numpy as np
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# Ensure ffmpeg is available for Whisper audio loading
if not shutil.which("ffmpeg"):
    try:
        import importlib
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = os.path.dirname(ffmpeg_exe)
        target_exe = os.path.join(bin_dir, 'ffmpeg.exe')
        if not os.path.exists(target_exe):
            shutil.copyfile(ffmpeg_exe, target_exe)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

app = FastAPI(title="AAIRS API")

# تحميل نموذج تفريغ الصوت العربي
stt_model = whisper.load_model("base")

# السماح للواجهة بالاتصال بالـ API بدون حظر من المتصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # هذا السطر مهم جداً لتخطي الحظر
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STOPWORDS = {
    "في", "من", "إلى", "على", "عن", "أن", "إن", "كان", "هذا", "هذه", "ذلك", "التي", "الذي",
    "و", "ثم", "أو", "لا", "ما", "لم", "لن", "كل", "بعض", "حتى", "مع", "عند", "قد", "لقد",
    "هو", "هي", "هم", "نحن", "أنت", "أنتم", "كما", "بين", "أيضا", "حيث", "لكن", "إذا",
    "كانت", "يكون", "له", "لها", "إلا", "بل", "قبل", "بعد", "غير"
}

def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    # إزالة التشكيل والتطويل
    text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
    # توحيد الألفات والهمزات
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    # تنظيف الرموز
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    tokens = text.strip().split()
    cleaned_tokens = []
    for t in tokens:
        if t in STOPWORDS or len(t) <= 1:
            continue
        # إزالة "ال" التعريف لزيادة دقة المطابقة في الاسترجاع
        if t.startswith("ال") and len(t) > 3:
            t = t[2:]
        cleaned_tokens.append(t)
        
    return " ".join(cleaned_tokens)

preprocess_arabic = normalize_arabic

# قاعدة بيانات الـ 50 تسجيلاً مع إضافة وسم التصنيف للنص المفهرس
CORPUS = [
    # الذكاء الاصطناعي (AI)
    {"doc_id": "AI_01", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "يحاكي الذكاء الاصطناعي القدرات الذهنية للبشر.", "audio_file": "AI_01.wav"},
    {"doc_id": "AI_02", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "يعتمد التعلم العميق على الشبكات العصبية.", "audio_file": "AI_02.wav"},
    {"doc_id": "AI_03", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "تعالج خوارزميات NLP النصوص واللغات البشرية.", "audio_file": "AI_03.wav"},
    {"doc_id": "AI_04", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "تحلل الرؤية الحاسوبية الصور والفيديوهات الرقمية.", "audio_file": "AI_04.wav"},
    {"doc_id": "AI_05", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "يواجه الذكاء الاصطناعي تحديات الانحياز والخصوصية .", "audio_file": "AI_05.wav"},
    {"doc_id": "AI_06", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "يركز التعلم المعزز على قرارات المكافأة.", "audio_file": "AI_06.wav"},
    {"doc_id": "AI_07", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "ينشئ الذكاءالتوليدي محتوى ونصوصاً جديدة.", "audio_file": "AI_07.wav"},
    {"doc_id": "AI_08", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "تسرع معالجات GPUs تدريب نماذج الضخمة.", "audio_file": "AI_08.wav"},
    {"doc_id": "AI_09", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "يقيس اختبار تورينج ذكاء وسلوك الآلات والذكاء الاصطناعي.", "audio_file": "AI_09.wav"},
    {"doc_id": "AI_10", "category": "AI", "category_ar": "الذكاء الاصطناعي", "transcript": "ترفع الأتمتة الذكية كفاءة الإنتاج بالمصانع.", "audio_file": "AI_10.wav"},

    # استرجاع المعلومات (IR)
    {"doc_id": "IR_01", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "يبحث استرجاع البيانات عن معلومات ملائمة.", "audio_file": "IR_01.wav"},
    {"doc_id": "IR_02", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "يمثل نموذج الفضاء المتجهي الوثائق رياضياً.", "audio_file": "IR_02.wav"},
    {"doc_id": "IR_03", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "تسرع الفهرسة المعكوسة البحث عن الكلمات.", "audio_file": "IR_03.wav"},
    {"doc_id": "IR_04", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "تقاس دقة استرجاع المعلومات بـ Precision وRecall.", "audio_file": "IR_04.wav"},
    {"doc_id": "IR_05", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "يفهم البحث الدلالي السياق ومعنى الاستعلام.", "audio_file": "IR_05.wav"},
    {"doc_id": "IR_06", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "تقيم خوارزمية TF-IDF أهمية الكلمة بالمستند.", "audio_file": "IR_06.wav"},
    {"doc_id": "IR_07", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "يقيس Cosine Similarity تشابه متجهات النصوص.", "audio_file": "IR_07.wav"},
    {"doc_id": "IR_08", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "ترتب خوارزمية BM25 نتائج البحث بدقة.", "audio_file": "IR_08.wav"},
    {"doc_id": "IR_09", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "تتضمن المعالجة المسبقة حذف كلمات التوقف.", "audio_file": "IR_09.wav"},
    {"doc_id": "IR_10", "category": "IR", "category_ar": "استرجاع المعلومات", "transcript": "يدمج نظام RAG استرجاع بالنماذج اللغوية.", "audio_file": "IR_10.wav"},

    # الصحة (HEALTHY)
    {"doc_id": "HEALTHY_01", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "يقوي الغذاء المتوازن مناعة جسم الإنسان.", "audio_file": "HEALTHY_01.wav"},
    {"doc_id": "HEALTHY_02", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "تقلل الرياضة مخاطر أمراض القلب.", "audio_file": "HEALTHY_02.wav"},
    {"doc_id": "HEALTHY_03", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "يجدد النوم الكافي خلايا الجسم والدماغ.", "audio_file": "HEALTHY_03.wav"},
    {"doc_id": "HEALTHY_04", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "تكشف الفحوصات الدورية الأمراض المزمنة مبكرا.", "audio_file": "HEALTHY_04.wav"},
    {"doc_id": "HEALTHY_05", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "يسبب الإفراط بالمضادات مناعة لدى البكتيريا.", "audio_file": "HEALTHY_05.wav"},
    {"doc_id": "HEALTHY_06", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "يحسن شرب الماء وظائف الكلى الحيوية.", "audio_file": "HEALTHY_06.wav"},
    {"doc_id": "HEALTHY_07", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "يرفع التوتر المستمر هرمون الكورتيزول الضار.", "audio_file": "HEALTHY_07.wav"},
    {"doc_id": "HEALTHY_08", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "توفر اللقاحات مناعة مجتمعية ضد الأوبئة.", "audio_file": "HEALTHY_08.wav"},
    {"doc_id": "HEALTHY_09", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "تزيد السمنة احتمالية الإصابة بآلام المفاصل   .", "audio_file": "HEALTHY_09.wav"},
    {"doc_id": "HEALTHY_10", "category": "HEALTHY", "category_ar": "الصحة", "transcript": "يصمم الطب الجينومي علاجات صحية مخصصة للأفراد.", "audio_file": "HEALTHY_10.wav"},

    # التعليم (EDU)
    {"doc_id": "EDU_01", "category": "EDU", "category_ar": "التعليم", "transcript": "ينمي التعليم التفاعلي التفكير النقدي للطلاب.", "audio_file": "EDU_01.wav"},
    {"doc_id": "EDU_02", "category": "EDU", "category_ar": "التعليم", "transcript": "يتيح التعليم الإلكتروني الوصول للمعلومات بسهولة.", "audio_file": "EDU_02.wav"},
    {"doc_id": "EDU_03", "category": "EDU", "category_ar": "التعليم", "transcript": "يدمج منهج STEM  العلوم التقنية والهندسة في التعليم.", "audio_file": "EDU_03.wav"},
    {"doc_id": "EDU_04", "category": "EDU", "category_ar": "التعليم", "transcript": "يتابع التقييم المستمر  تطور أداء المتعلم.", "audio_file": "EDU_04.wav"},
    {"doc_id": "EDU_05", "category": "EDU", "category_ar": "التعليم", "transcript": "يبني التعليم المبكر المهارات الاجتماعية للأطفال.", "audio_file": "EDU_05.wav"},
    {"doc_id": "EDU_06", "category": "EDU", "category_ar": "التعليم", "transcript": "يوفر الواقع الافتراضي تجارب تعليمية تفاعلية  .", "audio_file": "EDU_06.wav"},
    {"doc_id": "EDU_07", "category": "EDU", "category_ar": "التعليم", "transcript": "يعزز التعليم والتعلم الذاتي مهارات البحث المستقل.", "audio_file": "EDU_07.wav"},
    {"doc_id": "EDU_08", "category": "EDU", "category_ar": "التعليم", "transcript": "تركز المناهج الحديث على حل المشكلات.", "audio_file": "EDU_08.wav"},
    {"doc_id": "EDU_09", "category": "EDU", "category_ar": "التعليم", "transcript": "يرفع تدريب المعلمين جودة العملية التعليمية .", "audio_file": "EDU_09.wav"},
    {"doc_id": "EDU_10", "category": "EDU", "category_ar": "التعليم", "transcript": "يزيد التلعيب  دافعية الطلاب نحو التعلم.", "audio_file": "EDU_10.wav"},

    # الرياضة (SPORT)
    {"doc_id": "SPORT_01", "category": "SPORT", "category_ar": "الرياضة", "transcript": "تبني تمارين المقاومة الكتلة العضلية القوية.", "audio_file": "SPORT_01.wav"},
    {"doc_id": "SPORT_02", "category": "SPORT", "category_ar": "الرياضة", "transcript": "تعزز التمارين الرياضة الهوائية صحة الجهاز التنفسي.", "audio_file": "SPORT_02.wav"},
    {"doc_id": "SPORT_03", "category": "SPORT", "category_ar": "الرياضة", "transcript": "تغرس الرياضات الجماعية مهارات العمل المشترك.", "audio_file": "SPORT_03.wav"},
    {"doc_id": "SPORT_04", "category": "SPORT", "category_ar": "الرياضة", "transcript": "يمنع الإحماء الرياضي حدوث الإصابات العضلية.", "audio_file": "SPORT_04.wav"},
    {"doc_id": "SPORT_05", "category": "SPORT", "category_ar": "الرياضة", "transcript": "تسرع التغذية السليمة عملية الاستشفاء العضلي.", "audio_file": "SPORT_05.wav"},
    {"doc_id": "SPORT_06", "category": "SPORT", "category_ar": "الرياضة", "transcript": "تحسن تمارين الإطالة مرونة مفاصل الجسم.", "audio_file": "SPORT_06.wav"},
    {"doc_id": "SPORT_07", "category": "SPORT", "category_ar": "الرياضة", "transcript": "تتتبع الساعات الذكيةالسعرات والنشاط الرياضي.", "audio_file": "SPORT_07.wav"},
    {"doc_id": "SPORT_08", "category": "SPORT", "category_ar": "الرياضة", "transcript": "يسبب الجفاف انخفاض الأداء الرياضي العام.", "audio_file": "SPORT_08.wav"},
    {"doc_id": "SPORT_09", "category": "SPORT", "category_ar": "الرياضة", "transcript": "يساعد الإعداد الذهني في تحمل الضغوط .", "audio_file": "SPORT_09.wav"},
    {"doc_id": "SPORT_10", "category": "SPORT", "category_ar": "الرياضة", "transcript": "يعيد النوم بعد التمرين بناء العضلات.", "audio_file": "SPORT_10.wav"}
]

# تدريب TF-IDF
processed_corpus = [normalize_arabic(d["transcript"]) for d in CORPUS]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_corpus)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 10

class EvalQuery(BaseModel):
    query: str
    category: str

@app.get("/api/stats")
def get_stats():
    return {
        "total_documents": len(CORPUS),
        "vocabulary_size": len(vectorizer.vocabulary_)
    }

@app.get("/api/index-sample")
def get_index_sample():
    index_map = {}
    for doc in CORPUS:
        words = set(normalize_arabic(doc["transcript"]).split())
        for w in words:
            if w not in index_map:
                index_map[w] = []
            index_map[w].append(doc["doc_id"])
    
    sorted_terms = sorted(index_map.items(), key=lambda x: len(x[1]), reverse=True)[:18]
    return [{"term": k, "doc_ids": v} for k, v in sorted_terms]

@app.post("/api/search")
def search(payload: SearchQuery):
    cleaned_q = normalize_arabic(payload.query)
    if not cleaned_q:
        return {"results": []}
    
    q_vec = vectorizer.transform([cleaned_q])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    ranked_indices = np.argsort(scores)[::-1]
    
    results = []
    for idx in ranked_indices[:payload.top_k]:
        score = float(scores[idx])
        if score > 0:
            doc = CORPUS[idx].copy()
            doc["similarity"] = score
            results.append(doc)
            
    return {"results": results}

@app.post("/api/evaluate")
def evaluate(payload: EvalQuery):
    cleaned_q = normalize_arabic(payload.query)
    if not cleaned_q:
        return {"precision": 0, "recall": 0, "f1": 0}
    
    q_vec = vectorizer.transform([cleaned_q])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    ranked_indices = np.argsort(scores)[::-1]
    
    retrieved = [CORPUS[idx] for idx in ranked_indices[:10] if scores[idx] > 0]
    total_relevant = len([d for d in CORPUS if d["category"] == payload.category])
    relevant_retrieved = len([d for d in retrieved if d["category"] == payload.category])
    
    precision = relevant_retrieved / len(retrieved) if retrieved else 0.0
    recall = relevant_retrieved / total_relevant if total_relevant else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3)
    }

@app.post("/api/process-audio")
async def process_user_audio(file: UploadFile = File(...)):
    # 1. حفظ الملف الصوتي مؤقتاً للمعالجة
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 2. تحويل الصوت لنص عربي (Speech-to-Text)
        transcription_result = stt_model.transcribe(tmp_path, language="ar")
        query_text = transcription_result.get("text", "").strip()

        if not query_text:
            return {"error": "لم يتم التعرف على أي كلام داخل التسجيل"}

        # 3. معالجة النص وحساب TF-IDF والتشابه الجيبي
        processed_q = preprocess_arabic(query_text)
        query_vec = vectorizer.transform([processed_q])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # ترتيب النتائج المسترجعة
        top_indices = np.argsort(similarities)[::-1]
        results = []
        retrieved_cats = []
        
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim > 0.05:  # حد أدنى للصلة
                doc = CORPUS[idx]
                results.append({
                    "doc_id": doc["doc_id"],
                    "text": doc.get("transcript", doc.get("text", "")),
                    "category": doc["category"],
                    "similarity": round(sim, 4),
                    "audio_file": doc["audio_file"]
                })
                retrieved_cats.append(doc["category"])

        # 4. التنبؤ بالتصنيف الأكثر ملائمة (Category Classification)
        predicted_category = max(set(retrieved_cats), key=retrieved_cats.count) if retrieved_cats else "غير محدد"

        # 5. حساب مقاييس التقييم (Precision, Recall, F1)
        # نعتبر المستند ذا صلة (Relevant) إذا تطابق تصنيفه مع التصنيف المتنبأ به
        k = len(results)
        if k > 0 and predicted_category != "غير محدد":
            total_relevant_in_corpus = sum(1 for d in CORPUS if d["category"] == predicted_category)
            relevant_retrieved = sum(1 for r in results if r["category"] == predicted_category)
            
            precision = round(relevant_retrieved / k, 4)
            recall = round(relevant_retrieved / total_relevant_in_corpus, 4) if total_relevant_in_corpus > 0 else 0
            f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0

        return {
            "transcribed_text": query_text,
            "predicted_category": predicted_category,
            "results_count": len(results),
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "results": results
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)