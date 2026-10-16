import os
import whisper
from difflib import SequenceMatcher

model = whisper.load_model("tiny")  # نموذج سريع وخفيف
audio_dir = r"C:\my thing\MyCourses\IR_Project\AAIRS\frontend\audio"

# النصوص المستهدفة من main.py
targets = {
    "AI_01": "يحاكي الذكاء الاصطناعي القدرات الذهنية للبشر",
    "AI_02": "يعتمد التعلم العميق على الشبكات العصبية",
    "HEALTHY_01": "يقوي الغذاء الصحي المتوازن مناعة جسم الإنسان",
    "SPORT_01": "تبني تمارين المقاومة الكتلة العضلية القوية",
    "HEALTHY_02": "تساهم التمارين الهوائية في تحسين صحة القلب",
    "SPORT_02": "يساعد التدريب المتنوع على بناء عضلات متناسقة",
    "FITNESS_01": "تزيد تمارين الكارديو من قدرة التحمل البدني",
    "FITNESS_02": "يعتبر الإحماء ضروريًا لتجنب إصابات العضلات",
    "FITNESS_03": "يساعد تمرين الضغط على تقوية عضلات الذراعين",
    "FITNESS_04": "تعزز تمارين التمدد مرونة الجسم أثناء الحركة",
    "FITNESS_05": "يساهم تمرين السكوات في بناء عضلات الأرجل"
}

for filename in os.listdir(audio_dir):
    if not filename.endswith(('.mp3', '.m4a', '.wav')):
        continue
    filepath = os.path.join(audio_dir, filename)
    
    # تحويل الصوت لنص
    result = model.transcribe(filepath, language="ar")
    spoken_text = result["text"]
    
    # إيجاد أقرب جملة مطابقة
    best_match = None
    highest_ratio = 0
    for doc_id, text in targets.items():
        ratio = SequenceMatcher(None, spoken_text, text).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = doc_id
            
    if best_match and highest_ratio > 0.4:
        ext = os.path.splitext(filename)[1]
        new_name = f"{best_match}{ext}"
        os.rename(filepath, os.path.join(audio_dir, new_name))
        print(f"تمت مطابقة {filename} مع {new_name}")