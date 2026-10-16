import os
import shutil

# المسارات الأساسية
base_dir = r"C:\my thing\MyCourses\IR_Project\AAIRS\audios"
target_dir = r"C:\my thing\MyCourses\IR_Project\AAIRS\frontend\audio"

# إنشاء مجلد audio في frontend إذا لم يكن موجوداً
os.makedirs(target_dir, exist_ok=True)

# خريطة المجلدات والتسميات المقابلة لها
folders_mapping = {
    "ai.audio": "AI",
    "ir.audio": "IR",
    "health.audio": "HEALTHY",
    "edu.audio": "EDU",
    "sport.audio": "SPORT"
}

total_copied = 0

for folder_name, prefix in folders_mapping.items():
    current_folder = os.path.join(base_dir, folder_name)
    if not os.path.exists(current_folder):
        print(f"المجلد غير موجود: {folder_name}")
        continue
    
    # جلب ملفات الصوت فقط وترتيبها
    valid_exts = ('.mp3', '.m4a', '.wav', '.aac', '.ogg')
    files = [f for f in os.listdir(current_folder) if f.lower().endswith(valid_exts)]
    files.sort()  # ترتيب الملفات بالاسم/الرقم
    
    for i, old_file in enumerate(files, start=1):
        ext = os.path.splitext(old_file)[1].lower()
        new_file_name = f"{prefix}_{i:02d}{ext}"
        
        src_path = os.path.join(current_folder, old_file)
        dst_path = os.path.join(target_dir, new_file_name)
        
        shutil.copy2(src_path, dst_path)
        print(f"تم نسخ: {folder_name}/{old_file} -> frontend/audio/{new_file_name}")
        total_copied += 1

print(f"\n اكتمل النقل والتسمية بنجاح! تم نقل {total_copied} ملفاً إلى frontend/audio.")