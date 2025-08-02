import os
import shutil

# List of source folders (ganti dengan path folder sumber Anda)
source_folders = [
    r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\exam-act\object-detection\smartphone\datasets\labels\train',
    r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\exam-act\object-detection\smartphone\datasets\labels\val',
    r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\exam-act\object-detection\smartphone\datasets\labels\test',
]

# Target folder (folder tujuan gabungan)
target_folder = r'C:\Users\HeimTech\Downloads\Videos\object-detection\match\labels'
os.makedirs(target_folder, exist_ok=True)

# Ekstensi file gambar yang didukung
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.txt')

file_count = 0
skipped_count = 0

for folder in source_folders:
    if not os.path.exists(folder):
        print(f"[!] Folder tidak ditemukan: {folder}")
        continue

    for filename in os.listdir(folder):
        if not filename.lower().endswith(image_extensions):
            continue

        src_path = os.path.join(folder, filename)
        dst_path = os.path.join(target_folder, filename)

        if os.path.exists(dst_path):
            print(f"[⏩] Dilewati (sudah ada): {filename}")
            skipped_count += 1
            continue

        shutil.copy2(src_path, dst_path)
        file_count += 1
        print(f"[✓] Dipindahkan: {filename}")

print(f"✅ Total {file_count} file berhasil digabungkan ke: {target_folder}")
print(f"ℹ️ {skipped_count} file dilewati karena sudah ada.")
