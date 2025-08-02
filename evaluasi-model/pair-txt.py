import os
import shutil

# ==== KONFIGURASI ====
img_ref_folder = r"E:\Videos\dataset\pose\70\test"
label_folder = r"E:\Videos\dataset\pose\80\labels\val"
output_label_folder = r"E:\Videos\dataset\pose\70\labels\test"

os.makedirs(output_label_folder, exist_ok=True)

# ==== LOOP SETIAP GAMBAR ====
for img_file in os.listdir(img_ref_folder):
    if not img_file.lower().endswith((".jpg", ".png")):
        continue

    # Ambil nama file tanpa ekstensi
    base_name = os.path.splitext(img_file)[0]
    label_file = base_name + ".txt"
    label_path = os.path.join(label_folder, label_file)

    if os.path.exists(label_path):
        # Salin file label ke folder output
        shutil.copy(label_path, os.path.join(output_label_folder, label_file))
        print(f"[OK] {label_file} disalin.")
    else:
        print(f"[SKIP] Label tidak ditemukan untuk: {label_file}")

print(f"\n[SELESAI] Semua label yang cocok telah disalin ke: {output_label_folder}")
