import os
import cv2
import zipfile
from tqdm import tqdm

# Folder input berisi gambar dengan ukuran berbeda-beda
input_folder = r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\capstone\inference\frame\images\train"

# Folder output di Downloads
output_folder = os.path.join(os.path.expanduser('~'), 'Downloads', 'lab_output')
os.makedirs(output_folder, exist_ok=True)

# Target ukuran
target_size = (640, 640)

# Daftar ekstensi gambar yang didukung
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# Proses semua gambar
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
print(f"Total {len(image_files)} images found. Processing...")

for img_name in tqdm(image_files):
    input_path = os.path.join(input_folder, img_name)
    output_path = os.path.join(output_folder, img_name)

    # Baca dan resize gambar
    img = cv2.imread(input_path)
    if img is None:
        print(f"Warning: Gagal membaca {img_name}")
        continue

    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized_img)

print(f"Resize selesai. Gambar disimpan di: {output_folder}")
