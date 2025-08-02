import os
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np

# ======= KONFIGURASI MANUAL =======
input_folder = '/media/heimy/sd-card/capstone-project/output/frames'
output_folder = '/media/heimy/sd-card/capstone-project/fine-tune/images'
target_width = 640
target_height = 640
# ==================================

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Ekstensi yang didukung
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif')

# Ambil semua file gambar
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
print(f"Total {len(image_files)} images found. Processing...")

for img_name in tqdm(image_files):
    input_path = os.path.join(input_folder, img_name)

    try:
        # Buka gambar dengan PIL (dukung AVIF, WEBP)
        with Image.open(input_path) as img:
            img = img.convert('RGB')
            width, height = img.size

            # Hitung skala untuk resize lebar ke 640
            scale = target_width / width
            new_width = target_width
            new_height = int(height * scale)

            # Resize pakai OpenCV
            img_np = np.array(img)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            resized_img = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Hitung padding vertikal
            pad_vert = target_height - new_height
            pad_top = pad_vert // 2
            pad_bottom = pad_vert - pad_top

            # Tambahkan padding hitam di atas dan bawah
            padded_img = cv2.copyMakeBorder(
                resized_img,
                pad_top, pad_bottom, 0, 0,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)  # warna hitam
            )

            # Simpan hasil sebagai JPG
            base_name = os.path.splitext(img_name)[0]
            output_path = os.path.join(output_folder, base_name + '.jpg')
            cv2.imwrite(output_path, padded_img)

    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print(f"\nProses selesai. Gambar disimpan di: {output_folder}")
