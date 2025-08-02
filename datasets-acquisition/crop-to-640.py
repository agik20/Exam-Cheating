import os
from PIL import Image
import cv2
from tqdm import tqdm

# ======= KONFIGURASI MANUAL =======
input_folder = r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\capstone\inference\frame\images\train'
output_folder = r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\capstone\inference\frame\images\train\cropped'
target_size = (640, 640)
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
        # Buka gambar dengan PIL (karena mendukung lebih banyak format seperti AVIF, WEBP)
        with Image.open(input_path) as img:
            img = img.convert('RGB')  # pastikan mode RGB
            width, height = img.size

            # Crop ke rasio 1:1 (square) dari tengah
            min_side = min(width, height)
            left = (width - min_side) // 2
            top = (height - min_side) // 2
            right = left + min_side
            bottom = top + min_side
            img_cropped = img.crop((left, top, right, bottom))

            # Simpan sementara sebagai JPG agar bisa dibaca oleh OpenCV
            temp_path = os.path.join(output_folder, 'temp.jpg')
            img_cropped.save(temp_path, 'JPEG')

            # Resize dengan OpenCV
            img_cv = cv2.imread(temp_path)
            resized_img = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_AREA)

            # Simpan sebagai JPG dengan nama asli (ganti ekstensi ke .jpg)
            base_name = os.path.splitext(img_name)[0]
            output_path = os.path.join(output_folder, base_name + '.jpg')
            cv2.imwrite(output_path, resized_img)

            # Hapus file temp
            os.remove(temp_path)

    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print(f"\nProses selesai. Gambar disimpan di: {output_folder}")
