import os
from PIL import Image
import cv2
from tqdm import tqdm

# ======= KONFIGURASI MANUAL =======
input_folder = r'C:\Users\HeimTech\Downloads\Videos\New folder\images'
output_folder = r'C:\Users\HeimTech\Downloads\Videos\New folder\images\cropped'
target_size = (640, 640)
# ==================================

os.makedirs(output_folder, exist_ok=True)

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif')

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
print(f"Total {len(image_files)} images found. Processing...")

for img_name in tqdm(image_files):
    input_path = os.path.join(input_folder, img_name)

    try:
        with Image.open(input_path) as img:
            img = img.convert('RGB')
            width, height = img.size

            # Crop square dari sisi kanan (bukan tengah)
            if width > height:
                # Potong persegi dari kanan
                left = width - height
                top = 0
                right = width
                bottom = height
            else:
                # Kalau tingginya lebih dari lebar, tetap crop dari tengah atas (default)
                left = 0
                top = (height - width) // 2
                right = width
                bottom = top + width

            img_cropped = img.crop((left, top, right, bottom))

            # Simpan sementara
            temp_path = os.path.join(output_folder, 'temp.jpg')
            img_cropped.save(temp_path, 'JPEG')

            # Resize dengan OpenCV
            img_cv = cv2.imread(temp_path)
            resized_img = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_AREA)

            # Simpan hasil akhir
            base_name = os.path.splitext(img_name)[0]
            output_path = os.path.join(output_folder, base_name + '.jpg')
            cv2.imwrite(output_path, resized_img)

            os.remove(temp_path)

    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print(f"\nProses selesai. Gambar disimpan di: {output_folder}")
