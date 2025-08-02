import cv2
import os
import numpy as np

# ==== KONFIGURASI ====
img_folder = r"E:\Videos\dataset\pose\70\test"
label_folder = r"E:\Videos\dataset\pose\70\labels\test"
output_folder = r"E:\Videos\dataset\pose\70\images\test"
scale = 1.5  # skala pembesaran bounding box

os.makedirs(output_folder, exist_ok=True)

# ==== LOOP SETIAP GAMBAR ====
for img_name in os.listdir(img_folder):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(img_folder, img_name)
    label_path = os.path.join(label_folder, os.path.splitext(img_name)[0] + ".txt")

    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Tidak bisa baca gambar: {image_path}")
        continue
    h_img, w_img = image.shape[:2]

    # Buat canvas hitam untuk masking
    masked = np.zeros_like(image)

    if os.path.exists(label_path):
        # ==== BACA DAN PROSES LABEL ====
        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            print(f"[WARNING] Label kosong: {label_path}")
        else:
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5:
                    continue  # lewati baris label tidak valid
                class_id = int(parts[0])
                x_center, y_center, w, h = parts[1:5]

                # Konversi ke pixel absolut
                x_c, y_c = x_center * w_img, y_center * h_img
                w_box, h_box = w * w_img, h * h_img

                # Skala bbox
                new_w = w_box * scale
                new_h = h_box * scale
                x1 = int(max(x_c - new_w / 2, 0))
                y1 = int(max(y_c - new_h / 2, 0))
                x2 = int(min(x_c + new_w / 2, w_img))
                y2 = int(min(y_c + new_h / 2, h_img))

                # Salin bagian dalam bbox ke canvas
                masked[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    else:
        print(f"[WARNING] Label tidak ditemukan untuk: {img_name}")
        # masked tetap hitam penuh (tidak diubah)

    # Simpan hasil masking dengan nama asli (tanpa prefix)
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, masked)

print(f"\n[INFO] Semua gambar selesai diproses dan disimpan ke: {output_folder}")
