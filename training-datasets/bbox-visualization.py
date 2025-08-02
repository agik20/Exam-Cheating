import cv2
import os
from tkinter import filedialog, Tk
from glob import glob

# Supaya GUI tkinter tidak muncul
Tk().withdraw()

# Pilih folder gambar
print("📂 Silakan pilih folder GAMBAR (JPG/PNG):")
images_path = filedialog.askdirectory(title="Pilih Folder GAMBAR")

# Pilih folder label YOLO
print("📂 Silakan pilih folder LABEL (TXT):")
labels_path = filedialog.askdirectory(title="Pilih Folder LABEL")

# Pilih folder output
print("💾 Silakan pilih folder OUTPUT:")
output_folder = filedialog.askdirectory(title="Pilih Folder OUTPUT")

# Konfigurasi
IMG_SIZE = 640

# Label kelas
CLASS_NAMES = {
    0: "smartphone",
    1: "calculator"
}

# Warna untuk bounding box tiap kelas (BGR)
CLASS_COLORS = {
    0: (255, 0, 0),   # Biru
    1: (0, 255, 0)    # Hijau
}

# Ambil semua gambar di folder input
image_files = sorted(glob(os.path.join(images_path, "*.jpg")) + glob(os.path.join(images_path, "*.png")))

print(f"🔍 Ditemukan {len(image_files)} gambar di {images_path}")

for img_path in image_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_path, base_name + ".txt")

    # Lewati jika file label tidak ditemukan
    if not os.path.exists(label_path):
        print(f"Label tidak ditemukan untuk: {base_name}")
        continue

    # Baca gambar
    image = cv2.imread(img_path)
    if image is None:
        print(f"Gagal membaca gambar: {img_path}")
        continue

    # Baca label
    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        class_id = int(class_id)

        # Ubah ke koordinat piksel
        x_center *= IMG_SIZE
        y_center *= IMG_SIZE
        width *= IMG_SIZE
        height *= IMG_SIZE

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        color = CLASS_COLORS.get(class_id, (0, 255, 255))
        label = CLASS_NAMES.get(class_id, str(class_id))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Simpan hasil ke folder output
    output_path = os.path.join(output_folder, base_name + "_vis.jpg")
    cv2.imwrite(output_path, image)
    print(f"Disimpan: {output_path}")

print("Semua gambar selesai divisualisasikan dan disimpan.")
