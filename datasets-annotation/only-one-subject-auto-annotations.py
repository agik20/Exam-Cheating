import os
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 pose model
model = YOLO(r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\scripts\pre-processing-model\training-datasets\yolov8l-pose.pt')

# Label mapping
label_map = {
    'no-cheating': 0,
    'provide-object': 1,
    'see-friends-work': 2
}

# Path configuration
image_root = r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\exam-act\dataset-pose-v6\datasets\images-cropped'
output_root = r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\exam-act\dataset-pose-v6\datasets\labels'

# Main processing loop
for label_folder, class_id in label_map.items():
    input_folder = os.path.join(image_root, label_folder)
    output_folder = os.path.join(output_root, label_folder)
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"[!] Folder tidak ditemukan: {input_folder}")
        continue

    for img_file in os.listdir(input_folder):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[!] Tidak dapat membaca gambar: {img_path}")
            continue

        img_height, img_width = img.shape[:2]
        results = model(img_path)
        result = results[0]

        if result.keypoints is None or None or result.boxes is None:
            print(f"[i] Tidak ada deteksi pada: {img_file}")
            continue

        keypoints = result.keypoints.data.cpu().numpy()  # shape: (N, 17, 3)
        boxes = result.boxes.xywhn.cpu().numpy()

        if keypoints.shape[0] == 0 or boxes.shape[0] ==0:
            print(f"[i] Tidak ada orang yang terdeteksi secara lengkap pada: {img_file}")
            continue
        if keypoints.shape[0] != boxes.shape[0]:
            print(f"[!] Jumlah keypoints dan boxes tidak cocok pada: {img_file}")
            continue

        # Hitung rata-rata confidence untuk tiap person dan pilih yang tertinggi
        avg_conf = keypoints[:, :, 2].mean(axis=1)
        best_idx = np.argmax(avg_conf)
        best_box = boxes[best_idx]
        best_kps = keypoints[best_idx]  # shape: (17, 3)

        # Normalisasi keypoints
        best_kps[:, 0] /= img_width
        best_kps[:, 1] /= img_height

        # Bangun format label
        line_parts = [f"{class_id}"]
        line_parts.extend([f"{x:.6f}" for x in best_box])

        for x, y, conf in best_kps:
            x = max(min(x, 1.0), 0.0)
            y = max(min(y, 1.0), 0.0)

            # Tentukan v
            if x == 0.0 and y == 0.0:
                v = 0
            elif conf < 0.05:
                v = 1
            elif conf < 0.5:
                v = 1
            else:
                v = 2

            line_parts.extend([f"{x:.6f}", f"{y:.6f}", str(v)])

        # Simpan file label
        label_file = os.path.join(output_folder, os.path.splitext(img_file)[0] + '.txt')
        with open(label_file, 'w') as f:
            f.write(" ".join(line_parts))

        print(f"[✓] Label berhasil dibuat dan dibersihkan: {label_file}")

print("✅ Proses auto-annotation selesai!")
