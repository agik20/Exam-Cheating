import cv2
import numpy as np
from ultralytics import YOLO

# ========== CONFIGURATION ==========
model_path = 'yolov8m-pose.pt'  # Pastikan model tersedia
input_img_path = '/home/heimy/Music/normal-sample-skltn3.jpg'        # Ganti dengan path ke gambar kamu
output_img_path = '/home/heimy/Music/normal-sample-onlyskltn3.jpg'

# Load model YOLOv8 pose
model = YOLO(model_path)

# Load input image
img = cv2.imread(input_img_path)

# Inference
results = model(img)

# Ambil hasil keypoints
keypoints = results[0].keypoints
boxes = results[0].boxes
names = model.names  # Nama kelas

# Buat canvas putih
h, w = img.shape[:2]
white_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

# Warna skeleton default (bisa custom kalau mau)
skeleton_color = (0, 0, 255)
point_color = (255, 0, 0)

# Skeleton connections (COCO format 17 keypoints)
skeleton_pairs = [
    (5, 7), (7, 9),     # Left arm
    (6, 8), (8, 10),    # Right arm
    (5, 6),             # Shoulders
    (11, 12),           # Hips
    (5, 11), (6, 12),   # Torso
    (11, 13), (13, 15), # Left leg
    (12, 14), (14, 16), # Right leg
    (0, 1), (1, 2), (2, 3), (3, 4), # Head
    (0, 5), (0, 6)      # Neck to shoulders
]

for i, kpts in enumerate(keypoints.xy):
    # Gambar keypoint
    for kp in kpts:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(white_canvas, (x, y), 4, point_color, -1)

    # Gambar skeleton
    for pair in skeleton_pairs:
        partA, partB = pair
        if partA < len(kpts) and partB < len(kpts):
            x1, y1 = int(kpts[partA][0]), int(kpts[partA][1])
            x2, y2 = int(kpts[partB][0]), int(kpts[partB][1])
            cv2.line(white_canvas, (x1, y1), (x2, y2), skeleton_color, 2)

    # Tambahkan label kelas di atas kepala
    cls_id = int(boxes.cls[i].item()) if boxes and boxes.cls is not None else 0
    conf = float(boxes.conf[i].item()) if boxes and boxes.conf is not None else 0.0
    label = f"{names[cls_id]} {conf:.2f}"
    head_x, head_y = int(kpts[0][0]), int(kpts[0][1])
    cv2.putText(white_canvas, label, (head_x, head_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Simpan hasil gambar
cv2.imwrite(output_img_path, white_canvas)
print(f"[INFO] Gambar hasil skeleton disimpan di: {output_img_path}")