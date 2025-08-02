import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO('/media/heimy/sd-card/capstone-project/model/pose-70m.pt')

# Label mapping
label_map = {
    'no-cheating': 0
    'provide-object': 1
    'see-friends-work': 2
}
# Path configuration
image_root = '/media/heimy/sd-card/capstone-project/fine-tune/images/train'
output_root = '/media/heimy/sd-card/capstone-project/fine-tune/labels/train'

IOU_THRESHOLD = 0.5
POSE_SIMILARITY_THRESHOLD = 0.08  # empiris, bisa diubah

def compute_iou(box1, box2):
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def is_similar_pose(kps1, kps2, threshold=POSE_SIMILARITY_THRESHOLD):
    valid_mask = (kps1[:, 2] > 0.05) & (kps2[:, 2] > 0.05)
    if np.sum(valid_mask) == 0:
        return False
    diff = kps1[valid_mask, :2] - kps2[valid_mask, :2]
    mean_dist = np.linalg.norm(diff, axis=1).mean()
    return mean_dist < threshold

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

        if result.keypoints is None or len(result.keypoints) == 0:
            print(f"[i] Tidak ada deteksi pada: {img_file}")
            continue

        boxes = result.boxes.xywhn.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()

        indices = np.argsort(-scores)
        selected_boxes, selected_indices = [], []

        for idx in indices:
            current_box = boxes[idx]
            if all(compute_iou(current_box, sel_box) <= IOU_THRESHOLD for sel_box in selected_boxes):
                selected_boxes.append(current_box)
                selected_indices.append(idx)

        final_indices = []
        for idx in selected_indices:
            duplicate = False
            for fidx in final_indices:
                if is_similar_pose(keypoints[idx], keypoints[fidx]):
                    duplicate = True
                    break
            if not duplicate:
                final_indices.append(idx)

        lines = []
        for idx in final_indices:
            box = boxes[idx]
            kps = keypoints[idx]
            kps[:, 0] /= img_width
            kps[:, 1] /= img_height

            line_parts = [f"{class_id}"]
            line_parts.extend([f"{x:.6f}" for x in box])

            for x, y, conf in kps:
                x = max(min(x, 1.0), 0.0)
                y = max(min(y, 1.0), 0.0)
                if x == 0.0 and y == 0.0:
                    v = 0
                else:
                    v = 0 if conf < 0.05 else (1 if conf < 0.5 else 2)
                line_parts.extend([f"{x:.6f}", f"{y:.6f}", f"{v}"])

            lines.append(" ".join(line_parts))

        label_file = os.path.join(output_folder, os.path.splitext(img_file)[0] + '.txt')
        with open(label_file, 'w') as f:
            f.write('\n'.join(lines))

        print(f"[✓] Label berhasil dibuat: {label_file}")

print("Proses anotasi selesai!")