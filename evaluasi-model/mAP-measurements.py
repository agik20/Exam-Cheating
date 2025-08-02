import os
import numpy as np
from collections import defaultdict

# === KONFIGURASI PATH ===
gt_path = r"E:\Videos\dataset\pose\70\labels\val"
pred_path = r"E:\Videos\dataset\pose\70\filtered-pred"
img_size = 640  # Ukuran gambar jika tetap

# === 1. KONVERSI class_id FLOAT KE INTEGER (DAN SIMPAN ULANG) ===
def convert_class_id_to_int(folder_path):
    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue
        file_path = os.path.join(folder_path, fname)
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip error lines
            parts[0] = str(int(float(parts[0])))  # ubah class_id jadi int
            new_lines.append(" ".join(parts))

        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))

print("[INFO] Mengonversi class_id float ke integer di GT folder...")
convert_class_id_to_int(gt_path)

# === 2. KONVERSI YOLO KE ABSOLUTE BBOX ===
def yolo_to_bbox(yolo_box, img_size=640):
    x_c, y_c, w, h = yolo_box
    x = (x_c - w / 2) * img_size
    y = (y_c - h / 2) * img_size
    return [x, y, w * img_size, h * img_size]

# === 3. MEMBACA FILE DARI FOLDER LABEL ===
def load_yolo_folder(folder_path, with_conf=False):
    boxes = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue
        image_id = os.path.splitext(fname)[0]
        with open(os.path.join(folder_path, fname), "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                x_center = parts[1]
                y_center = parts[2]
                width = parts[3]
                height = parts[4]
                conf = parts[5] if with_conf and len(parts) > 5 else 1.0

                bbox = yolo_to_bbox([x_center, y_center, width, height], img_size)
                boxes.append({
                    "image_id": image_id,
                    "class_id": class_id,
                    "bbox": bbox,
                    "score": conf
                })
    return boxes

# === 4. FUNGSI IoU DAN mAP50 ===
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union if union > 0 else 0

def compute_ap(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    indices = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

def compute_map50(pred_boxes, gt_boxes, iou_thresh=0.5):
    classes = set([d["class_id"] for d in gt_boxes])
    ap_per_class = []

    for cls in classes:
        cls_pred = [p for p in pred_boxes if p["class_id"] == cls]
        cls_gt = [g for g in gt_boxes if g["class_id"] == cls]

        gt_by_img = defaultdict(list)
        for g in cls_gt:
            gt_by_img[g["image_id"]].append(g)

        cls_pred = sorted(cls_pred, key=lambda x: x["score"], reverse=True)
        tp = np.zeros(len(cls_pred))
        fp = np.zeros(len(cls_pred))
        matched = defaultdict(set)

        for i, pred in enumerate(cls_pred):
            gts = gt_by_img[pred["image_id"]]
            ious = [compute_iou(pred["bbox"], gt["bbox"]) for gt in gts]
            max_iou = max(ious) if ious else 0
            max_idx = np.argmax(ious) if ious else -1

            if max_iou >= iou_thresh and max_idx not in matched[pred["image_id"]]:
                tp[i] = 1
                matched[pred["image_id"]].add(max_idx)
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / (len(cls_gt) + 1e-6)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        ap = compute_ap(recall, precision)
        ap_per_class.append(ap)

    return np.mean(ap_per_class)

# === 5. EKSEKUSI PERHITUNGAN mAP50 ===
print("[INFO] Membaca label...")
gt_boxes = load_yolo_folder(gt_path, with_conf=False)
pred_boxes = load_yolo_folder(pred_path, with_conf=True)

print("[INFO] Menghitung mAP50...")
map50 = compute_map50(pred_boxes, gt_boxes)
print(f"\n>> Hasil mAP50: {map50:.4f}")
