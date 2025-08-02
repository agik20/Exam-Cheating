import os
import cv2
import glob
import random
import albumentations as A
import numpy as np

# Konfigurasi dasar
CLASSES = ["no-cheating", "provide-object", "see-friends-work"]
TARGET_COUNT = 700
KPT_NUM = 17
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

INPUT_BASE = r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\exam-act\dataset-pose-v5\datasets\convert"
OUTPUT_BASE = r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\exam-act\dataset-pose-v5\datasets\augmented"

# Transformasi augmentasi (tambahkan bboxes + class_labels)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
   bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def parse_label(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        bbox = list(map(float, parts[1:5]))
        keypoints = []
        for i in range(5, len(parts), 3):
            x = float(parts[i])
            y = float(parts[i+1])
            v = int(float(parts[i+2]))
            keypoints.append([x, y, v])
        labels.append({
            "cls": cls,
            "bbox": bbox,
            "keypoints": keypoints
        })
    return labels

def save_augmented(image, label_data, save_img_path, save_lbl_path):
    h, w = image.shape[:2]
    with open(save_lbl_path, "w") as f:
        for ann in label_data:
            bbox = ann["bbox"]
            kpts_str = ""
            for kp in ann["keypoints"]:
                x, y, v = kp
                kpts_str += f" {x:.6f} {y:.6f} {v}"
            line = f"{ann['cls']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}{kpts_str}\n"
            f.write(line)
    cv2.imwrite(save_img_path, image)

def augment_class(class_name):
    input_img_dir = os.path.join(INPUT_BASE, "images", class_name)
    input_lbl_dir = os.path.join(INPUT_BASE, "labels", class_name)
    output_img_dir = os.path.join(OUTPUT_BASE, "images", class_name)
    output_lbl_dir = os.path.join(OUTPUT_BASE, "labels", class_name)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_img_dir, "*.jpg")) + glob.glob(os.path.join(input_img_dir, "*.png")))
    existing_count = len(image_paths)
    augment_needed = max(0, TARGET_COUNT - existing_count)
    print(f"[{class_name}] Original: {existing_count}, Augment needed: {augment_needed}")

    if augment_needed == 0:
        return

    augment_index = 0
    while augment_index < augment_needed:
        img_path = random.choice(image_paths)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(input_lbl_dir, filename + ".txt")
        if not os.path.exists(lbl_path):
            continue

        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        label_data = parse_label(lbl_path)

        for _ in range(2):  # 2 augmentasi per iterasi
            keypoints_all = []
            visibility_all = []
            bboxes = []
            class_labels = []

            for ann in label_data:
                bboxes.append(ann["bbox"])  # format YOLO
                class_labels.append(ann["cls"])
                for kp in ann["keypoints"]:
                    if kp[2] > 0:
                        keypoints_all.append((kp[0] * w, kp[1] * h))
                    else:
                        keypoints_all.append((0.0, 0.0))
                    visibility_all.append(kp[2])

            try:
                transformed = transform(
                    image=image,
                    keypoints=keypoints_all,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
            except Exception as e:
                print(f"Skipping transform due to error: {e}")
                continue

            aug_img = transformed["image"]
            aug_kpts = transformed["keypoints"]
            aug_bboxes = transformed["bboxes"]
            aug_cls = transformed["class_labels"]

            idx = 0
            new_anns = []

            for i in range(len(aug_cls)):
                new_kpts = []
                for _ in range(KPT_NUM):
                    x_abs, y_abs = aug_kpts[idx]
                    vis = visibility_all[idx]
                    x_rel, y_rel = x_abs / w, y_abs / h
                    if not (0 <= x_rel <= 1) or not (0 <= y_rel <= 1) or vis == 0:
                        new_kpts.append([0.0, 0.0, 0])
                    else:
                        new_kpts.append([x_rel, y_rel, vis])
                    idx += 1

                # Ambil bounding box hasil augmentasi dari transformasi bbox asli
                x_c, y_c, bw, bh = aug_bboxes[i]
                x1 = (x_c - bw / 2) * w
                y1 = (y_c - bh / 2) * h
                x2 = (x_c + bw / 2) * w
                y2 = (y_c + bh / 2) * h

                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                new_xc = ((x1 + x2) / 2) / w
                new_yc = ((y1 + y2) / 2) / h
                new_bw = (x2 - x1) / w
                new_bh = (y2 - y1) / h

                if new_bw < 0.01 or new_bh < 0.01:
                    continue

                new_anns.append({
                    "cls": aug_cls[i],
                    "bbox": [new_xc, new_yc, new_bw, new_bh],
                    "keypoints": new_kpts
                })

            if not new_anns:
                continue

            aug_name = f"{filename}_aug{augment_index}.jpg"
            save_augmented(
                aug_img,
                new_anns,
                os.path.join(output_img_dir, aug_name),
                os.path.join(output_lbl_dir, aug_name.replace(".jpg", ".txt"))
            )
            augment_index += 1
            if augment_index >= augment_needed:
                break

# Jalankan untuk semua class
for cls in CLASSES:
    augment_class(cls)

print("Semua augmentasi selesai dan seimbang.")
