import os
import cv2
import glob
import random
import albumentations as A
import numpy as np

# === CONFIG ===
TARGET_TOTAL = 2000  # total jumlah image augmented yang diinginkan
KPT_USED = 13
KPT_TOTAL = 17
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]  # hanya untuk 13 keypoint
INPUT_IMG_DIR = r"C:\Users\HeimTech\Downloads\New folder\images"
INPUT_LBL_DIR = r"C:\Users\HeimTech\Downloads\New folder\labels"
OUTPUT_IMG_DIR = r"C:\Users\HeimTech\Downloads\augmented\images"
OUTPUT_LBL_DIR = r"C:\Users\HeimTech\Downloads\augmented\labels"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

# === AUGMENTATION TRANSFORM ===
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# === HELPER FUNCTIONS ===
def parse_label(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        bbox = list(map(float, parts[1:5]))
        keypoints = []
        for i in range(5, min(5 + KPT_USED * 3, len(parts)), 3):
            x = float(parts[i])
            y = float(parts[i+1])
            v = int(float(parts[i+2]))
            keypoints.append([x, y, v])
        labels.append({"cls": cls, "bbox": bbox, "keypoints": keypoints})
    return labels

def calculate_bbox_from_keypoints(keypoints, img_w, img_h):
    visible = [(kp[0] * img_w, kp[1] * img_h) for kp in keypoints if kp[2] > 0]
    if not visible:
        return None
    x_coords, y_coords = zip(*visible)
    x_min = max(0, min(x_coords))
    y_min = max(0, min(y_coords))
    x_max = min(img_w, max(x_coords))
    y_max = min(img_h, max(y_coords))
    x_c = ((x_min + x_max) / 2) / img_w
    y_c = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return [x_c, y_c, w, h] if w > 0.01 and h > 0.01 else None

def save_augmented(image, label_data, img_name):
    h, w = image.shape[:2]
    label_path = os.path.join(OUTPUT_LBL_DIR, img_name.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        for ann in label_data:
            bbox = ann["bbox"]
            kpts = ann["keypoints"]
            kpt_str = ''.join([f' {kp[0]:.6f} {kp[1]:.6f} {kp[2]}' for kp in kpts])
            f.write(f"{ann['cls']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}{kpt_str}\n")
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_name), image)

# === MAIN AUGMENTATION LOOP ===
image_paths = sorted(glob.glob(os.path.join(INPUT_IMG_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_IMG_DIR, "*.png")))
existing_aug = len(os.listdir(OUTPUT_IMG_DIR))
augment_index = 0

print(f"[INFO] Starting augmentations. Existing: {existing_aug}, Target: {TARGET_TOTAL}")

while existing_aug + augment_index < TARGET_TOTAL:
    img_path = random.choice(image_paths)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(INPUT_LBL_DIR, filename + ".txt")

    if not os.path.exists(lbl_path):
        continue

    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    anns = parse_label(lbl_path)

    # Flatten all keypoints
    keypoints_all = []
    visibility_all = []
    for ann in anns:
        for kp in ann["keypoints"]:
            x = kp[0] * w if kp[2] > 0 else 0
            y = kp[1] * h if kp[2] > 0 else 0
            keypoints_all.append((x, y))
            visibility_all.append(kp[2])

    try:
        transformed = transform(image=image, keypoints=keypoints_all)
    except Exception as e:
        print(f"[ERROR] Skipped due to transform error: {e}")
        continue

    aug_img = transformed["image"]
    aug_kpts = transformed["keypoints"]
    idx = 0
    new_anns = []

    for ann in anns:
        new_kpts = []
        for _ in range(KPT_USED):
            x_abs, y_abs = aug_kpts[idx]
            vis = visibility_all[idx]
            x_rel, y_rel = x_abs / w, y_abs / h
            if not (0 <= x_rel <= 1) or not (0 <= y_rel <= 1) or vis == 0:
                new_kpts.append([0.0, 0.0, 0])
            else:
                new_kpts.append([x_rel, y_rel, vis])
            idx += 1

        if transform.transforms[0].p > 0.5:
            new_kpts = [new_kpts[i] for i in FLIP_IDX]

        new_bbox = calculate_bbox_from_keypoints(new_kpts, w, h)
        if not new_bbox:
            continue

        new_anns.append({"cls": ann["cls"], "bbox": new_bbox, "keypoints": new_kpts})

    if not new_anns:
        continue

    aug_name = f"{filename}_aug{augment_index}.jpg"
    save_augmented(aug_img, new_anns, aug_name)
    augment_index += 1

print(f"Augmentasi selesai. Total ditambahkan: {augment_index}")
