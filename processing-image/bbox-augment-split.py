import os
import random
import shutil
from tqdm import tqdm
import albumentations as A
import cv2

# Paths
source_images = r"C:\Users\HeimTech\Downloads\Videos\object-detection\match\images"
source_labels = r"C:\Users\HeimTech\Downloads\Videos\object-detection\match\labels"

output_images = r"C:\Users\HeimTech\Downloads\Videos\object-detection\datasets\images"
output_labels = r"C:\Users\HeimTech\Downloads\Videos\object-detection\datasets\labels"

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_images, split), exist_ok=True)
    os.makedirs(os.path.join(output_labels, split), exist_ok=True)

# Gather all .jpg
all_files = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_files)

# Split data 70-15-15
n = len(all_files)
train_files = all_files[:int(0.7 * n)]
val_files = all_files[int(0.7 * n):int(0.85 * n)]
test_files = all_files[int(0.85 * n):]

splits = {"train": train_files, "val": val_files, "test": test_files}

# Copy original files to split folders
print("\nCopying original data to train/val/test folders...")
for split, files in splits.items():
    for fname in files:
        src_img = os.path.join(source_images, fname)
        src_lbl = os.path.join(source_labels, os.path.splitext(fname)[0] + ".txt")

        dst_img = os.path.join(output_images, split, fname)
        dst_lbl = os.path.join(output_labels, split, os.path.splitext(fname)[0] + ".txt")

        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

# Count class in train
print("\nCounting class distribution in train set...")
train_labels_path = os.path.join(output_labels, "train")
train_images_path = os.path.join(output_images, "train")
train_labels = os.listdir(train_labels_path)

class_counts = {0: 0, 1: 0}
img_contains_class0 = []

for lbl_file in train_labels:
    path = os.path.join(train_labels_path, lbl_file)
    with open(path, "r") as f:
        has_class0 = False
        for line in f:
            cls = int(line.split()[0])
            class_counts[cls] += 1
            if cls == 0:
                has_class0 = True
        if has_class0:
            img_contains_class0.append(lbl_file)

print(f"  Class 0 (smartphone): {class_counts[0]}")
print(f"  Class 1 (calculator): {class_counts[1]}")

# Augmentation config
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.MotionBlur(p=0.2),
], 
bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Augment class 0 data
needed_aug = class_counts[1] - class_counts[0]
done_aug = 0
print(f"\nAugmenting images with class 0 to balance dataset...")
for lbl_file in tqdm(img_contains_class0, total=len(img_contains_class0)):
    if done_aug >= needed_aug:
        break

    img_file = lbl_file.replace(".txt", ".jpg")
    img_path = os.path.join(train_images_path, img_file)
    lbl_path = os.path.join(train_labels_path, lbl_file)

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w = image.shape[:2]

    # Read original label
    bboxes = []
    class_labels = []
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls, x, y, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bboxes.append([x, y, bw, bh])
            class_labels.append(cls)

    # Apply augmentation
    augmented = augment(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_img = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_classes = augmented['class_labels']

    # Save augmented image & label
    aug_img_name = os.path.splitext(img_file)[0] + f"_aug{done_aug}.jpg"
    aug_lbl_name = os.path.splitext(lbl_file)[0] + f"_aug{done_aug}.txt"

    cv2.imwrite(os.path.join(train_images_path, aug_img_name), aug_img)

    with open(os.path.join(train_labels_path, aug_lbl_name), "w") as f:
        for cls, bbox in zip(aug_classes, aug_bboxes):
            f.write(f"{int(cls)} {' '.join(map(str, bbox))}\n")

    # Hitung hanya bbox class 0
    done_aug += sum(1 for c in aug_classes if c == 0)

print(f"\nAugmentasi selesai. Tambahan class 0: {done_aug}")
