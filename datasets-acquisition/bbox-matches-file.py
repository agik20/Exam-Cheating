import os
import shutil
from tqdm import tqdm

# Input paths
images_input = r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\val\images"
labels_input = r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\val\labels"

# Output paths
output_dir = r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\val\match"
images_output = os.path.join(output_dir, "images")
labels_output = os.path.join(output_dir, "labels")

# Create output dirs if they don't exist
os.makedirs(images_output, exist_ok=True)
os.makedirs(labels_output, exist_ok=True)

# Get list of files without extensions
image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
label_files = {os.path.splitext(f)[0]: f for f in os.listdir(labels_input) if f.endswith('.txt')}

# Intersection
common_files = sorted(set(image_files.keys()) & set(label_files.keys()))
orphan_images = sorted(set(image_files.keys()) - set(label_files.keys()))
orphan_labels = sorted(set(label_files.keys()) - set(image_files.keys()))

# Class counters
class_counts = {0: 0, 1: 0}
total_boxes = 0

# Process matching pairs with progress bar
print("Processing matching image-label pairs...\n")
for name in tqdm(common_files, desc="Processing", unit="file"):
    img_src = os.path.join(images_input, image_files[name])
    lbl_src = os.path.join(labels_input, label_files[name])
    
    img_dst = os.path.join(images_output, image_files[name])
    lbl_dst = os.path.join(labels_output, label_files[name])

    # Move files
    shutil.move(img_src, img_dst)
    shutil.move(lbl_src, lbl_dst)

    # Count classes in label file
    with open(lbl_dst, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls in class_counts:
                    class_counts[cls] += 1
                    total_boxes += 1

# Report orphan files
print("\n=== Orphan Images (no matching label) ===")
for name in orphan_images:
    print(image_files[name])

print("\n=== Orphan Labels (no matching image) ===")
for name in orphan_labels:
    print(label_files[name])

# Final report
print("\n=== Summary ===")
print(f"Total matched pairs: {len(common_files)}")
print(f"Total smartphone (class 0): {class_counts[0]}")
print(f"Total calculator (class 1): {class_counts[1]}")
print(f"Orphan images: {len(orphan_images)}")
print(f"Orphan labels: {len(orphan_labels)}")
