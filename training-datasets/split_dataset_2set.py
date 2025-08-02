import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

# === SESUAIKAN DENGAN DIREKTORI KAMU ===
images_dir = Path("/media/heimy/sd-card/capstone-project/fine-tune/fine-tuning/images")
labels_dir = Path("/media/heimy/sd-card/capstone-project/fine-tune/fine-tuning/labels")
output_base = Path("/media/heimy/sd-card/capstone-project/fine-tune/fine-tuning/datasets")

split_ratio = [0.8, 0.2]  # train 80%, val 20%

def is_valid_label(label_path):
    """Filter label file hanya yang memiliki dua kelas (0 atau 1) dan format bbox saja"""
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:  # harus format: class x_center y_center width height
                return False
            if parts[0] not in ['0', '1']:  # hanya kelas 0 dan 1
                return False
    return True

# Ambil pasangan file yang valid (hanya jika .jpg dan .txt keduanya ada DAN format label benar)
valid_pairs = []
for img_file in images_dir.glob("*.jpg"):
    label_file = labels_dir / (img_file.stem + ".txt")
    if label_file.exists() and is_valid_label(label_file):
        valid_pairs.append((img_file, label_file))

print(f"Ditemukan {len(valid_pairs)} pasangan gambar-label yang valid (2 kelas dengan bbox saja).")

# Buat label distribusi berdasarkan kelas utama (untuk stratifikasi)
def extract_main_class(label_path):
    classes = set()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(parts[0])
    return max(classes) if classes else '0'  # ambil kelas tertinggi sebagai representasi

# Kelompokkan berdasarkan kelas utama
class_buckets = defaultdict(list)
for img_path, lbl_path in valid_pairs:
    class_key = extract_main_class(lbl_path)
    class_buckets[class_key].append((img_path, lbl_path))

# Lakukan stratified sampling berdasarkan kelas utama
train_set, val_set = [], []
for group in class_buckets.values():
    n = len(group)
    n_train = int(n * split_ratio[0])
    random.shuffle(group)
    train_set.extend(group[:n_train])
    val_set.extend(group[n_train:])

splits = {
    "train": train_set,
    "val": val_set
}

# Salin ke output
for split, files in splits.items():
    img_out = output_base / "images" / split
    lbl_out = output_base / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print(f"\nMenyalin {len(files)} data ke folder {split}...")
    for img_path, lbl_path in files:
        shutil.copy(img_path, img_out / img_path.name)
        shutil.copy(lbl_path, lbl_out / lbl_path.name)

# Statistik kelas
from collections import Counter

def count_classes(label_folder):
    counter = Counter()
    for lbl_file in Path(label_folder).glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                cls_id = line.strip().split()[0]
                counter[cls_id] += 1
    return counter

for split in ["train", "val"]:
    lbl_path = output_base / "labels" / split
    stats = count_classes(lbl_path)
    print(f"\nDistribusi kelas di {split}:")
    for cls in ['0', '1']:  # Hanya tampilkan kelas 0 dan 1
        print(f"Class {cls}: {stats.get(cls, 0)} objek")