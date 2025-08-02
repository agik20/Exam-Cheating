import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

# === SESUAIKAN DENGAN DIREKTORI KAMU ===
images_dir = Path("fine-tune/images/train")
labels_dir = Path("fine-tune/object/labels/train")
output_base = Path("fine-tune\Datasets")

split_ratio = [0.7, 0.15, 0.15]  # train, val, test

# Ambil pasangan file yang valid (hanya jika .jpg dan .txt keduanya ada)
valid_pairs = []
for img_file in images_dir.glob("*.jpg"):
    label_file = labels_dir / (img_file.stem + ".txt")
    if label_file.exists():
        valid_pairs.append((img_file, label_file))

print(f"Ditemukan {len(valid_pairs)} pasangan gambar-label yang valid.")

# Buat label distribusi berdasarkan isi label file (untuk stratifikasi)
def extract_class_set(label_path):
    classes = set()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(parts[0])
    return tuple(sorted(classes))  # gunakan sebagai "multi-label key"

# Kelompokkan berdasarkan kombinasi label
class_buckets = defaultdict(list)
for img_path, lbl_path in valid_pairs:
    class_key = extract_class_set(lbl_path)
    class_buckets[class_key].append((img_path, lbl_path))

# Lakukan stratified sampling berdasarkan class_key
train_set, valtest_set = [], []
for group in class_buckets.values():
    n = len(group)
    n_train = int(n * split_ratio[0])
    random.shuffle(group)
    train_set.extend(group[:n_train])
    valtest_set.extend(group[n_train:])

val_set, test_set = train_test_split(
    valtest_set,
    test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
    random_state=42
)

splits = {
    "train": train_set,
    "val": val_set,
    "test": test_set
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

for split in ["train", "val", "test"]:
    lbl_path = output_base / "labels" / split
    stats = count_classes(lbl_path)
    print(f"\nDistribusi kelas di {split}:")
    for cls in ['0', '1', '2']:
        print(f"Class {cls}: {stats.get(cls, 0)} objek")
