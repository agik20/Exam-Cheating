import os

# Path ke direktori dataset
base_path = r"C:\Users\HeimTech\Downloads\Videos\object-detection\datasets"
label_subdirs = ['labels/train', 'labels/val', 'labels/test']

# Jumlah class yang valid
valid_classes = [0, 1]

def is_valid_line(line):
    try:
        parts = list(map(float, line.strip().split()))
        if len(parts) != 5:
            return False
        class_id = int(parts[0])
        if class_id not in valid_classes:
            return False
        if not all(0.0 <= val <= 1.0 for val in parts[1:]):
            return False
        return True
    except:
        return False

total_files = 0
invalid_files = []

for subdir in label_subdirs:
    folder = os.path.join(base_path, subdir)
    if not os.path.exists(folder):
        print(f"[SKIP] Folder tidak ditemukan: {folder}")
        continue

    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue

        total_files += 1
        full_path = os.path.join(folder, fname)
        with open(full_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not is_valid_line(line):
                    invalid_files.append(full_path)
                    break  # cukup satu error per file

# Hasil
print(f"\n🔍 Total file dicek: {total_files}")
if invalid_files:
    print(f"Ditemukan {len(invalid_files)} file label tidak valid:")
    for f in invalid_files:
        print(f"  - {f}")
else:
    print("Semua file label valid dan sesuai format YOLO.")
