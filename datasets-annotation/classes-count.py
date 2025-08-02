import os
from collections import defaultdict

# Path ke folder label
base_dir = r'C:\Users\HeimTech\Downloads\6.11.2025\labels'
splits = ['train', 'val', 'test']

# Inisialisasi counter untuk setiap class
class_counts = defaultdict(int)

# Loop melalui setiap split
for split in splits:
    label_dir = os.path.join(base_dir, split)
    if not os.path.exists(label_dir):
        continue
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(label_dir, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

# Output hasil
print("Jumlah objek per class:")
print(f"Class 0 (no-cheating): {class_counts[0]}")
print(f"Class 1 (provide-object): {class_counts[1]}")
print(f"Class 2 (see-friends-work): {class_counts[2]}")
