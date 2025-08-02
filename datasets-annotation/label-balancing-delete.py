import os
import random

label_folder = r'C:\Users\HeimTech\Downloads\6.11.2025\labels'  # Ganti dengan folder label kamu
target_class = 0
max_target_count = 2705

# --- Hitung dan kumpulkan semua objek class 0 ---
class0_entries = []  # Format: (file_path, line_index)

for filename in os.listdir(label_folder):
    if filename.endswith('.txt'):
        path = os.path.join(label_folder, filename)

        # Hapus file kosong
        if os.path.getsize(path) == 0:
            os.remove(path)
            print(f"Hapus file kosong: {filename}")
            continue

        # Cari baris dengan class 0
        with open(path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith(f'{target_class} '):
                class0_entries.append((path, i))

# --- Hitung total class 0 dan tentukan yang akan dihapus ---
current_class0_count = len(class0_entries)
to_delete = current_class0_count - max_target_count
print(f"Total objek class 0: {current_class0_count}")
print(f"Jumlah yang akan dihapus: {to_delete}")

if to_delete > 0:
    # Pilih acak entry class 0 yang akan dihapus
    random.shuffle(class0_entries)
    to_remove = class0_entries[:to_delete]

    # Mapping: path → list of line indices to remove
    file_line_map = {}
    for path, idx in to_remove:
        file_line_map.setdefault(path, set()).add(idx)

    # Hapus baris sesuai mapping
    for path, indices in file_line_map.items():
        with open(path, 'r') as f:
            lines = f.readlines()

        new_lines = [line for i, line in enumerate(lines) if i not in indices]

        if new_lines:
            with open(path, 'w') as f:
                f.writelines(new_lines)
        else:
            os.remove(path)
            print(f"Hapus file (semua baris class 0): {os.path.basename(path)}")

    print("✅ Pengurangan class 0 selesai.")
else:
    print("✅ Tidak ada class 0 yang perlu dihapus (sudah seimbang).")
