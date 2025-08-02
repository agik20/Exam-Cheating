import os

label_folder = r"E:\Videos\dataset\pose\70\labels\test"
backup_folder = r"E:\Videos\dataset\pose\70\labels\test_backup"

os.makedirs(backup_folder, exist_ok=True)

for label_file in os.listdir(label_folder):
    if not label_file.endswith(".txt"):
        continue

    path = os.path.join(label_folder, label_file)
    backup_path = os.path.join(backup_folder, label_file)

    with open(path, "r") as f:
        lines = f.readlines()

    # Buat backup dulu
    with open(backup_path, "w") as f:
        f.writelines(lines)

    new_lines = []
    for line in lines:
        parts = line.strip().split()

        # Lewati jika baris tidak sesuai format label pose (minimal 56 nilai)
        if len(parts) != 56:
            print(f"[SKIP] Format tidak valid: {label_file}")
            new_lines.append(line.strip())
            continue

        header = parts[:5]
        kpts = parts[5:]
        fixed_kpts = []

        for i in range(0, len(kpts), 3):
            x = kpts[i]
            y = kpts[i + 1]
            try:
                v = str(int(float(kpts[i + 2])))
            except:
                v = kpts[i + 2]  # biarkan nilai aslinya
            fixed_kpts.extend([x, y, v])

        new_lines.append(" ".join(header + fixed_kpts))

    with open(path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

print("[DONE] Semua file telah diperbaiki. Backup tersedia di:", backup_folder)
