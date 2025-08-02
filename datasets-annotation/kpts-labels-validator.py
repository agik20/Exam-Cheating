import os

# Path label input
LABEL_DIR = r"C:\Users\HeimTech\Downloads\6.11.2025\labels"

def is_invalid(val):
    try:
        v = float(val)
        return v < 0.0 or v > 1.0
    except ValueError:
        return True

def validate_labels(label_dir):
    invalid_files = []

    for root, _, files in os.walk(label_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                lines = f.read().splitlines()

            for i, line in enumerate(lines):
                parts = line.strip().split()

                if len(parts) < 6:
                    print(f"[SKIP] Format tidak lengkap di {file} baris {i+1}")
                    continue

                # Bounding box
                bbox_vals = parts[1:5]
                if any(is_invalid(v) for v in bbox_vals):
                    print(f"[INVALID BBOX] {file} baris {i+1} -> {bbox_vals}")
                    invalid_files.append(file)
                    break

                # Keypoints
                kpt_vals = parts[5:]
                if len(kpt_vals) % 3 != 0:
                    print(f"[INVALID KPT LEN] {file} baris {i+1} -> jumlah keypoint tidak kelipatan 3")
                    invalid_files.append(file)
                    break

                for j in range(0, len(kpt_vals), 3):
                    x = kpt_vals[j]
                    y = kpt_vals[j+1]
                    if is_invalid(x) or is_invalid(y):
                        print(f"[INVALID KPT] {file} baris {i+1} -> x={x}, y={y}")
                        invalid_files.append(file)
                        break
                else:
                    continue  # tidak break dari for
                break  # jika ada keypoint invalid, stop per file

    if not invalid_files:
        print("✅ Semua file valid (bbox & keypoint dalam rentang 0-1).")
    else:
        print(f"\nDitemukan {len(set(invalid_files))} file yang tidak valid.")

# Jalankan validasi
validate_labels(LABEL_DIR)
