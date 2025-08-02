import os

# PATH KONFIGURASI
folder_keypoints = r"C:\Users\HeimTech\Downloads\vsix\labels\val"
folder_bboxes = r"C:\Users\HeimTech\Downloads\vsix\labels\labels\val"
folder_output = r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\exam-act\dataset-pose-v5\datasets\v5\labels\val"

# Pastikan folder output ada
os.makedirs(folder_output, exist_ok=True)

# Ambil semua file di folder keypoints
for filename in os.listdir(folder_keypoints):
    if not filename.endswith(".txt"):
        continue

    path_kpt = os.path.join(folder_keypoints, filename)
    path_box = os.path.join(folder_bboxes, filename)
    path_out = os.path.join(folder_output, filename)

    # Lewatkan kalau tidak ada bounding box
    if not os.path.exists(path_box):
        print(f"[SKIP] Bounding box not found: {filename}")
        continue

    with open(path_kpt, 'r') as f_kpt, open(path_box, 'r') as f_box:
        lines_kpt = f_kpt.readlines()
        lines_box = f_box.readlines()

    new_lines = []

    for kpt_line, box_line in zip(lines_kpt, lines_box):
        kpt_parts = kpt_line.strip().split()
        box_parts = box_line.strip().split()

        if len(kpt_parts) < 6 or len(box_parts) < 5:
            print(f"[WARNING] Format salah di file: {filename}")
            continue

        # Ambil 4 koordinat bbox dari bounding-box label
        class_id = kpt_parts[0]
        new_bbox = box_parts[1:5]  # xc, yc, w, h

        # Gabungkan: class_id + bbox_baru + keypoints_lama
        new_line = ' '.join([class_id] + new_bbox + kpt_parts[5:])
        new_lines.append(new_line)

    # Tulis hasil ke file output
    with open(path_out, 'w') as f_out:
        f_out.write('\n'.join(new_lines))

    print(f"[OK] Replaced bbox in: {filename}")
