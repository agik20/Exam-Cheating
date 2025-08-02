from ultralytics import YOLO
import os
import torch
import torchvision.ops as ops

# Load model
model = YOLO(r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\capstone\model\object_detection\before_best_object_model.pt")

# Folder konfigurasi
img_folder = r"C:\Users\HeimTech\Downloads\07012025\images"
output_label_folder = r"C:\Users\HeimTech\Downloads\07012025\labels"
os.makedirs(output_label_folder, exist_ok=True)

# Ukuran gambar untuk normalisasi koordinat
img_width, img_height = 640, 640

# Jalankan prediksi streaming agar hemat RAM
results = model.predict(source=img_folder, save=False, conf=0.25, stream=True)

# Loop per gambar
for result in results:
    img_name = os.path.splitext(os.path.basename(result.path))[0]
    output_label_path = os.path.join(output_label_folder, f"{img_name}.txt")

    boxes = result.boxes

    # Cek apakah ada deteksi
    if boxes is None or boxes.cls is None or len(boxes.cls) == 0:
        # Tetap simpan file kosong
        with open(output_label_path, "w") as f:
            pass
        print(f"[INFO] {img_name}: Tidak ada deteksi. File kosong disimpan.")
        continue

    cls_list = boxes.cls.tolist()
    conf_list = boxes.conf.tolist()
    xyxy_list = boxes.xyxy.tolist()

    # Tidak filter kelas: simpan semua deteksi
    filtered = [(i, xyxy, conf_list[i]) for i, (xyxy, cls) in enumerate(zip(xyxy_list, cls_list))]

    # Jika masih tidak ada deteksi valid
    if not filtered:
        with open(output_label_path, "w") as f:
            pass
        print(f"[INFO] {img_name}: Tidak ada deteksi valid. File kosong disimpan.")
        continue

    # NMS manual
    indices, bboxes, confidences = zip(*[(i, torch.tensor(b), torch.tensor(c)) for i, b, c in filtered])
    bboxes_tensor = torch.stack(bboxes)
    scores_tensor = torch.stack(confidences)
    keep = ops.nms(bboxes_tensor, scores_tensor, iou_threshold=0.4)

    # Tulis hasil ke file
    with open(output_label_path, "w") as f:
        for idx in keep:
            i = indices[idx]
            xywh_abs = boxes.xywh[i].tolist()
            x_center = xywh_abs[0] / img_width
            y_center = xywh_abs[1] / img_height
            width = xywh_abs[2] / img_width
            height = xywh_abs[3] / img_height
            cls = int(boxes.cls[i].item())
            line = f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            f.write(line)

    print(f"[OK] {img_name}: {len(keep)} deteksi disimpan ke {output_label_path}")

print(f"\nSemua prediksi bounding box disimpan di: {output_label_folder}")
