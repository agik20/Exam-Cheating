from ultralytics import YOLO
import cv2
import os

# ====== KONFIGURASI ======
input_folder = r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\lab-act\Dataset\crowd'
output_label_folder = r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\datasets\lab-act\Dataset\crowd\labels'
model_path = r'C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\scripts\pre-processing-model\training-datasets\yolov8l-pose.pt'  # atau yolov8m.pt / yolov8l.pt
confidence_threshold = 0.4  # bisa disesuaikan
target_class_id = 0  # 0 = person di COCO dataset
# ==========================

os.makedirs(output_label_folder, exist_ok=True)

# Load model
model = YOLO(model_path)

# Loop semua gambar
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Jalankan deteksi
        results = model(image)[0]

        # Ambil ukuran gambar
        height, width = image.shape[:2]

        # Filter hasil hanya class 'person'
        lines = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == target_class_id and conf > confidence_threshold:
                # YOLO format (normalized x_center, y_center, width, height)
                x1, y1, x2, y2 = box.xyxy[0]
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        # Simpan file anotasi
        label_path = os.path.join(output_label_folder, os.path.splitext(filename)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

print(f"Selesai! Label disimpan di: {output_label_folder}")
