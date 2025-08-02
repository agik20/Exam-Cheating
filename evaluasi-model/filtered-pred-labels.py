import os
from pathlib import Path

def parse_yolo_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Baris tidak valid, terlalu pendek: {line.strip()}")

    try:
        class_id = int(float(parts[0]))  # Tangani class_id berbentuk float
    except ValueError:
        raise ValueError(f"Class ID tidak valid: {parts[0]}")

    bbox = list(map(float, parts[1:5]))  # xc, yc, w, h
    return class_id, bbox, line.strip()

def yolo_to_xyxy(xc, yc, w, h):
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return x1, y1, x2, y2

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def filter_predictions_by_gt(gt_dir, pred_dir, output_dir, iou_threshold=0.3):
    os.makedirs(output_dir, exist_ok=True)
    gt_files = sorted(Path(gt_dir).glob("*.txt"))

    for gt_file in gt_files:
        pred_file = Path(pred_dir) / gt_file.name
        out_file = Path(output_dir) / gt_file.name

        if not pred_file.exists():
            continue

        # Load GT and Pred labels
        with open(gt_file, 'r') as f:
            gt_lines = [parse_yolo_line(line) for line in f if line.strip()]
        with open(pred_file, 'r') as f:
            pred_lines = [parse_yolo_line(line) for line in f if line.strip()]

        matched_preds = set()
        for gt_class, gt_bbox, _ in gt_lines:
            gt_box = yolo_to_xyxy(*gt_bbox)
            best_iou = 0
            best_idx = -1

            for idx, (pred_class, pred_bbox, _) in enumerate(pred_lines):
                if idx in matched_preds:
                    continue
                pred_box = yolo_to_xyxy(*pred_bbox)
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_threshold and best_idx != -1:
                matched_preds.add(best_idx)

        # Simpan hanya prediksi yang match
        filtered = [pred_lines[i][2] for i in matched_preds]
        with open(out_file, 'w') as f:
            f.write('\n'.join(filtered) + '\n' if filtered else '')

    print(f"Selesai. Filtered predictions disimpan di: {output_dir}")

# Ganti path sesuai kebutuhan
gt_path = r"E:\Videos\dataset\pose\70\labels\val"
pred_path = r"E:\Videos\dataset\pose\70\pred-labels-val"
filtered_output = r"E:\Videos\dataset\pose\70\filtered-pred"

filter_predictions_by_gt(gt_path, pred_path, filtered_output)
