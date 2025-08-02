import os
import csv
import numpy as np

def read_label_file(file_path, is_gt=True):
    """Membaca file label (GT atau prediksi) dan mengembalikan list of entries"""
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Parse data dasar
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:5]))
            
            # Parse keypoints
            kpts = []
            if is_gt:
                # Format GT: x y v x y v ...
                kpts_data = parts[5:]
                for i in range(0, len(kpts_data), 3):
                    x = float(kpts_data[i])
                    y = float(kpts_data[i+1])
                    v = int(kpts_data[i+2])
                    kpts.append({'x': x, 'y': y, 'v': v})
            else:
                # Format Prediksi: x y conf x y conf ...
                kpts_data = parts[5:]
                for i in range(0, len(kpts_data), 3):
                    x = float(kpts_data[i])
                    y = float(kpts_data[i+1])
                    conf = float(kpts_data[i+2])
                    kpts.append({'x': x, 'y': y, 'conf': conf})
            
            entries.append({
                'class_id': class_id,
                'bbox': bbox,
                'kpts': kpts,
                'source_file': os.path.basename(file_path)
            })
    return entries

def calculate_iou(box1, box2):
    """Menghitung Intersection over Union (IoU) antara dua bounding box"""
    # box format: [x_center, y_center, width, height]
    
    # Konversi ke format (x1, y1, x2, y2)
    box1_x1 = box1[0] - box1[2]/2
    box1_y1 = box1[1] - box1[3]/2
    box1_x2 = box1[0] + box1[2]/2
    box1_y2 = box1[1] + box1[3]/2
    
    box2_x1 = box2[0] - box2[2]/2
    box2_y1 = box2[1] - box2[3]/2
    box2_x2 = box2[0] + box2[2]/2
    box2_y2 = box2[1] + box2[3]/2
    
    # Hitung area intersection
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Hitung area union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area
    
    # Hitung IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def match_entries(gt_entries, pred_entries, iou_threshold=0.5):
    """Mencocokkan entri GT dan prediksi berdasarkan IoU bbox"""
    matches = []
    unmatched_gt = []
    unmatched_pred = []
    
    # Buat matriks IoU antara semua GT dan prediksi
    iou_matrix = np.zeros((len(gt_entries), len(pred_entries)))
    for i, gt in enumerate(gt_entries):
        for j, pred in enumerate(pred_entries):
            iou_matrix[i, j] = calculate_iou(gt['bbox'], pred['bbox'])
    
    # Cocokkan GT dan prediksi dengan IoU tertinggi
    while True:
        max_iou = np.max(iou_matrix)
        if max_iou < iou_threshold:
            break
            
        gt_idx, pred_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        matches.append((gt_entries[gt_idx], pred_entries[pred_idx]))
        
        # Set baris dan kolom yang sudah dicocokkan ke 0
        iou_matrix[gt_idx, :] = 0
        iou_matrix[:, pred_idx] = 0
    
    # Temukan GT yang tidak cocok
    for i, gt in enumerate(gt_entries):
        if not any(gt == match[0] for match in matches):
            unmatched_gt.append(gt)
    
    # Temukan prediksi yang tidak cocok
    for j, pred in enumerate(pred_entries):
        if not any(pred == match[1] for match in matches):
            unmatched_pred.append(pred)
    
    return matches, unmatched_gt, unmatched_pred

def save_to_csv_alternating(matches, unmatched_gt, unmatched_pred, output_file, num_kpts=13):
    """Menyimpan hasil pencocokan ke file CSV dengan format alternating rows"""
    with open(output_file, 'w', newline='') as csvfile:
        # Buat header kolom
        fieldnames = [
            'row_type', 'source_file', 'class_id', 
            'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'iou'
        ]
        
        # Tambahkan kolom untuk setiap keypoint
        for i in range(num_kpts):
            fieldnames.extend([
                f'kpt_{i}_x', f'kpt_{i}_y', f'kpt_v_or_conf'
            ])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Fungsi untuk menulis satu baris data
        def write_row(row_type, entry, iou=None, is_gt=True):
            row = {
                'row_type': row_type,
                'source_file': entry['source_file'],
                'class_id': entry['class_id'],
                'bbox_x': entry['bbox'][0],
                'bbox_y': entry['bbox'][1],
                'bbox_w': entry['bbox'][2],
                'bbox_h': entry['bbox'][3],
                'iou': iou if iou is not None else ''
            }
            
            for i, kpt in enumerate(entry['kpts']):
                if i >= num_kpts:
                    break
                if is_gt:
                    row.update({
                        f'kpt_{i}_x': kpt['x'],
                        f'kpt_{i}_y': kpt['y'],
                        'kpt_v_or_conf': kpt['v']
                    })
                else:
                    row.update({
                        f'kpt_{i}_x': kpt['x'],
                        f'kpt_{i}_y': kpt['y'],
                        'kpt_v_or_conf': kpt['conf']
                    })
            
            # Isi kolom yang kosong jika keypoint kurang dari num_kpts
            for i in range(len(entry['kpts']), num_kpts):
                row.update({
                    f'kpt_{i}_x': '',
                    f'kpt_{i}_y': '',
                    'kpt_v_or_conf': ''
                })
            
            writer.writerow(row)
        
        # Tulis entri yang cocok dalam format alternating
        for gt, pred in matches:
            iou = calculate_iou(gt['bbox'], pred['bbox'])
            writer.writerow({'row_type': 'matched_pair_header', 'source_file': f"IoU: {iou:.2f}"})
            write_row('gt', gt, iou, is_gt=True)
            write_row('pred', pred, iou, is_gt=False)
        
        # Tulis GT yang tidak cocok
        if unmatched_gt:
            writer.writerow({'row_type': 'unmatched_header', 'source_file': 'Unmatched Ground Truth'})
            for gt in unmatched_gt:
                write_row('unmatched_gt', gt, is_gt=True)
        
        # Tulis prediksi yang tidak cocok
        if unmatched_pred:
            writer.writerow({'row_type': 'unmatched_header', 'source_file': 'Unmatched Predictions'})
            for pred in unmatched_pred:
                write_row('unmatched_pred', pred, is_gt=False)

def get_output_filename(gt_file, pred_file):
    """Generate output filename based on input folder names"""
    # Extract folder names from paths
    gt_folder = os.path.basename(os.path.dirname(gt_file))
    pred_folder = os.path.basename(os.path.dirname(pred_file))
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(gt_file))[0]
    
    # Create combined filename
    return f"{base_name}_{gt_folder}_vs_{pred_folder}_combined_results.csv"

def main():
    """Fungsi utama untuk memproses file dan menghasilkan output CSV"""
    # Path file yang sudah ditentukan
    gt_file = r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\capstone\inference\data-test\gt-labels\test\frame_1050_145850.txt"
    pred_file = r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\capstone\inference\data-test\labels\test\frame_1050_145850.txt"
    
    # Generate output filename
    output_csv = get_output_filename(gt_file, pred_file)
    output_dir = os.path.dirname(gt_file)
    output_path = os.path.join(output_dir, output_csv)
    
    # Threshold IoU dan jumlah keypoints
    iou_threshold = 0.3
    num_keypoints = 13  # Sesuaikan dengan jumlah keypoints Anda
    
    print("=== Program Pencocokan Ground Truth dan Prediksi ===")
    print(f"\nMenggunakan file ground truth: {gt_file}")
    print(f"Menggunakan file prediksi: {pred_file}")
    print(f"Output akan disimpan ke: {output_path}")
    print(f"Menggunakan threshold IoU: {iou_threshold}")
    print(f"Jumlah keypoints: {num_keypoints}")
    
    # Baca file GT dan prediksi
    try:
        print("\nMembaca file ground truth...")
        gt_entries = read_label_file(gt_file, is_gt=True)
        print("Membaca file prediksi...")
        pred_entries = read_label_file(pred_file, is_gt=False)
    except Exception as e:
        print(f"\nError membaca file: {e}")
        return
    
    # Cocokkan entri berdasarkan bbox
    print("Mencocokkan entri berdasarkan bounding box...")
    matches, unmatched_gt, unmatched_pred = match_entries(gt_entries, pred_entries, iou_threshold)
    
    # Simpan ke CSV
    try:
        print("Menyimpan hasil ke CSV...")
        save_to_csv_alternating(matches, unmatched_gt, unmatched_pred, output_path, num_keypoints)
        print(f"\nProses selesai. Hasil disimpan ke {output_path}")
    except Exception as e:
        print(f"\nError menyimpan file CSV: {e}")
        return
    
    # Tampilkan statistik
    print("\nStatistik:")
    print(f"Jumlah GT: {len(gt_entries)}")
    print(f"Jumlah Prediksi: {len(pred_entries)}")
    print(f"Jumlah yang cocok: {len(matches)}")
    print(f"GT tidak cocok: {len(unmatched_gt)}")
    print(f"Prediksi tidak cocok: {len(unmatched_pred)}")
    
    # Tampilkan lokasi file output
    print(f"\nFile output dapat ditemukan di: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()