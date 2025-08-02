import os
import csv
import numpy as np
import math

# Constants for OKS calculation
K_COCO = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079,
          0.072, 0.072, 0.062, 0.062, 0.107, 0.107]  # Per-keypoint constants
SIGMA = 2  # Standard deviation multiplier for OKS

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

def calculate_oks(gt_entry, pred_entry):
    """
    Menghitung OKS total antara GT dan prediksi
    Mengembalikan:
    - OKS total (rata-rata keypoint yang valid)
    - Jumlah keypoint valid
    """
    # Hitung scale factor (s) sebagai akar luas bbox GT
    s = math.sqrt(gt_entry['bbox'][2] * gt_entry['bbox'][3])
    
    sum_oks = 0
    valid_kpt_count = 0
    
    for i, (gt_kpt, pred_kpt) in enumerate(zip(gt_entry['kpts'], pred_entry['kpts'])):
        # Skip jika keypoint tidak valid
        if gt_kpt['v'] == 0 or pred_kpt['conf'] < 0.01:
            continue
            
        # Hitung jarak Euclidean
        dx = gt_kpt['x'] - pred_kpt['x']
        dy = gt_kpt['y'] - pred_kpt['y']
        d_squared = dx*dx + dy*dy
        
        # Dapatkan konstanta k untuk keypoint ini
        k = K_COCO[i] if i < len(K_COCO) else 0.1  # Default jika K_COCO tidak mencukupi
        
        # Hitung OKS untuk keypoint ini
        exponent = -d_squared / (2 * (s * k * SIGMA) ** 2)
        oks = math.exp(exponent)
        
        sum_oks += oks
        valid_kpt_count += 1
    
    # Hitung OKS total (rata-rata keypoint yang valid)
    oks_total = sum_oks / valid_kpt_count if valid_kpt_count > 0 else 0.0
    
    return oks_total, valid_kpt_count

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
        
        # Hitung OKS untuk pasangan ini
        oks_total, valid_kpt_count = calculate_oks(gt_entries[gt_idx], pred_entries[pred_idx])
        
        # Tambahkan ke matches dengan informasi OKS
        matches.append({
            'gt': gt_entries[gt_idx],
            'pred': pred_entries[pred_idx],
            'iou': max_iou,
            'oks_total': oks_total,
            'valid_kpt_count': valid_kpt_count
        })
        
        # Set baris dan kolom yang sudah dicocokkan ke 0
        iou_matrix[gt_idx, :] = 0
        iou_matrix[:, pred_idx] = 0
    
    # Temukan GT yang tidak cocok
    for i, gt in enumerate(gt_entries):
        if not any(gt == match['gt'] for match in matches):
            unmatched_gt.append(gt)
    
    # Temukan prediksi yang tidak cocok
    for j, pred in enumerate(pred_entries):
        if not any(pred == match['pred'] for match in matches):
            unmatched_pred.append(pred)
    
    return matches, unmatched_gt, unmatched_pred

def save_to_csv_alternating(matches, unmatched_gt, unmatched_pred, output_file, num_kpts=13):
    """Menyimpan hasil pencocokan ke file CSV dengan format alternating rows"""
    with open(output_file, 'w', newline='') as csvfile:
        # Buat header kolom
        fieldnames = [
            'row_type', 'source_file', 'class_id', 
            'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 
            'iou', 'oks_total', 'valid_kpt_count'
        ]
        
        # Tambahkan kolom untuk setiap keypoint (hanya koordinat dan visibility/confidence)
        for i in range(num_kpts):
            fieldnames.extend([
                f'kpt_{i}_x', f'kpt_{i}_y', f'kpt_v_or_conf'
            ])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Fungsi untuk menulis satu baris data
        def write_row(row_type, entry, iou=None, oks_total=None, valid_kpt_count=None, is_gt=True):
            row = {
                'row_type': row_type,
                'source_file': entry['source_file'],
                'class_id': entry['class_id'],
                'bbox_x': entry['bbox'][0],
                'bbox_y': entry['bbox'][1],
                'bbox_w': entry['bbox'][2],
                'bbox_h': entry['bbox'][3],
                'iou': iou if iou is not None else '',
                'oks_total': oks_total if oks_total is not None else '',
                'valid_kpt_count': valid_kpt_count if valid_kpt_count is not None else ''
            }
            
            for i, kpt in enumerate(entry['kpts']):
                if i >= num_kpts:
                    break
                
                row.update({
                    f'kpt_{i}_x': kpt['x'],
                    f'kpt_{i}_y': kpt['y'],
                    'kpt_v_or_conf': kpt['v'] if is_gt else kpt['conf']
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
        for match in matches:
            # Header untuk pasangan
            writer.writerow({
                'row_type': 'matched_pair_header', 
                'source_file': f"IoU: {match['iou']:.4f}, OKS: {match['oks_total']:.4f}, Valid KP: {match['valid_kpt_count']}"
            })
            
            # Tulis GT
            write_row(
                'gt', 
                match['gt'], 
                iou=match['iou'], 
                oks_total=match['oks_total'],
                valid_kpt_count=match['valid_kpt_count'],
                is_gt=True
            )
            
            # Tulis Prediksi
            write_row(
                'pred', 
                match['pred'], 
                iou=match['iou'], 
                oks_total=match['oks_total'],
                valid_kpt_count=match['valid_kpt_count'],
                is_gt=False
            )
        
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
    print(f"Menggunakan nilai sigma untuk OKS: {SIGMA}")
    
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
    
    if matches:
        avg_oks = sum(match['oks_total'] for match in matches) / len(matches)
        avg_valid_kpt = sum(match['valid_kpt_count'] for match in matches) / len(matches)
        print(f"\nRata-rata OKS untuk pasangan yang cocok: {avg_oks:.4f}")
        print(f"Rata-rata keypoint valid per pasangan: {avg_valid_kpt:.1f}")
    
    # Tampilkan lokasi file output
    print(f"\nFile output dapat ditemukan di: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()