import os
import numpy as np
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
gt_path = r"E:\Videos\dataset\pose\70\labels\test" 
pred_path = r"E:\Videos\dataset\pose\70\filtered-pred" 

class_map = {
    0: 'no-cheating',
    1: 'provide-object',
    2: 'see-friends-work'
}

classes = list(class_map.values())

# Threshold for matching bounding boxes (IoU or distance threshold)
IOU_THRESHOLD = 0.5  # You can adjust this based on your needs

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is [x_center, y_center, width, height]
    """
    # Convert from center coordinates to corner coordinates
    box1 = [
        box1[0] - box1[2]/2,  # x1
        box1[1] - box1[3]/2,  # y1
        box1[0] + box1[2]/2,  # x2
        box1[1] + box1[3]/2   # y2
    ]
    box2 = [
        box2[0] - box2[2]/2,
        box2[1] - box2[3]/2,
        box2[0] + box2[2]/2,
        box2[1] + box2[3]/2
    ]
    
    # Calculate intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

def match_boxes(gt_boxes, pred_boxes):
    """
    Match GT and predicted boxes based on IoU.
    Returns matched pairs, unmatched GTs, and unmatched predictions.
    """
    matched_pairs = []
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    # First pass: find best matches for each GT
    for gt_idx, gt_box in enumerate(gt_boxes):
        best_iou = IOU_THRESHOLD
        best_pred_idx = -1
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            if pred_idx in matched_pred_indices:
                continue
                
            iou = calculate_iou(gt_box[1:5], pred_box[1:5])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
                
        if best_pred_idx != -1:
            matched_pairs.append((gt_idx, best_pred_idx))
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(best_pred_idx)
    
    # Second pass: find remaining predictions that might match any GT
    for pred_idx, pred_box in enumerate(pred_boxes):
        if pred_idx in matched_pred_indices:
            continue
            
        best_iou = IOU_THRESHOLD
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt_indices:
                continue
                
            iou = calculate_iou(gt_box[1:5], pred_box[1:5])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        if best_gt_idx != -1:
            matched_pairs.append((best_gt_idx, pred_idx))
            matched_gt_indices.add(best_gt_idx)
            matched_pred_indices.add(pred_idx)
    
    # Identify unmatched GTs and predictions
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gt_indices]
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_pred_indices]
    
    return matched_pairs, unmatched_gts, unmatched_preds

def calculate_metrics(cm):
    """Calculate precision, recall, f1-score for each class from confusion matrix"""
    metrics = {}
    
    for i, class_name in enumerate(classes):
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': sum(cm[i, :])
        }
    
    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    
    return metrics, accuracy

def main():
    y_true = []
    y_pred = []
    orphan_gts = []
    orphan_preds = []
    
    gt_files = sorted(glob(os.path.join(gt_path, "*.txt")))
    
    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_path, filename)
        
        # Load GT data
        try:
            gt_data = np.loadtxt(gt_file, ndmin=2)
            gt_boxes = [row[:5] for row in gt_data]  # class_id, x, y, w, h
        except Exception:
            gt_boxes = []
        
        # Load prediction data
        try:
            if os.path.exists(pred_file):
                pred_data = np.loadtxt(pred_file, ndmin=2)
                pred_boxes = [row[:5] for row in pred_data]  # class_id, x, y, w, h, ...
            else:
                pred_boxes = []
        except Exception:
            pred_boxes = []
        
        # Match boxes between GT and predictions
        matched_pairs, unmatched_gts, unmatched_preds = match_boxes(gt_boxes, pred_boxes)
        
        # Process matched pairs
        for gt_idx, pred_idx in matched_pairs:
            gt_class = int(gt_boxes[gt_idx][0])
            pred_class = int(pred_boxes[pred_idx][0])
            
            y_true.append(class_map[gt_class])
            y_pred.append(class_map[pred_class])
        
        # Record orphans
        for gt_idx in unmatched_gts:
            gt_box = gt_boxes[gt_idx]
            orphan_gts.append({
                'file': filename,
                'class': class_map[int(gt_box[0])],
                'bbox': gt_box[1:5].tolist()
            })
        
        for pred_idx in unmatched_preds:
            pred_box = pred_boxes[pred_idx]
            orphan_preds.append({
                'file': filename,
                'class': class_map[int(pred_box[0])],
                'bbox': pred_box[1:5].tolist()
            })
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Calculate metrics
    metrics, accuracy = calculate_metrics(cm)
    
    # Print results
    print("\nEvaluation Results:")
    print("="*50)
    print(f"Total matched pairs: {len(y_true)}")
    print(f"Orphan GT boxes: {len(orphan_gts)}")
    print(f"Orphan predicted boxes: {len(orphan_preds)}\n")
    
    print("Confusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print("="*50)
    header = " " * 15 + " ".join([f"{c[:10]:>10}" for c in classes])
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i][:14]:<15}", end="")
        for val in row:
            print(f"{val:>10}", end=" ")
        print()
    print()
    
    print("Classification Metrics:")
    print("="*50)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    for class_name in classes:
        m = metrics[class_name]
        print(f"{class_name:<20} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f} {m['support']:<10}")
    print(f"\nOverall Accuracy: {accuracy:.4f}\n")
    
    # Print orphan information
    if orphan_gts:
        print("\nOrphan GT Boxes (not matched to any prediction):")
        print("="*50)
        for orphan in orphan_gts[:5]:  # Print first 5 to avoid too much output
            print(f"File: {orphan['file']}, Class: {orphan['class']}, BBox: {orphan['bbox']}")
        if len(orphan_gts) > 5:
            print(f"... and {len(orphan_gts) - 5} more orphan GT boxes")
    
    if orphan_preds:
        print("\nOrphan Predicted Boxes (not matched to any GT):")
        print("="*50)
        for orphan in orphan_preds[:5]:  # Print first 5 to avoid too much output
            print(f"File: {orphan['file']}, Class: {orphan['class']}, BBox: {orphan['bbox']}")
        if len(orphan_preds) > 5:
            print(f"... and {len(orphan_preds) - 5} more orphan predicted boxes")
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix', fontsize=15, pad=20)
    plt.xlabel('Prediction', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.show()

if __name__ == "__main__":
    main()