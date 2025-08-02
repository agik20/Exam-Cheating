import os
import cv2
import numpy as np
from ultralytics import YOLO

def predict_images_in_folder(model_path, input_folder, output_folder, conf_threshold=0.25):
    # Load model
    model = YOLO(model_path)
    
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image in input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt')
            
            # Read and predict image with confidence threshold
            img = cv2.imread(image_path)
            results = model.predict(img, conf=conf_threshold)
            
            # Prepare to write results to file
            with open(output_path, 'w') as f:
                for result in results:
                    boxes = result.boxes
                    keypoints = result.keypoints
                    
                    # If no detections, skip this image
                    if len(boxes) == 0:
                        continue
                    
                    # Filter boxes by confidence threshold (additional safety check)
                    mask = boxes.conf.cpu().numpy() >= conf_threshold
                    boxes_np = boxes.xywhn.cpu().numpy()[mask]  # Normalized xywh
                    confs_np = boxes.conf.cpu().numpy()[mask]   # Confidence scores
                    class_ids_np = boxes.cls.cpu().numpy()[mask] # Class IDs
                    
                    # Get keypoints data (filtered by the same mask)
                    kpts_data = []
                    kpts_confs = []
                    if keypoints is not None:
                        kpts_data = [k.xyn[0].cpu().numpy() for k, m in zip(keypoints, mask) if m]
                        kpts_confs = [k.conf[0].cpu().numpy() if k.conf is not None else np.ones(len(k.xyn[0])) 
                                     for k, m in zip(keypoints, mask) if m]
                    
                    # Apply NMS to the filtered boxes
                    keep_indices = nms(boxes_np, confs_np)
                    
                    # Write only the kept results to file
                    for idx in keep_indices:
                        class_id = int(class_ids_np[idx])
                        bbox = boxes_np[idx]
                        
                        # Prepare output line
                        line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                        
                        # Add keypoints if available
                        if len(kpts_data) > idx:
                            kpts = kpts_data[idx]
                            confs = kpts_confs[idx] if idx < len(kpts_confs) else np.ones(len(kpts))
                            
                            for kpt, conf in zip(kpts, confs):
                                x, y = kpt
                                line += f" {x:.6f} {y:.6f} {conf:.2f}"
                        
                        f.write(line + "\n")

def nms(boxes, scores, iou_threshold=0.5):
    # Convert xywh to xyxy format
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    areas = width * height
    order = scores.argsort()[::-1]  # Sort boxes by confidence score (highest first)
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate IoU between the current box and all remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]  # +1 because we skipped the first element
    
    return keep

if __name__ == "__main__":
    # Configure paths and parameters
    model_path = r"C:\Users\HeimTech\Downloads\063025\pose-ft-13kpt-70m\weights\best.pt"
    input_folder = r"E:\Videos\dataset\pose\70\images_before_preprocessing\val"
    output_folder = r"E:\Videos\dataset\pose\70\pred-labels-val"
    
    # Adjustable confidence threshold (default: 0.25)
    conf_threshold = 0.25  # You can change this value between 0 and 1
    
    # Run prediction with confidence threshold
    predict_images_in_folder(model_path, input_folder, output_folder, conf_threshold)
    print(f"Prediction complete with confidence threshold {conf_threshold}. Results saved in {output_folder}")