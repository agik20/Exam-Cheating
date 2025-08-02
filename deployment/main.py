import cv2
import os
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from collections import deque

# ========== CONFIGURATION ==========
class Config:
    # Path configurations
    VIDEO_PATH = r"E:\capstone-project\datasets\exam-act\raw\1207.mp4"
    POSE_MODEL_PATH = r"C:\Users\HeimTech\Downloads\063025\pose-ft-13kpt\weights\best.pt"
    OBJECT_MODEL_PATH = r"C:\Users\HeimTech\Downloads\object-finetuning\object-70m.pt"

    OUTPUT_POSE_BASE = r"E:\capstone-project\datasets\exam-act\deteksi\pose"
    OUTPUT_OBJ_BASE = r"E:\capstone-project\datasets\exam-act\deteksi\object"
    OUTPUT_FULLROOM = r"E:\capstone-project\datasets\exam-act\deteksi\room-view"

    # Detection parameters
    POSE_CLASSES = ["no cheating", "provide object", "see friends work"]
    OBJECT_CLASSES = ["smartphone", "calculator"]
    SAVE_POSE_CLASSES = ["provide object", "see friends work"]
    SAVE_OBJECT_CLASSES = ["smartphone"]
    
    CONF_THRESH_POSE = 0.45
    CONF_THRESH_OBJ = 0.45
    DISPLAY_CROP_SCALE = 1.5      # Scale for display bounding boxes
    SAVE_CROP_SCALE = 2.5         # Scale for saved crops
    MIN_CROP_BRIGHTNESS = 30
    
    # Display parameters (adjusted to be thinner)
    POSE_BOX_THICKNESS = 1        # Thinner bounding boxes for poses
    OBJECT_BOX_THICKNESS = 1      # Thinner bounding boxes for objects
    TEXT_THICKNESS = 1            # Thickness for text labels
    
    # Cooldown parameters
    POSE_COOLDOWN = 60
    OBJECT_COOLDOWN = 60
    REQUIRED_DETECTION_TIME = 3
    
    # Color mappings
    POSE_COLORS = {
        "no cheating": (0, 255, 0), 
        "provide object": (255, 0, 0), 
        "see friends work": (0, 255, 255)
    }
    OBJECT_COLORS = {
        "smartphone": (0, 0, 255), 
        "calculator": (0, 255, 0)
    }

# ========== UTILITY FUNCTIONS ==========
def expand_crop(x1, y1, x2, y2, img_w, img_h, scale=2.5):
    """Expand bounding box coordinates by given scale while staying within image bounds"""
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    new_w, new_h = int(w * scale), int(h * scale)
    new_x1 = max(0, cx - new_w // 2)
    new_y1 = max(0, cy - new_h // 2)
    new_x2 = min(img_w, cx + new_w // 2)
    new_y2 = min(img_h, cy + new_h // 2)
    return new_x1, new_y1, new_x2, new_y2

def is_valid_crop(crop, min_brightness=30):
    """Check if crop meets minimum brightness requirement"""
    if crop is None or crop.size == 0:
        return False
    brightness = np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    return brightness >= min_brightness

def save_crop(folder_path, filename, crop):
    """Save crop image to specified folder"""
    os.makedirs(folder_path, exist_ok=True)
    cv2.imwrite(os.path.join(folder_path, filename), crop)

def log_info(msg, log_file=None, print_to_console=True):
    """Log information with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    log_msg = f"[{timestamp}] {msg}"
    if print_to_console:
        print(log_msg)
    if log_file:
        log_file.write(log_msg + "\n")

# ========== DETECTION PIPELINE ==========
class CheatingDetectionPipeline:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.initialize_models()
        self.id_info_pose = {}
        self.id_info_obj = {}
        self.frame_idx = 0
        self.prev_time = time.time()
        self.start_time = time.time()
        
    def setup_directories(self):
        """Create required output directories"""
        log_info("Setting up output directories...", print_to_console=False)
        for cls in self.config.SAVE_POSE_CLASSES:
            os.makedirs(os.path.join(self.config.OUTPUT_POSE_BASE, cls), exist_ok=True)
        for cls in self.config.SAVE_OBJECT_CLASSES:
            os.makedirs(os.path.join(self.config.OUTPUT_OBJ_BASE, cls), exist_ok=True)
        os.makedirs(self.config.OUTPUT_FULLROOM, exist_ok=True)
        log_info("Output directories ready", print_to_console=False)
    
    def initialize_models(self):
        """Initialize YOLO models"""
        log_info("Initializing pose estimation model...")
        self.pose_model = YOLO(self.config.POSE_MODEL_PATH)
        log_info("Pose model loaded successfully")
        
        log_info("Initializing object detection model...")
        self.object_model = YOLO(self.config.OBJECT_MODEL_PATH)
        log_info("Object model loaded successfully")
    
    def process_video(self):
        """Main processing pipeline for video"""
        log_file = open("detection_log.txt", "w")
        log_info("Starting video processing pipeline", log_file)
        
        cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        if not cap.isOpened():
            log_info("Error: Could not open video file", log_file)
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
        log_info(f"Video source: {self.config.VIDEO_PATH}", log_file)
        log_info(f"Frame rate: {cap.get(cv2.CAP_PROP_FPS):.2f}", log_file)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                log_info("End of video stream reached", log_file)
                break
            
            self.frame_idx += 1
            current_time = datetime.now()
            timestamp = current_time.strftime("%H%M%S")
            
            # Process frame
            log_info(f"Processing frame {self.frame_idx}", log_file, print_to_console=False)
            annotated_frame = self.process_frame(frame, current_time, timestamp, log_file)
            
            # Display and control
            self.display_results(annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log_info("Pipeline stopped by user", log_file)
                break
        
        # Calculate and log processing statistics
        processing_time = time.time() - self.start_time
        avg_fps = self.frame_idx / processing_time
        log_info(f"Processing completed. Total frames: {self.frame_idx}", log_file)
        log_info(f"Total processing time: {processing_time:.2f} seconds", log_file)
        log_info(f"Average FPS: {avg_fps:.2f}", log_file)
        
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()

    def process_frame(self, frame, current_time, timestamp, log_file):
        """Process a single frame through the detection pipeline"""
        img_h, img_w = frame.shape[:2]
        annotated = frame.copy()
        
        # Calculate FPS
        curr_time = time.time()
        display_fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time
        
        # Save full room view at specific frame
        if self.frame_idx == 1:
            log_info("Saving full room view snapshot", log_file, print_to_console=False)
            cv2.imwrite(os.path.join(self.config.OUTPUT_FULLROOM, f"room_{timestamp}.jpg"), frame)
        
        # Pose detection and tracking
        log_info("Running pose detection...", log_file, print_to_console=False)
        annotated = self.detect_pose(annotated, frame, current_time, timestamp, img_w, img_h, log_file)
        
        # Object detection and tracking
        log_info("Running object detection...", log_file, print_to_console=False)
        annotated = self.detect_objects(annotated, frame, current_time, timestamp, img_w, img_h, log_file)
        
        # Log frame information
        log_msg = f"Frame {self.frame_idx} processed - Pose IDs: {len(self.id_info_pose)}, Object IDs: {len(self.id_info_obj)}"
        log_info(log_msg, log_file, print_to_console=False)
        
        # Add FPS and frame info to display
        cv2.putText(annotated, f"FPS: {display_fps:.1f}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), self.config.TEXT_THICKNESS)
        cv2.putText(annotated, f"Frame: {self.frame_idx}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), self.config.TEXT_THICKNESS)
        
        return annotated
    
    def detect_pose(self, annotated, frame, current_time, timestamp, img_w, img_h, log_file):
        pose_results = self.pose_model.track(
            source=frame, 
            conf=self.config.CONF_THRESH_POSE, 
            persist=True, 
            imgsz=640
        )
        pose_boxes = pose_results[0].boxes
        
        if pose_boxes is not None and pose_boxes.id is not None and pose_boxes.cls is not None:
            log_info(f"Found {len(pose_boxes)} pose detections", log_file, print_to_console=False)
            
            for box, cls_id, track_id, conf in zip(pose_boxes.xyxy, pose_boxes.cls, pose_boxes.id, pose_boxes.conf):
                cls_id, track_id = int(cls_id), int(track_id)
                class_name = self.config.POSE_CLASSES[cls_id]
                color = self.config.POSE_COLORS.get(class_name, (255, 255, 255))
                
                x1, y1, x2, y2 = map(int, box)
                dx1, dy1, dx2, dy2 = expand_crop(x1, y1, x2, y2, img_w, img_h, self.config.DISPLAY_CROP_SCALE)
                
                cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), color, self.config.POSE_BOX_THICKNESS)
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                cv2.putText(annotated, label, (dx1, dy1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, self.config.TEXT_THICKNESS)
                
                if class_name in self.config.SAVE_POSE_CLASSES:
                    self.process_detection(
                        track_id, class_name, x1, y1, x2, y2, 
                        frame, current_time, timestamp, img_w, img_h,
                        self.id_info_pose, self.config.POSE_COOLDOWN,
                        self.config.OUTPUT_POSE_BASE, "pose", log_file,
                        conf=float(conf)  # ✅ Pass confidence ke process_detection
                    )
        else:
            log_info("No pose detections in this frame", log_file, print_to_console=False)
        
        return annotated
    
    def detect_objects(self, annotated, frame, current_time, timestamp, img_w, img_h, log_file):
        obj_results = self.object_model.track(
            source=frame, 
            conf=self.config.CONF_THRESH_OBJ, 
            persist=True, 
            imgsz=640
        )
        obj_boxes = obj_results[0].boxes
        
        if obj_boxes is not None and obj_boxes.id is not None:
            log_info(f"Found {len(obj_boxes)} object detections", log_file, print_to_console=False)
            
            for box, cls_id, track_id, conf in zip(obj_boxes.xyxy, obj_boxes.cls, obj_boxes.id, obj_boxes.conf):
                cls_id, track_id = int(cls_id), int(track_id)
                class_name = self.config.OBJECT_CLASSES[cls_id]
                color = self.config.OBJECT_COLORS.get(class_name, (255, 255, 255))
                
                x1, y1, x2, y2 = map(int, box)
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.config.OBJECT_BOX_THICKNESS)
                cv2.putText(annotated, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, self.config.TEXT_THICKNESS)
                
                if class_name in self.config.SAVE_OBJECT_CLASSES:
                    self.process_detection(
                        track_id, class_name, x1, y1, x2, y2, 
                        frame, current_time, timestamp, img_w, img_h,
                        self.id_info_obj, self.config.OBJECT_COOLDOWN,
                        self.config.OUTPUT_OBJ_BASE, "object", log_file,
                        conf=float(conf)  # ✅ Pass confidence ke process_detection
                    )
        else:
            log_info("No object detections in this frame", log_file, print_to_console=False)
        
        return annotated
    
    def process_detection(self, track_id, class_name, x1, y1, x2, y2, 
                        frame, current_time, timestamp, img_w, img_h,
                        id_info, cooldown, output_base, detection_type, log_file, conf=None):
        """Process and save detection if conditions are met"""
        if track_id not in id_info:
            id_info[track_id] = {
                'first_time': current_time,
                'last_saved': None,
                'last_class': class_name,
                'last_conf': conf
            }
            log_info(f"New {detection_type} ID detected: {track_id} ({class_name})", log_file, print_to_console=False)
        
        info = id_info[track_id]
        
        if conf is not None:
            info['last_conf'] = conf
        
        if info['last_saved'] and (current_time - info['last_saved']).total_seconds() < cooldown:
            log_info(f"{detection_type.capitalize()} ID {track_id} in cooldown", log_file, print_to_console=False)
            return
        
        if info['last_class'] != class_name:
            info['first_time'] = current_time
            info['last_class'] = class_name
            log_info(f"{detection_type.capitalize()} ID {track_id} class changed to {class_name}", log_file, print_to_console=False)
        
        elif (current_time - info['first_time']).total_seconds() >= self.config.REQUIRED_DETECTION_TIME:
            # Gunakan SAVE_CROP_SCALE untuk semua deteksi (pose & object)
            ex1, ey1, ex2, ey2 = expand_crop(x1, y1, x2, y2, img_w, img_h, self.config.SAVE_CROP_SCALE)
            crop = frame[ey1:ey2, ex1:ex2]
            
            if is_valid_crop(crop, self.config.MIN_CROP_BRIGHTNESS):
                conf_str = f"{info['last_conf']:.2f}" if info['last_conf'] is not None else "0.00"
                filename = f"{track_id}_{class_name}_{conf_str}_{timestamp}.jpg"
                save_path = os.path.join(output_base, class_name, filename)
                save_crop(os.path.join(output_base, class_name), filename, crop)
                info['last_saved'] = current_time
                log_info(f"Saved {detection_type} crop: {save_path}", log_file, print_to_console=False)
            else:
                log_info(f"Invalid crop for {detection_type} ID {track_id} - brightness too low", log_file, print_to_console=False)
    
    def display_results(self, annotated_frame):
        """Display annotated frame with additional information"""
        # Add processing information overlay with thin text
        cv2.putText(annotated_frame, f"Tracking {len(self.id_info_pose)} persons", 
                   (10, annotated_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), self.config.TEXT_THICKNESS)
        cv2.putText(annotated_frame, f"Tracking {len(self.id_info_obj)} objects", 
                   (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), self.config.TEXT_THICKNESS)
        
        cv2.imshow("Detection Results", annotated_frame)

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    log_info("Initializing cheating detection pipeline")
    pipeline = CheatingDetectionPipeline(Config())
    log_info("Starting video processing")
    pipeline.process_video()
    log_info("Pipeline execution completed")