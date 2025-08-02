import os
import cv2
import numpy as np
from ultralytics import YOLO

def visualize_and_save_images(pose_model_path, object_model_path, input_folder, output_folder,
                               conf_threshold=0.25, iou_threshold=0.5):

    pose_model = YOLO(pose_model_path)
    object_model = YOLO(object_model_path)
    os.makedirs(output_folder, exist_ok=True)

    config = {
        'POSE_CLASSES': {
            0: "no-cheating",
            1: "provide-object",
            2: "see-friends-work"
        },
        'POSE_COLORS': {
            "no-cheating": (0, 255, 0),
            "provide-object": (255, 0, 0),
            "see-friends-work": (0, 0, 255)
        },
        'OBJECT_CLASSES': {
            0: "smartphone",
            1: "calculator"
        },
        'OBJECT_COLORS': {
            "smartphone": (0, 255, 255),   # Yellow
            "calculator": (255, 0, 255)    # Magenta
        },
        'POSE_BOX_THICKNESS': 1,
        'OBJECT_BOX_THICKNESS': 1,
        'TEXT_THICKNESS': 1,
        'DISPLAY_CROP_SCALE': 0.2
    }

    def expand_crop(x1, y1, x2, y2, img_w, img_h, scale):
        width, height = x2 - x1, y2 - y1
        dx, dy = int(width * scale), int(height * scale)
        return max(0, x1 - dx), max(0, y1 - dy), min(img_w, x2 + dx), min(img_h, y2 + dy)

    def nms(boxes, scores, iou_threshold):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h)
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        return keep

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue
        img_h, img_w = img.shape[:2]

        ### === POSE INFERENCE ===
        pose_results = pose_model.predict(img, conf=conf_threshold)
        for result in pose_results:
            boxes, keypoints = result.boxes, result.keypoints
            if len(boxes) == 0:
                continue
            boxes_np = boxes.xyxy.cpu().numpy()
            confs_np = boxes.conf.cpu().numpy()
            class_ids_np = boxes.cls.cpu().numpy().astype(int)
            keep_indices = nms(boxes_np, confs_np, iou_threshold)
            kpts_np = None
            if keypoints is not None:
                kpts_np = [k.xy[0].cpu().numpy() for k in keypoints]
            for idx in keep_indices:
                class_id = class_ids_np[idx]
                class_name = config['POSE_CLASSES'].get(class_id, f"class-{class_id}")
                color = config['POSE_COLORS'].get(class_name, (255, 255, 255))
                bbox = boxes_np[idx].astype(int)
                conf = confs_np[idx]
                x1, y1, x2, y2 = bbox
                dx1, dy1, dx2, dy2 = expand_crop(x1, y1, x2, y2, img_w, img_h, config['DISPLAY_CROP_SCALE'])
                cv2.rectangle(img, (dx1, dy1), (dx2, dy2), color, config['POSE_BOX_THICKNESS'])
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (dx1, dy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, config['TEXT_THICKNESS'])
                if kpts_np is not None and idx < len(kpts_np):
                    for kp in kpts_np[idx]:
                        cv2.circle(img, tuple(kp.astype(int)), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        ### === OBJECT INFERENCE ===
        object_results = object_model.predict(img, conf=conf_threshold)
        for result in object_results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            boxes_np = boxes.xyxy.cpu().numpy()
            confs_np = boxes.conf.cpu().numpy()
            class_ids_np = boxes.cls.cpu().numpy().astype(int)
            keep_indices = nms(boxes_np, confs_np, iou_threshold)
            for idx in keep_indices:
                class_id = class_ids_np[idx]
                class_name = config['OBJECT_CLASSES'].get(class_id, f"class-{class_id}")
                color = config['OBJECT_COLORS'].get(class_name, (0, 255, 255))
                bbox = boxes_np[idx].astype(int)
                conf = confs_np[idx]
                x1, y1, x2, y2 = bbox
                dx1, dy1, dx2, dy2 = expand_crop(x1, y1, x2, y2, img_w, img_h, config['DISPLAY_CROP_SCALE'])
                cv2.rectangle(img, (dx1, dy1), (dx2, dy2), color, config['OBJECT_BOX_THICKNESS'])
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (dx1, dy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, config['TEXT_THICKNESS'])

        cv2.imwrite(output_path, img)
        print(f"Visualized image saved: {output_path}")


if __name__ == "__main__":
    pose_model_path = r"/media/heimy/sd-card/capstone-project/model/pose-70m.pt"
    object_model_path = r"/media/heimy/sd-card/capstone-project/model/object-70m.pt"
    input_folder = r"/media/heimy/sd-card/capstone-project/output/frames"
    output_folder = r"/media/heimy/sd-card/capstone-project/output/vis"

    visualize_and_save_images(
        pose_model_path,
        object_model_path,
        input_folder,
        output_folder,
        conf_threshold=0.25,
        iou_threshold=0.5
    )
    print("Dual-model visualization complete!")
