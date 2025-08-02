from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-pose.pt")

# Train the model
model.train(
    data=r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\scripts\pre-processing-model\training-datasets\exam-act\config.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    weight_decay=0.0005,
    project="runs/train",
    name="keypoints-cheating-v5",
    save_period=10, 
    val=True,
    close_mosaic=10,
    patience=10         
)
