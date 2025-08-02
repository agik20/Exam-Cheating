from ultralytics import YOLO

# Load YOLOv8 Medium
model = YOLO("yolov8m.pt")

# Train model
model.train(
    data=r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\scripts\pre-processing-model\training-datasets\exam-act\smartphone-detection.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    weight_decay=0.0005,
    project="runs/train",
    name="bounding-box-smartphone",
    save_period=10,
    val=True,
    patience=10,
    box=0.05,
    cls=0.5,
    mosaic=0.5,
    mixup=0.1,
    translate=0.05,
    scale=0.5,
    degrees=0.0,
    shear=0.0,
    plots=True
)
