from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\HeimTech\Documents\Tugas Akhir\Final Project\scripts\pre-processing-model\training-datasets\yolov8m-pose.pt")  # load an official model
model = YOLO(r"C:\Users\HeimTech\Downloads\063025\pose-ft-13kpt-70m\weights\best.pt")  # load a custom model

# Validate the model
metrics = model.val(data=r"E:\Videos\dataset\pose\70\data.yaml")  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
metrics.pose.map  # map50-95(P)
metrics.pose.map50  # map50(P)
metrics.pose.map75  # map75(P)
metrics.pose.maps  # a list contains map50-95(P) of each category