from ultralytics import YOLO

model = YOLO(r"C:\Users\HeimTech\Downloads\lab_output\pose-ft-13kpt-70m\weights\best.pt")

metrics = model.val(
    data=r"D:\Videos\model\pose-estimation\training_akhir\dataset\data.yaml",
    split="test"  # gunakan dataset test, bukan val
)

metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
metrics.pose.map  # map50-95(P)
metrics.pose.map50  # map50(P)
metrics.pose.map75  # map75(P)
metrics.pose.maps