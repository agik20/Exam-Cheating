import pandas as pd
import matplotlib.pyplot as plt

# Ganti path dengan lokasi file kamu
df = pd.read_csv(r"E:\Videos\model\pose\yolo-pose-70m\results.csv")
df.columns = df.columns.str.strip()  # hilangkan spasi ekstra di nama kolom

# Scatter plot: val/pose_loss vs mAP@0.5(P)
plt.figure(figsize=(10, 6))
plt.scatter(
    df["val/pose_loss"], 
    df["metrics/mAP50(P)"], 
    c=df["epoch"], 
    cmap="viridis", 
    edgecolor="black"
)

plt.colorbar(label="Epoch")
plt.xlabel("Validation Pose Loss")
plt.ylabel("mAP@0.5 (Pose)")
plt.title("Trade-off Between Pose Loss and Pose mAP@0.5")
plt.grid(True)
plt.tight_layout()
plt.show()
