import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\Videos\model\pose\yolo-pose-70m\results.csv")
df.columns = df.columns.str.strip()

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5 (Box)")
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95 (Box)")
plt.plot(df["epoch"], df["metrics/mAP50(P)"], label="mAP@0.5 (Pose)")
plt.plot(df["epoch"], df["metrics/mAP50-95(P)"], label="mAP@0.5:0.95 (Pose)")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("mAP Score Progression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
