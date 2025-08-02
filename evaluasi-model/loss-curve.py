import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\Videos\model\pose\yolo-pose-70m\results.csv")
df.columns = df.columns.str.strip()

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
plt.plot(df["epoch"], df["train/pose_loss"], label="train/pose_loss")
plt.plot(df["epoch"], df["val/box_loss"], label="val/box_loss")
plt.plot(df["epoch"], df["val/pose_loss"], label="val/pose_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
