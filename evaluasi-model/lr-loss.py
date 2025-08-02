import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\Videos\model\pose\yolo-pose-70m\results.csv")
df.columns = df.columns.str.strip()

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Gunakan kolom yang benar
ax1.plot(df["epoch"], df["val/pose_loss"], 'g-', label="val_pose_loss")
ax2.plot(df["epoch"], df["lr/pg0"], 'b--', label="lr_pg0")

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Pose Loss', color='g')
ax2.set_ylabel('Learning Rate (pg0)', color='b')
plt.title("Validation Pose Loss vs Learning Rate")
fig.tight_layout()
plt.show()
