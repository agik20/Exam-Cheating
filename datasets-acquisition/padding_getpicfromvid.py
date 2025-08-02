import cv2
import os
import numpy as np
from tkinter import Tk, filedialog
from datetime import timedelta

# ========== Fungsi Dialog ==========
def pilih_file_video():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Pilih File Video",
        filetypes=[("Video files", "*.mp4 *.mov *.avi")]
    )

def pilih_folder_output():
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Pilih Folder Output Gambar")

# ========== Resize + Padding ==========
def resize_dan_padding_ke_640x640(frame):
    height, width = frame.shape[:2]
    target_width = 640
    target_height = 640

    scale = target_width / width
    new_width = target_width
    new_height = int(height * scale)

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    pad_vert = target_height - new_height
    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, 0, 0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded

# ========== Cek Nama File Unik ==========
def buat_nama_file_unik(folder, base_name):
    base_path = os.path.join(folder, base_name + ".jpg")
    if not os.path.exists(base_path):
        return base_path

    # Tambah suffix _1, _2, dst
    counter = 1
    while True:
        new_name = f"{base_name}_{counter}.jpg"
        new_path = os.path.join(folder, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

# ========== Proses Ekstraksi ==========
def ekstrak_frame_per_3_detik(video_path, output_folder, interval_detik=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Gagal membuka video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    durasi_video = total_frame / fps
    print(f"Durasi video: {durasi_video:.2f} detik, FPS: {fps}")

    frame_interval = int(fps * interval_detik)
    frame_ke = 0
    saved = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ke)
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = resize_dan_padding_ke_640x640(frame)

        timestamp = str(timedelta(seconds=int(frame_ke / fps))).replace(":", "-")
        base_name = f"frame_{timestamp}"
        output_path = buat_nama_file_unik(output_folder, base_name)

        cv2.imwrite(output_path, output_frame)
        print(f"Disimpan: {os.path.basename(output_path)}")

        frame_ke += frame_interval
        saved += 1

    cap.release()
    print(f"Total frame disimpan: {saved}")

# ========== Eksekusi ==========
video = pilih_file_video()
if not video:
    print("Video tidak dipilih.")
    exit()

folder_output = pilih_folder_output()
if not folder_output:
    print("Folder output tidak dipilih.")
    exit()

ekstrak_frame_per_3_detik(video, folder_output)
