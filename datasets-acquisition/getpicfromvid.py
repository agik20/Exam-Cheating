import cv2
import os
from tkinter import Tk, filedialog, simpledialog
from datetime import timedelta

# Buka dialog untuk memilih file video
def pilih_file_video():
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Pilih File Video",
        filetypes=[("Video files", "*.mp4 *.mov *.avi")]
    )
    return filepath

# Buka dialog untuk memilih folder output
def pilih_folder_output():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Pilih Folder Output Gambar")
    return folder

def ekstrak_frame_per_3_detik(video_path, output_folder, interval_detik=1):
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

        # Simpan frame asli (tanpa resize)
        timestamp = str(timedelta(seconds=int(frame_ke / fps))).replace(":", "-")
        nama_file = f"frame_{timestamp}.jpg"
        output_path = os.path.join(output_folder, nama_file)
        cv2.imwrite(output_path, frame)
        print(f"Disimpan: {nama_file}")

        frame_ke += frame_interval
        saved += 1

    cap.release()
    print(f"Total frame disimpan: {saved}")

# === Eksekusi utama ===
video = pilih_file_video()
if not video:
    print("Video tidak dipilih.")
    exit()

folder_output = pilih_folder_output()
if not folder_output:
    print("Folder output tidak dipilih.")
    exit()

# Langsung ekstrak tanpa minta ukuran
ekstrak_frame_per_3_detik(video, folder_output)
