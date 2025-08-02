import numpy as np
import cv2
import os
import sys
import time
from PIL import Image
from tkinter import Tk, filedialog

# --- 1. Pilih folder input dan output lewat file explorer ---
Tk().withdraw()

print("\U0001F4C2 Silakan pilih folder INPUT gambar:")
input_folder = filedialog.askdirectory(title="Pilih Folder INPUT Gambar")

print("\U0001F4BE Silakan pilih folder OUTPUT untuk menyimpan hasil:")
output_folder = filedialog.askdirectory(title="Pilih Folder OUTPUT Gambar")

if not input_folder or not output_folder:
    print("❌ Folder input atau output tidak dipilih.")
    sys.exit(1)

# --- 2. Fungsi bantu ---
def resize_dengan_padding_ke_640(pil_img):
    img = np.array(pil_img)
    h, w = img.shape[:2]
    skala = 640 / max(h, w)
    new_w, new_h = int(w * skala), int(h * skala)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    x_offset = (640 - new_w) // 2
    y_offset = (640 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas.astype(np.float32) / 255.0  # return float32 [0..1] image

def gaussian(x2, sigma):
    return np.exp(-x2 / (2 * sigma**2))

def bilateral_filter(image, sigma_space, sigma_intensity):
    kernel_size = int(2 * sigma_space + 1)
    half = kernel_size // 2
    result = np.zeros_like(image, dtype=np.float32)
    W = np.zeros_like(image, dtype=np.float32)

    for x in range(-half, half + 1):
        for y in range(-half, half + 1):
            Gspace = gaussian(x**2 + y**2, sigma_space)
            shifted = np.roll(image, [x, y], axis=(0, 1))
            Gintensity = gaussian((image - shifted)**2, sigma_intensity)
            result += Gspace * Gintensity * shifted
            W += Gspace * Gintensity

    return (result / (W + 1e-8)).astype(np.float32)

def unsharp_mask(image, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 1)

# --- 3. Siapkan folder output ---
os.makedirs(output_folder, exist_ok=True)

# --- 4. Proses semua gambar ---
start_time = time.time()
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total_files = len(image_files)
total_processed = 0

for idx, filename in enumerate(image_files):
    image_path = os.path.join(input_folder, filename)

    try:
        pil_img = Image.open(image_path).convert('RGB')
        input_image = resize_dengan_padding_ke_640(pil_img)

        # === Enhanced Processing ===
        filtered = bilateral_filter(input_image, sigma_space=5, sigma_intensity=0.05)
        enhanced = unsharp_mask(filtered, amount=1.2)

        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR))
        total_processed += 1

        # --- Tampilan Progress Bar ---
        percent = int((idx + 1) / total_files * 100)
        bar = ('#' * (percent // 2)).ljust(50)
        print(f"\rProgress: |{bar}| {percent}% ({idx + 1}/{total_files})", end='', flush=True)

    except Exception as e:
        print(f"\nError saat memproses {filename}: {e}")

# --- Statistik ---
elapsed = time.time() - start_time
print(f"\n\nSemua gambar selesai diproses.")
print(f"\U0001F4CA Total gambar  : {total_processed}")
print(f"Waktu total   : {elapsed:.2f} detik")
if total_processed > 0:
    print(f"Rata-rata per gambar: {elapsed / total_processed:.2f} detik")
