import os
from tkinter import Tk, filedialog
from PIL import Image

def flip_images_horizontally(input_dir, output_dir):
    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in supported_ext):
            img_path = os.path.join(input_dir, filename)
            try:
                img = Image.open(img_path)
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                output_path = os.path.join(output_dir, filename)
                flipped_img.save(output_path)
                print(f"Flipped: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    root = Tk()
    root.withdraw()  # Hide main window

    print("Pilih folder input (berisi gambar):")
    input_dir = filedialog.askdirectory(title="Pilih folder input")
    if not input_dir:
        print("Folder input tidak dipilih. Program dihentikan.")
        return

    print("Pilih folder output (tempat hasil flip disimpan):")
    output_dir = filedialog.askdirectory(title="Pilih folder output")
    if not output_dir:
        print("Folder output tidak dipilih. Program dihentikan.")
        return

    flip_images_horizontally(input_dir, output_dir)
    print("Selesai.")

if __name__ == "__main__":
    main()
