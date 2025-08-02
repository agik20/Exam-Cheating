import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

WATCHED_DIR = os.path.dirname(os.path.abspath(__file__))  # processing-image folder

class FolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            folder_path = event.src_path
            print(f"📁 Folder baru terdeteksi: {folder_path}")

            # Jalankan script pre-processing
            script_path = os.path.join(WATCHED_DIR, "image-pre-processing.py")
            try:
                subprocess.run(["python", script_path, folder_path], check=True)
                print("Pre-processing selesai")
            except subprocess.CalledProcessError as e:
                print("Gagal menjalankan image-pre-processing.py")
                print(e)

if __name__ == "__main__":
    print(f"👁️ Memantau folder: {WATCHED_DIR}")
    event_handler = FolderHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCHED_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()