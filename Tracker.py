import os
import time
import threading
import datetime
import json
import numpy as np
from PIL import ImageGrab, Image, ImageChops, ImageStat
import pystray
from pystray import MenuItem as item
from sentence_transformers import SentenceTransformer
import torch
import gc
import asyncio
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.graphics.imaging import BitmapDecoder
from winrt.windows.storage import StorageFile
from pynput import keyboard, mouse
import ctypes
from ctypes import wintypes
import psutil

# --- Configuration ---
SAVE_DIR = r"Z:\\TrackerData"
INTERVAL = 5
MAX_rez = 1080 # Still used for resizing before embedding (faster processing)
DIFF_THRESHOLD = 2 
IMAGE_MODEL_NAME = 'clip-ViT-B-32'
TEXT_MODEL_NAME = 'BAAI/bge-small-en-v1.5'

class RecallRecorder:
    def __init__(self):
        self.running = False
        self.icon = None
        self.thread = None
        self.last_image_thumb = None 
        self.models = {
            'image': None,
            'text': None,
        }
        self.input_counts = {'keys': 0, 'mouse': 0}
        self.key_listener = None
        self.mouse_listener = None
    
    def get_active_window_info(self):
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
            window_title = buff.value
            
            pid = wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process = psutil.Process(pid.value)
            process_name = process.name()
            return window_title, process_name
        except Exception:
            return None, None

    def run_windows_ocr(self, image_path):
        async def _ocr_task():
            abs_path = os.path.abspath(image_path)
            file = await StorageFile.get_file_from_path_async(abs_path)
            stream = await file.open_async(0) 
            decoder = await BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            engine = OcrEngine.try_create_from_user_profile_languages()
            result = await engine.recognize_async(bitmap)
            return result.text

        try:
            return asyncio.run(_ocr_task())
        except Exception as e:
            print(f"Windows OCR Error: {e}")
            return ""

    def _on_press(self, key):
        self.input_counts['keys'] += 1

    def _on_click(self, x, y, button, pressed):
        if pressed: 
            self.input_counts['mouse'] += 1

    def _on_scroll(self, x, y, dx, dy):
        self.input_counts['mouse'] += 1

    def should_save(self, current_image):
        current_thumb = current_image.resize((50, 50), Image.Resampling.NEAREST).convert("L")
        if self.last_image_thumb is None:
            self.last_image_thumb = current_thumb
            return True
        diff = ImageChops.difference(current_thumb, self.last_image_thumb)
        stat = ImageStat.Stat(diff)
        avg_diff = sum(stat.mean) / len(stat.mean)
        return avg_diff > DIFF_THRESHOLD

    def take_screenshot(self):
        try:
            # 1. Capture RAW (Full Resolution)
            raw_screenshot = ImageGrab.grab()
            
            if not self.should_save(raw_screenshot):
                return

            # --- 2. AI Processing ---
            
            # A. Extract Text (Windows Native OCR)
            # We still need a temp file for Windows OCR API, but we delete it immediately.
            temp_ocr_path = os.path.join(SAVE_DIR, "temp_ocr_buffer.png")
            raw_screenshot.save(temp_ocr_path)
            
            full_text = self.run_windows_ocr(temp_ocr_path)
            
            try:
                os.remove(temp_ocr_path)
            except Exception:
                pass

            # B. Generate Embeddings
            # We resize for the model speed, but we DON'T save the image to disk
            if raw_screenshot.width > MAX_rez:
                ratio = MAX_rez / float(raw_screenshot.width)
                h = int(float(raw_screenshot.height) * float(ratio))
                small_img = raw_screenshot.resize((MAX_rez, h), Image.Resampling.LANCZOS)
            else:
                small_img = raw_screenshot

            if self.models['image']:
                image_embedding = self.models['image'].encode(small_img, convert_to_tensor=True)
            else:
                image_embedding = torch.tensor([])
            
            if full_text.strip():
                if self.models['text']:
                    text_embedding = self.models['text'].encode(full_text, convert_to_tensor=True)
                    text_embedding_list = text_embedding.tolist()
                else:
                    text_embedding_list = []
            else:
                text_embedding_list = []

            # --- 3. Save Metadata Only (Appended to JSONL) ---
            
            now = datetime.datetime.now()
            folder_path = os.path.join(SAVE_DIR, str(now.year), f"{now.month:02d}", f"{now.day:02d}")
            os.makedirs(folder_path, exist_ok=True)
            
            # JSON Lines file for the day
            daily_file_path = os.path.join(folder_path, "data.jsonl")

            keys_pressed = self.input_counts['keys']
            mouse_activity = self.input_counts['mouse']
            self.input_counts['keys'] = 0
            self.input_counts['mouse'] = 0

            active_window, process_name = self.get_active_window_info()

            meta_payload = {
                "timestamp": now.isoformat(),
                # "image_path": REMOVED for privacy/space
                "window_title": active_window,
                "app_name": process_name,
                "keys_pressed": keys_pressed,
                "mouse_activity": mouse_activity,
                "image_embedding": image_embedding.tolist(),
                "text_embedding": text_embedding_list,
                # We do NOT save 'full_text' here, maintaining privacy
            }

            # Append Mode ('a') is efficient and safe
            with open(daily_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(meta_payload) + "\n")

        except Exception as e:
            print(f"Critical Error in take_screenshot: {e}")

    def loop(self):
        while self.running:
            start_time = time.time()
            self.take_screenshot()
            elapsed = time.time() - start_time
            time.sleep(max(0, INTERVAL - elapsed))

    def start_recording(self, icon, item):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Models on {device}...")
        
        self.models['image'] = SentenceTransformer(IMAGE_MODEL_NAME, device=device)
        self.models['text'] = SentenceTransformer(TEXT_MODEL_NAME, device=device)
        
        print("Using Windows Native OCR (Zero VRAM)...")

        self.key_listener = keyboard.Listener(on_press=self._on_press)
        self.key_listener.start()
        
        self.mouse_listener = mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll)
        self.mouse_listener.start()
        
        print("All Models Loaded. Starting Loop.")
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.icon.update_menu()

    def stop_recording(self, icon, item):
        print("Stopping...")
        self.running = False

        if self.key_listener:
            self.key_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
        
        self.models = {'image': None, 'text': None}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Stopped and VRAM cleared.")
        self.icon.update_menu()

    def open_folder(self, icon, item):
        os.startfile(SAVE_DIR)

    def quit_app(self, icon, item):
        self.running = False
        if self.key_listener:
            self.key_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
        self.models = {'image': None, 'text': None}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        icon.stop()

    def run(self):
        if not os.path.exists("icon.ico"):
            img = Image.new('RGB', (64, 64), color = (73, 109, 137))
            img.save('icon.ico')

        image = Image.open("icon.ico")
        menu = (
            item('Start Recording', self.start_recording, visible=lambda item: not self.running),
            item('Stop Recording', self.stop_recording, visible=lambda item: self.running),
            item('Open Folder', self.open_folder),
            item('Quit', self.quit_app)
        )
        self.icon = pystray.Icon("Tracker", image, "Tracker", menu)
        self.running = False
        self.icon.run()

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    app = RecallRecorder()
    app.run()