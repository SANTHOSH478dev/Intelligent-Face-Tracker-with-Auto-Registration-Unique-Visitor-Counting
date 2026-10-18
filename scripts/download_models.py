"""Downloads a YOLOv8-face weight file (with 5 landmarks). InsightFace buffalo_l
auto-downloads on first run. If the URL is stale, grab any 'yolov8n-face.pt'
from the lindevs/yolov8-face or akanametov/yolo-face GitHub releases."""
import os, urllib.request
URL = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
os.makedirs("models", exist_ok=True)
dst = "models/yolov8n-face.pt"
print("Downloading", URL)
urllib.request.urlretrieve(URL, dst)
print("Saved", dst)
