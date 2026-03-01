# tracker_video_file.py

import cv2
import numpy as np
import torch
import sys
import os
sys.path.insert(0, '/home/factora12/projects/Eco_vision/yolov9')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Simple IoU-based tracker
class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def update(self, detections):
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = det
                self.next_id += 1
            return {i: d for i, d in zip(range(1, len(detections)+1), detections)}

        result = {}
        used_ids = set()
        for det in detections:
            best_id, best_iou = None, 0.3
            for tid, tbox in self.tracks.items():
                if tid in used_ids:
                    continue
                iou_score = self.iou(det[:4], tbox[:4])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_id = tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            result[best_id] = det
            used_ids.add(best_id)
        self.tracks = result
        return result

# Device and model
device = select_device("cpu")  # or "0" for GPU
weights = '/home/factora12/projects/Eco_vision/yolov9/runs/detect/exp4/best.pt'
model = DetectMultiBackend(weights, device=device)
model.eval()

# Colors and class names
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}
NAMES = {0: "car", 1: "garbage"}

tracker = SimpleTracker()

# Input video path
video_input_path = "/home/factora12/projects/Eco_vision/video_2026-03-01_18-51-42.mp4"
cap = cv2.VideoCapture(video_input_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_input_path}")

# Video writer
output_dir = "tracked_video_file"
os.makedirs(output_dir, exist_ok=True)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_output_path = os.path.join(output_dir, "tracked_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and convert to tensor
        img = cv2.resize(frame, (640, 640))
        img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(device)/255.0

        # Model inference
        with torch.no_grad():
            pred = model(img_tensor)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = non_max_suppression(pred, 0.25, 0.45)

        # Process detections
        detections = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]),
                                       float(conf), int(cls)])

        # Update tracker
        tracks = tracker.update(detections)

        # Draw tracks
        for tid, det in tracks.items():
            x1, y1, x2, y2, conf, cls = det
            color = COLORS.get(cls, (255,255,0))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID:{tid} {NAMES.get(cls,cls)} {conf:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Write frame to video
        out.write(frame)
        frame_count += 1

except KeyboardInterrupt:
    print("Processing stopped by user.")

cap.release()
out.release()
print(f"Processed {frame_count} frames. Output video saved at {video_output_path}")/home/factora12/projects/Eco_vision/video_2026-03-01_18-51-42.mp4