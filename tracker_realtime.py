import cv2
import numpy as np
import torch
import sys
import os
from math import sqrt

sys.path.insert(0, '/home/factora12/projects/Eco_vision/yolov9')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device


class RadiusVectorTracker:
    def __init__(self, max_distance=50):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = max_distance

    @staticmethod
    def center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def distance(self, box1, box2):
        cx1, cy1 = self.center(box1)
        cx2, cy2 = self.center(box2)
        return sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def update(self, detections):
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = det
                self.next_id += 1
            return dict(zip(range(1, len(detections) + 1), detections))

        result = {}
        used_ids = set()
        for det in detections:
            best_id = None
            best_dist = self.max_distance
            for tid, tbox in self.tracks.items():
                if tid in used_ids:
                    continue
                dist = self.distance(det[:4], tbox[:4])
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            result[best_id] = det
            used_ids.add(best_id)

        self.tracks = result
        return result


def box_center(box):
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean(c1, c2):
    return sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


# ── Device & model ───────────────────────────────────────────────────────────
device  = select_device("cpu")  # use "0" for GPU
weights = '/home/factora12/projects/Eco_vision/yolov9/runs/detect/exp4/best.pt'
model   = DetectMultiBackend(weights, device=device)
model.eval()

COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}
NAMES  = {0: "car",       1: "garbage"}

# Distance (in pixels) at which garbage is considered "near" a car
NEAR_THRESHOLD = 150

tracker = RadiusVectorTracker(max_distance=60)

# ── Video source ─────────────────────────────────────────────────────────────
# 0 = webcam, or specify an RTSP/HTTP stream:
# SOURCE = "rtsp://user:pass@192.168.1.1:554/stream"
SOURCE = 0

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть источник: {SOURCE}")

# Для ускорения реального времени — минимальный буфер
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ── Output folder ────────────────────────────────────────────────────────────
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ── Garbage track state ──────────────────────────────────────────────────────
# garbage_state[gid] = {
#   "first_near_frame": ndarray | None,
#   "was_near":         bool,
#   "screenshot_saved": bool,
# }
garbage_state = {}

print("Running in real-time mode. Press 'q' to quit.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received from source.")
        break
    frame_count += 1

    # ── Inference ─────────────────────────────────────────────────────────────
    img        = cv2.resize(frame, (640, 640))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        pred = model(img_tensor)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = non_max_suppression(pred, 0.25, 0.45)

    detections = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                detections.append([int(xyxy[0]), int(xyxy[1]),
                                   int(xyxy[2]), int(xyxy[3]),
                                   float(conf), int(cls)])

    tracks = tracker.update(detections)

    # ── Separate cars and garbage ─────────────────────────────────────────────
    cars    = {tid: det for tid, det in tracks.items() if det[5] == 0}
    garbage = {tid: det for tid, det in tracks.items() if det[5] == 1}

    # ── Analyse each garbage track ────────────────────────────────────────────
    for gid, gdet in garbage.items():
        gc = box_center(gdet)

        min_dist = min((euclidean(gc, box_center(c)) for c in cars.values()),
                       default=float('inf'))

        is_near = min_dist <= NEAR_THRESHOLD

        if gid not in garbage_state:
            garbage_state[gid] = {
                "first_near_frame": None,
                "was_near":         False,
                "screenshot_saved": False,
            }

        state = garbage_state[gid]

        if is_near:
            # Save the first frame where this garbage was near a car
            if state["first_near_frame"] is None:
                state["first_near_frame"] = frame.copy()
            state["was_near"] = True

        else:
            # Garbage was near but is now moving away — save screenshot
            if state["was_near"] and not state["screenshot_saved"] \
                    and state["first_near_frame"] is not None:
                path = os.path.join(output_dir, f"garbage_{gid}_frame{frame_count}.jpg")
                cv2.imwrite(path, state["first_near_frame"])
                print(f"[Frame {frame_count}] Garbage ID {gid} moved away from car → {path}")
                state["screenshot_saved"] = True

    # ── Draw tracks ───────────────────────────────────────────────────────────
    for tid, det in tracks.items():
        x1, y1, x2, y2, conf, cls = det
        color = COLORS.get(cls, (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"ID:{tid} {NAMES.get(cls, cls)} {conf:.2f}",
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Highlight garbage that is near a car
        if cls == 1:
            state = garbage_state.get(tid, {})
            if state.get("was_near") and not state.get("screenshot_saved"):
                cv2.putText(frame, "NEAR CAR",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow("EcoVision — Real Time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Frames processed: {frame_count}. Screenshots saved in: {output_dir}/")
