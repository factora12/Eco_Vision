📘 README (English)
EcoVision — Real-Time Car & Garbage Interaction Detection

EcoVision is a real-time computer vision system based on YOLOv9 that detects and tracks cars and garbage, and automatically captures a screenshot when garbage moves away from a car after being close to it.

The system uses:

YOLOv9 for object detection

A custom distance-based tracker (RadiusVectorTracker)

OpenCV for video processing

PyTorch for inference

🚀 Features

Real-time object detection (car & garbage)

Custom object tracking using Euclidean distance

Detection of “garbage near car” events

Automatic screenshot saving when garbage moves away

Works with:

Webcam (SOURCE = 0)

RTSP streams

HTTP streams

🧠 How It Works

The model detects objects in each frame.

Objects are tracked using a radius-based distance tracker.

The system:

Calculates the distance between garbage and cars.

If garbage is within NEAR_THRESHOLD pixels → marked as “near”.

When garbage moves away after being near → a screenshot of the first “near” frame is saved.

All screenshots are saved in the output/ directory.

📂 Project Structure
project/
│
├── your_script.py
├── output/                  # Saved screenshots
└── yolov9/                  # YOLOv9 repository
⚙️ Requirements

Python 3.8+

PyTorch

OpenCV

NumPy

YOLOv9 repository

Install dependencies:

pip install torch torchvision opencv-python numpy

Make sure YOLOv9 is cloned and accessible:

git clone https://github.com/WongKinYiu/yolov9
🖥 Configuration
Select device
device = select_device("cpu")  # use "0" for GPU
Set model weights
weights = 'path/to/best.pt'
Set video source
SOURCE = 0  # webcam
# or
SOURCE = "rtsp://user:pass@ip:port/stream"
Distance threshold
NEAR_THRESHOLD = 150
▶️ Run
python your_script.py

Press q to exit.

📸 Output

Screenshots are saved as:

output/garbage_<ID>_frame<frame_number>.jpg

Each screenshot represents the first moment when garbage was detected near a car before moving away.

🧩 Classes Detected
Class ID	Label
0	car
1	garbage
📌 Notes

For better performance, use GPU (select_device("0")).

CAP_PROP_BUFFERSIZE is set to 1 for real-time responsiveness.

Tracker is distance-based (not Kalman filter or DeepSORT).
