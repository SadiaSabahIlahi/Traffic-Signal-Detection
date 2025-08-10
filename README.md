🚦 Traffic Signal Detection Web App
📌 Overview
This project is a real-time traffic signal detection system built using YOLO (You Only Look Once) object detection and Flask for web streaming.
It processes a video input (input.mp4) and detects traffic signals such as red, yellow, and green lights using a pre-trained YOLO model.
The detection results are streamed live on a web interface.

✨ Features
YOLO-based Object Detection for traffic signals.
Flask Web Interface to view real-time detection output.
Start & Stop Control for video streaming.
Non-Maximum Suppression (NMS) to filter weak detections.
Resized frames for faster processing.

🛠 Technologies Used
Python
OpenCV – for video processing & drawing detection boxes.
NumPy – for array operations.
Flask – for serving the detection results on a web page.
YOLO (with model.cfg, model.weights, and sign.names).

📂 Project Structure
fina_traffic.py       # Main application code
index.html            # Web interface template
model.cfg             # YOLO model configuration
model.weights         # YOLO trained weights (stored via Git LFS)
sign.names            # Class names for YOLO
input.mp4             # Input video for detection (stored via Git LFS)
inp.jpg               # Sample image

🚀 How to Run
1️⃣ Install dependencies
pip install flask opencv-python numpy

2️⃣ Ensure YOLO files are present
Place model.cfg, model.weights, and sign.names in the project root folder.

3️⃣ Run the Flask app
python fina_traffic.py

4️⃣ View the detection
Open your browser and go to url

📌 Controls
/start → Start video streaming with detection.
/stop → Stop the video stream.

⚠ Notes
Large files (input.mp4 and model.weights) are stored using Git LFS. Make sure to install Git LFS before cloning:
git lfs install

