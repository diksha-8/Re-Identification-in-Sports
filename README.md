# Re-Identification-in-Sports
# Player Tracking using YOLOv11 and DeepSORT

This project performs player tracking in a football match video using **YOLOv11 for detection** and **DeepSORT for tracking**. Each player is assigned a **unique ID** that is maintained as long as they stay in the frame.


## Project Structure
├── Intern_Assignment.py # Main Python script
├── best.pt # YOLOv11 trained model file
├── 15sec_input_720p.mp4 # Input video
├── output_reid.mp4 # Output video with player tracking
└── README.md # Project documentation (this file)

---

##  How to Run the Project

### 1. Install Required Libraries

Use the following commands to install the required Python packages:

pip install ultralytics
pip install opencv-python
pip install deep_sort_realtime
Make sure best.pt and 15sec_input_720p.mp4 are in the same directory as the script.

Then run:

python Assignment.py
This will:
Detect players in the video using your YOLO model
Track players with consistent IDs using DeepSORT
Save the output video as output_reid.mp4

 Features
 YOLOv11 detection using a trained model (best.pt)
 DeepSORT-based player tracking
 Persistent unique ID assigned to each player (e.g., ID: 4)
 HD 1280×720 output resolution
