# Head Detection and People Counting System

## Overview
This project implements a real-time head detection and people counting system using YOLOv8 and OpenCV. The system detects and tracks individuals in a video stream, specifically focusing on identifying and counting people exiting a room.

## Features
- **Real-time head detection** using YOLOv8.
- **People tracking** using object ID assignment.
- **Exit counting mechanism** to track people leaving the frame.
- **Live video feed processing** from a webcam.
- **Bounding box drawing** for visual feedback.

## Dependencies
Ensure you have the following dependencies installed:

```bash
pip install opencv-python torch numpy ultralytics
```

## File Structure
- `main.py`: Main script for real-time detection and tracking.
- `models/yolov8n.pt`: Pre-trained YOLOv8 model (must be downloaded separately).

## How It Works
1. **Load YOLOv8 Model:** The script initializes a YOLOv8 model trained for person detection.
2. **Frame Processing:** Each frame from the webcam is processed to detect people and track their movement.
3. **Head Extraction:** Bounding boxes are adjusted to focus on head regions.
4. **Tracking & Counting:** IDs are assigned to individuals, and the system tracks when they exit the frame.
5. **Display Output:** The processed frame is displayed with bounding boxes and exit count.

## Usage
Run the script using:

```bash
python main.py
```

Press `q` to exit the application.

## Notes
- Ensure that `models/yolov8n.pt` is placed in the correct directory.
- Adjust confidence threshold in `model.track(frame, conf=0.6)` if needed.

## Future Improvements
- Enhance tracking accuracy with motion prediction.
- Improve head region extraction logic.
- Implement database logging for long-term analysis.

## License
This project is for educational purposes. Modify and distribute as needed.

