import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

exited_people = set()
people_exited = 0
tracked_people = {}

# Load YOLOv8 model
model = YOLO("models/yolov8n.pt")

# Dictionary để theo dõi ID của người
tracked_people = defaultdict(int)  # Lưu số frame xuất hiện của mỗi ID
people_exited = 0  # Biến đếm số người ra khỏi khung

def extract_head(box):
    """ Điều chỉnh bounding box thành hình vuông và chỉ lấy phần đầu người. """
    x1, y1, x2, y2 = map(int, box)

    width = x2 - x1
    height = y2 - y1

    # Xác định kích thước hình vuông dựa trên phần đầu
    head_size = max(int(0.5 * width), int(0.35 * height))

    # Tính vị trí trung tâm
    x_center = (x1 + x2) // 2
    y_top = y1
    y_center = y_top + head_size // 2

    # Xác định bounding box hình vuông
    new_x1 = x_center - head_size // 2
    new_x2 = x_center + head_size // 2
    new_y1 = y_top
    new_y2 = y_top + head_size

    return new_x1, new_y1, new_x2, new_y2

def process_frame(frame):
    global people_exited, exited_people, tracked_people
    results = model.track(frame, persist=True, conf=0.6)  # Kích hoạt tracking

    if isinstance(results, list):
        results = results[0]

    current_ids = set()  # Lưu ID của người trong khung hình hiện tại

    if hasattr(results, 'boxes') and results.boxes is not None and results.boxes.id is not None:
        for box, conf, cls, track_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls, results.boxes.id):
            if track_id is None:
                continue  # Bỏ qua nếu không có track_id hợp lệ

            if int(cls) == 0 and conf > 0.6:  # Chỉ nhận diện người
                track_id = int(track_id)

                # Nếu người này đã từng được đếm là rời đi trước đó, thì hủy bỏ việc rời đi
                if track_id in exited_people:
                    people_exited -= 1  # Giảm số người rời đi
                    exited_people.remove(track_id)  # Xóa khỏi danh sách rời đi

                current_ids.add(track_id)
                tracked_people[track_id] = 0  # Reset số frame vắng mặt
                x1, y1, x2, y2 = extract_head(box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ bounding box

    # Kiểm tra ID nào biến mất khỏi khung
    to_delete = []
    for person_id in list(tracked_people.keys()):
        if person_id not in current_ids:
            tracked_people[person_id] += 1
            if tracked_people[person_id] > 10:  # Nếu vắng mặt trên 10 frame
                if person_id not in exited_people:  # Đảm bảo không đếm trùng lặp
                    people_exited += 1  # Đếm số người rời đi
                    exited_people.add(person_id)  # Thêm vào danh sách đã rời đi
                to_delete.append(person_id)

    # Xóa ID đã ra khỏi khung hình để tránh nhớ sai ID
    for person_id in to_delete:
        del tracked_people[person_id]

    # Hiển thị số người trong khung và số người đã rời đi
    cv2.putText(frame, f'so nguoi trong phong: {len(current_ids)}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'so nguoi da roi di: {people_exited}', (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame

cap = cv2.VideoCapture(0)  # Use webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = process_frame(frame)
    cv2.imshow("Head Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()