from ultralytics import YOLO
import cv2
from collections import deque
import time
import requests

url = "http://127.0.0.1:5000/"


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


phone_detection_history = deque(maxlen=50)
freeze_history = deque(maxlen=10)
frs_cooldown = False  
cooldown_start_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    cv2.imwrite("test_image.jpg", frame)
    files = {
        "image": open("test_image.jpg", "rb"),
        "phone_detection_history" : phone_detection_history,
        "freez_history" : freez_history,
        "frs_cooldown" : frs_cooldown,
        "cooldown_start_time" : cooldown_start_time
    }
    response = requests.post(url+"recognize", files=files)

    response = response.json()
    person_name = response["name"]
    phone_detection_history = response["phone_detection_history"]
    freez_history = response["freez_history"]
    frs_cooldown =   response["frs_cooldown"]
    cooldown_start_time =  response["cooldown_start_time"]

    cv2.imshow('YOLOv8 Webcam Inference', frame)
    if person_name:
        print(f"Person detected: {person_name}")
        cv2.putText(frame, f"Person: {person_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()