from ultralytics import YOLO
import cv2
from collections import deque
import time
import requests

# Load the YOLO model
model = YOLO("best.pt")  # Replace with your trained model file

# Initialize the webcam
cap = cv2.VideoCapture(1)  # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define your face recognition system (frs) function
def frs(frame):
    """
    Process the frame for face recognition.
    Args:
        frame: The input frame from the webcam.
    Returns:
        name: The name of the person detected (or None if no face is detected).
    """
    cv2.imwrite("test_image.jpg", frame)
    url = "http://127.0.0.1:5000/recognize"
    files = {"image": open("test_image.jpg", "rb")}
    response = requests.post(url, files=files)
    print("Recognition Result:", response.json())
    json_obj = response.json()
    if "name" in json_obj:
        return json_obj["name"]
    else:
        return None


phone_detection_history = deque(maxlen=50)
freeze_history = deque(maxlen=10)
frs_cooldown = False  # Flag to control the cooldown period
cooldown_start_time = 0  # Variable to store the start time of the cooldown
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform inference on the frame
    results = model(frame)  # Pass the frame to the YOLO model

    # Check if a phone is detected in the frame
    phone_detected = False
    phone_area_threshold = 0.45 * frame.shape[0] * frame.shape[1]

    for result in results:
        # Check if any detected object is a "phone" (assuming class ID for phone is known)
        for box in result.boxes:
            # print(box)
            class_id = int(box.cls)  # Get the class ID of the detected object
            conf = box.conf
            if class_id == 0 and conf > 0.3:  # Replace 67 with the class ID for "phone" in your model
                phone_detected = True
                # Calculate the area of the detected phone
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                phone_area = (x2 - x1) * (y2 - y1)
                if phone_area >= phone_area_threshold:
                    # If the phone area is 20% or more of the frame, start the cooldown
                    freeze_history.append(True)
                    majority_freeze_history = max(set(freeze_history), key=freeze_history.count)
                    if majority_freeze_history and len(freeze_history) == 10:
                       frs_cooldown = True
                       cooldown_start_time = time.time()
                else:
                    freeze_history.append(False)
                break

    phone_detection_history.append(phone_detected)

    majority_phone_detected = max(set(phone_detection_history), key=phone_detection_history.count)

    if frs_cooldown and (time.time() - cooldown_start_time) >= 20:
        frs_cooldown = False  # Reset the cooldown flag after 20 seconds

    if len(phone_detection_history) < 2:
        phone_detection_history.append(False)
    # If no phone is detected, pass the frame to the frs function
    if not majority_phone_detected and not phone_detected and not phone_detection_history[-2] and not frs_cooldown:
        person_name = frs(frame)  # Call the face recognition system
        if person_name:
            print(f"Person detected: {person_name}")
            # Optionally, display the name on the frame
            cv2.putText(frame, f"Person: {person_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with annotations
    for result in results:
        # frame = result.plot()  # Annotate the frame with YOLO detections
        cv2.imshow('YOLOv8 Webcam Inference', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()