from ultralytics import YOLO
import cv2
from collections import deque
import time
import requests

# Load the YOLO model
model = YOLO("best.pt")  # Replace with your trained model file

model2 = YOLO("fake_detection.pt")  # Replace with your trained model file

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


phone_detection_history = deque(maxlen=25)
fake_detection_history = deque(maxlen=25)
frs_cooldown = False  # Flag to control the cooldown period
cooldown_start_time = 0  # Variable to store the start time of the cooldown

# Check if a phone is detected in the frame
fake_face_detected = False
phone_detected = False
majority_phone_detected = False
majority_fake_detected = False
person_name = None
frame_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    frame_count+=1

    # Perform inference on the frame
    if frame_count%3 == 0:

        results = model(frame)  # Pass the frame to the YOLO model

        # Check if a phone is detected in the frame
        phone_detected = False

        for result in results:
            # Check if any detected object is a "phone" (assuming class ID for phone is known)
            for box in result.boxes:
                # print(box)
                class_id = int(box.cls)  # Get the class ID of the detected object
                conf = box.conf
                if class_id == 0 and conf > 0.3:  # Replace 67 with the class ID for "phone" in your model
                    phone_detected = True
                    break

        phone_detection_history.append(phone_detected)

        majority_phone_detected = max(set(phone_detection_history), key=phone_detection_history.count)

        # Perform inference on the frame
    if frame_count%3 == 1:

        results = model2(frame)  # Pass the frame to the YOLO model

        # Check if a phone is detected in the frame
        fake_face_detected = False

        for result in results:
            # Check if any detected object is a "phone" (assuming class ID for phone is known)
            for box in result.boxes:
                # print(box)
                class_id = int(box.cls)  # Get the class ID of the detected object
                conf = box.conf
                if class_id == 1 and conf > 0.5:  # Replace 67 with the class ID for "phone" in your model
                    fake_face_detected = True
                    break

        fake_detection_history.append(fake_face_detected)

        majority_fake_detected = max(set(fake_detection_history), key=fake_detection_history.count)


    if len(phone_detection_history) < 2:
        phone_detection_history.append(False)
        phone_detection_history.append(False)
    if len(fake_detection_history) < 2:
        fake_detection_history.append(False)
        fake_detection_history.append(False)
    # If no phone is detected, pass the frame to the frs function
    condition1 = (not majority_phone_detected and not phone_detected and not phone_detection_history[-2]) 
    condition2 = (not majority_fake_detected and not fake_face_detected and not fake_detection_history[-2])

    print('Phone Detected : ',phone_detected)
    print('Majority Detected : ',majority_phone_detected)

    if condition1 and condition2 and frame_count%3 == 2:
        person_name = frs(frame)  # Call the face recognition system
        if person_name:
            print(f"Person detected: {person_name}")
            # Optionally, display the name on the frame
            

    if condition1 and condition2:
        if person_name:
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