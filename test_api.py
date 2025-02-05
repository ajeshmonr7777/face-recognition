import cv2
import requests

# Capture an image from the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image")
    exit()

cv2.imwrite("test_image.jpg", frame)

# Send to the recognition API
url = "http://127.0.0.1:5000/recognize"
files = {"image": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)

print("Recognition Result:", response.json())
