import os
import cv2
import numpy as np
import json
import torch
from flask import Flask, request, jsonify
from insightface.app import FaceAnalysis

app = Flask(__name__)

# Load ArcFace model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

## load YOLO
det_model = YOLO("best.pt")

# Storage for reference faces
REFERENCE_FACES = "reference_faces.json"
if os.path.exists(REFERENCE_FACES):
    with open(REFERENCE_FACES, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

def detect_phone(img):
    results = det_model(img)
    return results

def save_face_db():
    with open(REFERENCE_FACES, "w") as f:
        json.dump(face_db, f)

@app.route("/upload_reference", methods=["POST"])
def upload_reference():
    file = request.files['image']
    name = request.form['name']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    faces = face_app.get(img)
    
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400
    
    face_embedding = faces[0].normed_embedding.tolist()
    face_db[name] = face_embedding
    save_face_db()
    
    return jsonify({"message": "Reference face added successfully"})


def face_recognition(img):
    faces = face_app.get(img)
    
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400
    
    face_embedding = torch.tensor(faces[0].normed_embedding)
    
    best_match = None
    best_score = -1
    
    for name, ref_embedding in face_db.items():
        ref_embedding = torch.tensor(ref_embedding)
        score = torch.nn.functional.cosine_similarity(face_embedding, ref_embedding, dim=0).item()
        
        if score > best_score:
            best_score = score
            best_match = name
    
    if best_score < 0.5:  # Threshold for match
        return {"match": False, "name": None, "score": best_score}
    
    return {"match": True, "name": best_match, "score": best_score}

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files['image']
    phone_detection_history = request.files["phone_detection_history"]
    freez_history = request.files["freez_history"]
    frs_cooldown =   request.files["frs_cooldown"]
    cooldown_start_time =  request.files["cooldown_start_time"]

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = detect_phone(img)
    phone_detected = False
    phone_area_threshold = 0.45 * img.shape[0] * img.shape[1]

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
    person_name = None
    if not majority_phone_detected and not phone_detected and not phone_detection_history[-2] and not frs_cooldown:
        frs_result = face_recognition(img)

        if "name" in frs_cooldown:
            person_name = frs_result["name"]
    
    return jsonify({
        "name" : person_name,
        "phone_detection_history" : phone_detection_history,
        "freez_history" : freez_history,
        "frs_cooldown" : frs_cooldown,
        "cooldown_start_time" : cooldown_start_time
    })

        






if __name__ == "__main__":
    app.run(debug=True)