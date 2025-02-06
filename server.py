import os
import cv2
import numpy as np
import json
import torch
from ultralytics import YOLO
from collections import deque
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
    # Get the image file
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Deserialize JSON data from request
    payload = json.loads(request.form["json"])

    phone_detection_history = deque(payload["phone_detection_history"], maxlen=50)
    freeze_history = deque(payload["freeze_history"], maxlen=10)
    frs_cooldown = payload["frs_cooldown"]
    cooldown_start_time = payload["cooldown_start_time"]

    # Perform phone detection
    results = detect_phone(img)
    phone_detected = False
    phone_area_threshold = 0.45 * img.shape[0] * img.shape[1]

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)  
            conf = box.conf
            if class_id == 0 and conf > 0.3:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                phone_area = (x2 - x1) * (y2 - y1)

                if phone_area >= phone_area_threshold:
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

    # Reset cooldown after 20 seconds
    if frs_cooldown and (time.time() - cooldown_start_time) >= 20:
        frs_cooldown = False

    if len(phone_detection_history) < 2:
        phone_detection_history.append(False)

    # Perform face recognition if no phone is detected and no cooldown
    person_name = None
    if not majority_phone_detected and not phone_detected and not phone_detection_history[-2] and not frs_cooldown:
        frs_result = face_recognition(img)
        if "name" in frs_result:
            person_name = frs_result["name"]

    # Return the updated state as JSON
    return jsonify({
        "name": person_name,
        "phone_detection_history": list(phone_detection_history),  # Convert deque to list
        "freeze_history": list(freeze_history),  # Convert deque to list
        "frs_cooldown": frs_cooldown,
        "cooldown_start_time": cooldown_start_time
    })


        






if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)