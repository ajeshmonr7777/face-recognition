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

# Storage for reference faces
REFERENCE_FACES = "reference_faces.json"
if os.path.exists(REFERENCE_FACES):
    with open(REFERENCE_FACES, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

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

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
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
        return jsonify({"match": False, "name": None, "score": best_score})
    
    return jsonify({"match": True, "name": best_match, "score": best_score})

if __name__ == "__main__":
    app.run(debug=True)