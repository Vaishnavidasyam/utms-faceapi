from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import insightface
from PIL import Image
import base64
import io
import pickle

app = Flask(__name__)
CORS(app)

# Initialize model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))

# Folder to store face embeddings
DATA_DIR = "embeddings"
os.makedirs(DATA_DIR, exist_ok=True)

# Function to compute face embedding
def get_embedding(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    faces = model.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

@app.route('/')
def home():
    return jsonify({"message": "Face Recognition API Running"}), 200

# ---------- ENROLL NEW STUDENT ----------
@app.route('/enroll', methods=['POST'])
def enroll():
    name = request.form['name']
    rollnumber = request.form['rollnumber']
    file = request.files['photo']

    embedding = get_embedding(file.read())
    if embedding is None:
        return jsonify({"error": "No face detected"}), 400

    file_path = os.path.join(DATA_DIR, f"{rollnumber}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump({"name": name, "rollnumber": rollnumber, "embedding": embedding}, f)

    return jsonify({"message": f"{name} enrolled successfully"}), 200

# ---------- RECOGNIZE STUDENTS ----------
@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['photo']
    query_embedding = get_embedding(file.read())
    if query_embedding is None:
        return jsonify({"error": "No face detected"}), 400

    recognized_students = []

    for filename in os.listdir(DATA_DIR):
        with open(os.path.join(DATA_DIR, filename), 'rb') as f:
            data = pickle.load(f)
            enrolled_embedding = data['embedding']
            sim = np.dot(query_embedding, enrolled_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(enrolled_embedding)
            )
            if sim > 0.45:  # similarity threshold
                recognized_students.append(data['rollnumber'])

    return jsonify({"recognized": recognized_students}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)