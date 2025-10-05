from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import os
from PIL import Image
import io
import pickle
import time

app = Flask(__name__)
CORS(app)

DATA_DIR = "embeddings"
os.makedirs(DATA_DIR, exist_ok=True)

def get_embedding(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    boxes = face_recognition.face_locations(arr)
    if not boxes:
        return None
    encodings = face_recognition.face_encodings(arr, boxes)
    if not encodings:
        return None
    return encodings[0]

@app.route('/')
def home():
    return jsonify({"message": "Face Recognition API Running"}), 200

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
            if sim > 0.45:
                recognized_students.append(data['rollnumber'])

    return jsonify({"recognized": recognized_students}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
