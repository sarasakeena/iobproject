import dlib
import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
from deepface import DeepFace
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from database import register_user
from src.utils.face_recognition import capture_face_embedding
from database.crud import (
    register_user,
    save_face_embedding
    
)
import face_recognition
import psycopg2
from dotenv import load_dotenv
load_dotenv()
import subprocess
import webbrowser
import threading


def main():
    try:
        # Try to register the user
        try:
            user_id = register_user(
                username='john_doe',
                pin='1234',
                full_name='John Doe'
            )
            print(f"✅ User registered: ID {user_id}")
        except ValueError as ve:
            print(f"⚠️ {ve} - trying to fetch user ID from DB")
            # Assuming you have a method to get user_id by username
            conn = psycopg2.connect("dbname=postgres user=postgres password=sakeena123 host=localhost port=5432")
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s", ("john_doe",))
            result = cursor.fetchone()
            if result:
                user_id = result[0]
                print(f"✅ Existing user found: ID {user_id}")
            else:
                raise ValueError("User exists but cannot be found in DB.")
        
        # Capture and save face embedding
        try:
            face_embedding = capture_face_embedding()
            save_face_embedding(user_id, face_embedding)
        except Exception as e:
            print(f"❌ Error during face embedding capture: {e}")
            return

        # Matching captured embedding with known faces
        known_faces = []
        known_face_ids = []
        for file in os.listdir('known_faces'):
            image = face_recognition.load_image_file(os.path.join('known_faces', file))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces.append(encodings[0])
                known_face_ids.append(file.split('.')[0])

        if not known_faces:
            print("⚠️ No known faces found.")
            return

        face_distances = face_recognition.face_distance(known_faces, face_embedding)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < 0.6:
            print(f"✅ Face matched: {known_face_ids[best_match_index]}")
        else:
            print("❌ No matching face found.")
    except Exception as err:
        print(f"💥 Unexpected error in main(): {err}")


def authenticate_pin(user_id: int, pin: str) -> bool:
    user_pins = {
        1: '1234',
        2: '5678',
    }
    return user_id in user_pins and user_pins[user_id] == pin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "datasett")
PROCESSED_FOLDER = os.path.join(DATASET_ROOT, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
SHAPE_PREDICTOR =  r"C:\iobproject\models\shape_predictor_68_face_landmarks.dat"
liveness_model = load_model(os.path.join(MODELS_DIR, "deepfake_cnn_model.h5"))
ref_image = os.path.join(BASE_DIR, "data", "ref_image.jpg")
DATABASE_URL = psycopg2.connect("dbname=face_auth_db user=postgres password=sakeena123 host=localhost port=5432")



image_path = os.path.join(BASE_DIR, "ref_image.jpg")
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_FOLDER, "train", "real"), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

try:
    face_model = load_model(os.path.join(MODELS_DIR, "deepfake_cnn_model.h5"))
    face_model.summary()
except Exception as load_error:
    print(f"❌ Model loading failed: {str(load_error)}")
    exit(1)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key_123")
def has_webcam():
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    return available

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

@app.route('/')
def index():
    return redirect(url_for('login'))
@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    password = request.form['password']

    print(f"[DEBUG] Received registration for: {name}")

    try:
        # Run face_register.py with the given name and password
        result = subprocess.run(
            [r'C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe', 'data/face_register.py', name, password],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        print("[DEBUG] Subprocess STDOUT:\n", result.stdout)
        print("[DEBUG] Subprocess STDERR:\n", result.stderr)

        if result.returncode == 0:
            session['message'] = "Camera will open in a few seconds, press 'c' to capture your face."
            return redirect(url_for('biometric'))  # Redirect to the biometric page after registration
        else:
            error_message = result.stderr if result.stderr else result.stdout
            flash(error_message, 'error')
            return redirect(url_for('register_page'))

    except Exception as e:
        print(f"[❌ ERROR] Exception occurred: {e}")
        flash(f"Server error: {e}", 'error')
        return redirect(url_for('register_page'))

def open_browser():
    webbrowser.open("http://localhost:5000")


@app.route('/login')
def login():
    return render_template('login.html')
    

@app.route("/detect_landmarks", methods=["POST"])
def detect_landmarks():
    try:
        image_data = request.json.get("image")
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        img_data = base64.b64decode(image_data.split(",")[1])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        if not faces:
            return jsonify({"error": "No faces detected"}), 400

        landmarks = predictor(gray, faces[0])
        return jsonify({
            "landmarks": [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
        })
    except Exception as landmark_error:
        return jsonify({"error": str(landmark_error)}), 500

@app.route('/verify_biometric', methods=['POST'])
def verify_biometric():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        image_data = data.get('image') 

        try:
            # Decode and preprocess image
            img_data = base64.b64decode(image_data.split(',')[1])
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))

            # Liveness prediction
            liveness_score = liveness_model.predict(np.expand_dims(img / 255.0, axis=0))[0][0]
            if liveness_score < 0.5:
                return jsonify({'error': 'Liveness check failed'}), 401

            # Save temporary image for verification
            temp_path = os.path.join(BASE_DIR, "temp_verify.jpg")
            with open(temp_path, 'wb') as f:
                f.write(img_data)

            # Perform face verification
            verification = DeepFace.verify(
                img1_path=ref_image,
                img2_path=temp_path,
                detector_backend="skip",
                enforce_detection=False
            )

            # Clean up temp image
            os.remove(temp_path)

            # Return safe JSON
            return jsonify({
                'verified': bool(verification['verified']),
                'similarity': float(1 - verification['distance']),
                'liveness': float(liveness_score)
            })

        except Exception as processing_error:
            return jsonify({'error': str(processing_error)}), 500

    except Exception as general_error:
        return jsonify({'error': str(general_error)}), 500

@app.route('/select_user_type', methods=['POST'])
def select_user_type():
    user_type = request.form.get('user_type')
    if user_type in ['new', 'existing']:
        session['user_type'] = user_type
        session.pop('pin_registered', None)  # Clear previous registration if any
    return redirect(url_for('pin'))

@app.route('/register_pin', methods=['POST'])
def register_pin():
    if session.get('user_type') != 'new':
        flash('Invalid request', 'error')
        return redirect(url_for('pin'))
    
    pin = request.form.get('pin')
    
    # Validate PIN
    if not pin or not pin.isdigit() or len(pin) < 4 or len(pin) > 6:
        flash('PIN must be 4-6 digits', 'error')
        return redirect(url_for('pin'))
    
    # Store the PIN (in a real app, store hashed version in database)
    session['temp_pin'] = pin
    session['pin_registered'] = True
    flash('PIN registered successfully! Please verify your PIN', 'success')
    return redirect(url_for('pin'))

@app.route('/verify_pin', methods=['POST'])
def verify_pin():
    print("Received request for PIN verification")  # Debugging output
    if 'temp_pin' not in session:
        flash('No PIN registered for verification', 'error')
        return redirect(url_for('pin'))
    
    entered_pin = request.form.get('pin')
    
    if session['temp_pin'] == entered_pin:
        flash('PIN verification successful!', 'success')
        session.pop('temp_pin', None)
        session.pop('pin_registered', None)
        return redirect(url_for('biometric'))
    else:
        flash('Invalid PIN. Please try again.', 'error')
        return redirect(url_for('pin'))
    
@app.route('/biometric')
def biometric():
    return render_template('biometric.html')

# Update your existing pin route
@app.route('/pin')
def pin():
    return render_template('pin.html')


if __name__ == "__main__":
    #main()
    app.run(host="0.0.0.0", port=5000, debug=False)