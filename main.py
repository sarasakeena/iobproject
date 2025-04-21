import dlib
import bcrypt
import os
import psycopg2
import re
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
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, redirect, url_for
import time
from datetime import datetime, timedelta
from mtcnn import MTCNN  # Added MTCNN import

# Initialize MTCNN detector
mtcnn_detector = MTCNN()

# Track attempts and block suspicious activity
attempts = {}
BLOCK_TIME = 600  # 10 minutes for blocking
COOLDOWN = 120     # 2 minutes after 3 failed attempts

def alert_io_team(user_ip, similarity):
    # Implement email/API alert (e.g., Slack/WhatsApp)
    print(f"ALERT: Suspicious activity from IP {user_ip}. Similarity: {similarity}%")

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
SHAPE_PREDICTOR = r"C:\iobproject\models\shape_predictor_68_face_landmarks.dat"
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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "ba38facd1e2a59cc22318b0712c14597")
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)

def has_webcam():
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    return available

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                 max_num_faces=1, 
                                 min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

@app.route('/')
def transaction_page():
    return render_template('transaction_page.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
@app.route('/proceed')
def proceed():
    return render_template('proceed.html')

@app.route('/biometric', methods=['GET', 'POST'])
def biometric():
    if request.method == 'POST':
        user_ip = request.remote_addr
        current_time = time.time()

        # Check if user is blocked
        if user_ip in attempts and attempts[user_ip]['block_until'] > current_time:
            return jsonify({
                "status": "blocked",
                "message": "Transaction blocked. Try again after 10 minutes.",
                "unlock_time": attempts[user_ip]['block_until']
            })

        # Simulate verification (replace with your actual logic)
        similarity = float(request.form.get('similarity', 0))
        threshold = 0.5  # Adjust as needed

        if similarity >= threshold:
            # Successful: Redirect to transaction page
            return jsonify({
                "status": "success",
                "message": "Verification successful!",
                "redirect": url_for('proceed')
            })
        else:
            # Failed: Update attempts
            if user_ip not in attempts:
                attempts[user_ip] = {'count': 0, 'last_attempt': 0, 'block_until': 0}

            attempts[user_ip]['count'] += 1
            attempts[user_ip]['last_attempt'] = current_time

            # Block after 3 failures
            if attempts[user_ip]['count'] >= 3:
                attempts[user_ip]['block_until'] = current_time + BLOCK_TIME
                # Alert IO team (e.g., send email/log)
                alert_io_team(user_ip, similarity)
                return jsonify({
                    "status": "blocked",
                    "message": "Transaction blocked. Try again after 10 minutes.",
                    "unlock_time": attempts[user_ip]['block_until']
                })
            else:
                # Suggest retry with hints
                hints = ["Ensure proper lighting.", "Position your face clearly.", "Remove obstructions (glasses/mask)."]
                return jsonify({
                    "status": "failed",
                    "message": f"Verification failed! Attempts left: {3 - attempts[user_ip]['count']}",
                    "hints": hints
                })

    return render_template('biometric.html')

@app.route('/login')
def login():
    return render_template('login.html')

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

@app.route("/detect_landmarks", methods=["POST"])
def detect_landmarks():
    try:
        image_data = request.json.get("image")
        if not image_data:
            return jsonify({"error": "No image provided"}), 200

        img_data = base64.b64decode(image_data.split(",")[1])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run MediaPipe FaceMesh
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return jsonify({"error": "No face landmarks detected"}), 200

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        height, width, _ = rgb_image.shape

        for lm in face_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            landmarks.append((x, y))

        return jsonify({
            "landmarks": landmarks
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
                return jsonify({ 
                    'error': 'We couldn\'t verify this was a live person. ' + 
                            'Please ensure: \n' +
                            '1. Good lighting on your face\n' +
                            '2. No photos/screens being used\n' +
                            '3. You\'re looking directly at the camera'
                            }), 401

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
    
    # Hash the PIN using bcrypt
    hashed_pin = bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt())
    
    # Store the hashed PIN (in a real app, store this in a database, not in the session)
    session['hashed_pin'] = hashed_pin  # Store the hashed PIN
    
    # Remove plain-text PIN from the session (it's now unnecessary and should not be stored)
    session.pop('temp_pin', None)
    
    # Indicate that PIN is registered
    session['pin_registered'] = True
    flash('PIN registered successfully! Please verify your PIN', 'success')
    
    return redirect(url_for('pin'))


@app.route('/check_pin', methods=['GET'])
def check_pin():
    # Check if the hashed PIN is stored in the session
    if 'hashed_pin' in session:
        hashed_pin = session['hashed_pin']
        
        # Check if the stored hashed PIN matches the bcrypt format (60 characters long)
        if re.match(r'^\$2[ab]\$\d{2}\$[A-Za-z0-9./]{53}$', hashed_pin):
            flash('PIN is stored in a valid hashed form.', 'success')
        else:
            flash('Stored value is not a valid hash.', 'error')
    else:
        flash('No hashed PIN found in session.', 'error')
    
    return redirect(url_for('pin'))


@app.route('/verify_pin', methods=['POST'])
def verify_pin():
    if 'hashed_pin' not in session:
        flash('No PIN registered for verification', 'error')
        return redirect(url_for('pin'))
    
    entered_pin = request.form.get('pin')
    
    # Check if the entered PIN matches the stored hashed PIN
    if bcrypt.checkpw(entered_pin.encode('utf-8'), session['hashed_pin']):
        flash('PIN verification successful!', 'success')
        session.pop('hashed_pin', None)  # Remove the hashed PIN after verification
        session.pop('pin_registered', None)  # Clear the PIN registration status
        return redirect(url_for('biometric'))
    else:
        flash('Invalid PIN, please try again', 'error')
        return redirect(url_for('pin'))@app.route('/verify_pin', methods=['POST'])

@app.route('/pin')
def pin():
    return render_template('pin.html')

# New MTCNN endpoint for face detection
@app.route('/detect_faces_mtcnn', methods=['POST'])
def detect_faces_mtcnn():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read image file
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Convert to RGB (MTCNN expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces with MTCNN
        faces = mtcnn_detector.detect_faces(img_rgb)
        
        # Prepare response
        response = {
            'num_faces': len(faces),
            'faces': []
        }
        
        # Draw bounding boxes if requested
        if request.args.get('draw_boxes', 'false').lower() == 'true':
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Convert back to bytes for response
            _, img_encoded = cv2.imencode('.jpg', img)
            response['image_with_boxes'] = base64.b64encode(img_encoded).decode('utf-8')
        
        # Add face details to response
        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            keypoints = face['keypoints']
            
            response['faces'].append({
                'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h},
                'confidence': confidence,
                'keypoints': keypoints
            })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # fallback to 5000 for local dev
    app.run(host='0.0.0.0', port=port)