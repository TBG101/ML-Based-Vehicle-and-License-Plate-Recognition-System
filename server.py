from flask import Flask, json, request, jsonify, send_file
import os
import psycopg2
import datetime
from datetime import datetime, timedelta, timezone
import jwt
from dotenv import load_dotenv
from db import get_db_connection
from werkzeug.security import generate_password_hash, check_password_hash
from preload import preload_models, model_ready, get_license_plates, carPredict
import threading

def ensure_user_table_exists(connection: psycopg2.extensions.connection):
    """Ensure the 'user' table exists in the database."""
    if connection is None:
        return

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role VARCHAR(50) DEFAULT 'user'
                );
            """)
            connection.commit()
    except Exception as e:
        print(f"Error ensuring the 'users' table exists: {e}")

def create_token(user_id, username, role="user"):
    payload = {
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "sub": str(user_id),
        "username": username,
        "role": role
    }
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token

def decode_token(token):
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        print(payload)
        return payload
    except jwt.ExpiredSignatureError:
        print("Token has expired.")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid token: {e}")
        return None

def check_auth(auth_header: str):
    print(f"Auth header: {auth_header}")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ")[1]
    payload = decode_token(token)
    print(f"Decoded payload: {payload}")
    if not payload:
        return None
    user_id = payload["sub"]
    return user_id

threading.Thread(target=preload_models, daemon=True).start()
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
load_dotenv()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
secret_key = os.getenv("JWT_SECRET", "default_secret_key")
db = get_db_connection()
ensure_user_table_exists(db)

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/api/v1/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    cur = db.cursor()
    cur.execute(
        "SELECT id, password FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    if user and check_password_hash(user[1], password):
        user_id = user[0]
        token = create_token(user_id, username=username, role="user")
        return jsonify({"message": "Login successful", "token": token}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route("/api/v1/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    # Check if the username already exists in the database
    cur = db.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user = cur.fetchone()

    if user:
        return jsonify({"error": "Username already exists"}), 400

    hashed_password = generate_password_hash(password)

    cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, hashed_password))
    db.commit()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    # Close the cursor
    cur.close()
    if user:
        user_id = user[0]
        token = create_token(user_id, username=username, role="user")
        return jsonify({"message": "Signup successful", "token": token}), 201
    else:
        return jsonify({"error": "Failed to create user"}), 500

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    if not model_ready:
        return jsonify({"error": "Model is still loading"}), 503
    try:
        user_id = check_auth(request.headers.get("Authorization"))
        if not user_id:
            return jsonify({"error": "Unauthorized"}), 401

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        processed_image, all_plates = get_license_plates(image=image_file)
        carType, confidence = carPredict(image_file)
        if processed_image is None:
            return jsonify({"error": "Image processing failed"}), 400

        img_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        if processed_image.size > 0:
            with open(img_path, "wb") as f:
                f.write(processed_image.tobytes())

        return jsonify(
            {
                "number_plate": all_plates,
                "car_type": carType+" " + str(confidence),
                "image_url": f"/uploads/{image_file.filename}",
            }
        )
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/v1/me", methods=["GET"])
def me():
    user_id = check_auth(request.headers.get("Authorization"))
    return jsonify({"user_id": user_id}), 200

@app.route("/api/v1/uploads/<filename>", methods=["GET"])
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
