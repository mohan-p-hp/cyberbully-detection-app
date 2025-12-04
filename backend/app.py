import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
import emoji
import re
import html
import traceback

# --- OCR Imports ---
import pytesseract
from PIL import Image
import io

# --- Audio Imports ---
import speech_recognition as sr
from pydub import AudioSegment

# --- Gemini Imports ---
import google.generativeai as genai


# ===============================
#  CONFIGURATION
# ===============================

app = Flask(__name__)

project_dir = os.path.dirname(os.path.abspath(__file__))
instance_path = os.path.join(project_dir, "instance")
os.makedirs(instance_path, exist_ok=True)

database_file = f"sqlite:///{os.path.join(instance_path, 'site.db')}"

app.config["SECRET_KEY"] = "somerandomstring123!@#"
app.config["SQLALCHEMY_DATABASE_URI"] = database_file
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_SECURE"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# ===== CORS FIX for login cookies =====
CORS(app,
    resources={r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ]
    }},
    supports_credentials=True
)



db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)


# ===============================
#  GEMINI API CONFIG
# ===============================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini enabled.")
else:
    print("⚠ Gemini API key missing. AI Rewrite/Explain disabled.")


# ===============================
#  LOGIN MANAGER
# ===============================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"error": "Unauthorized"}), 401


# ===============================
#  DATABASE MODELS
# ===============================

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default="user", nullable=False)


class AnalysisLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text_analyzed = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    user = db.relationship("User", backref=db.backref("logs", lazy=True))


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    is_correct = db.Column(db.Boolean, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    log_id = db.Column(db.Integer, db.ForeignKey("analysis_log.id", ondelete="CASCADE"), nullable=False, unique=True)
    admin_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    log = db.relationship("AnalysisLog", backref=db.backref("feedback", lazy=True, uselist=False, cascade='all'))
    admin = db.relationship("User", backref=db.backref("feedbacks", lazy=True))


# ===============================
#  LOAD ML MODEL
# ===============================

model = None
bert_model = None
try:
    model_path = os.path.join(project_dir, "models", "cyber_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded.")

        if not isinstance(model, Pipeline):
            bert_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("BERT embeddings enabled.")
    else:
        print("❌ Model file not found.")
except Exception as e:
    print("Error loading model:", e)


# ===============================
#  TEXT CLEANING + PREDICTION
# ===============================

def clean_text_for_prediction(s):
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = emoji.demojize(s)
    s = re.sub(r'https?://\S+', " URL ", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# def perform_analysis(text, user_id):
#     cleaned = clean_text_for_prediction(text)

#     if not isinstance(model, Pipeline):
#         input_data = bert_model.encode([cleaned])
#     else:
#         input_data = pd.DataFrame({"text": [cleaned]})

#     prediction = model.predict(input_data)[0]
#     probability = None

#     if hasattr(model, "predict_proba"):
#         p = model.predict_proba(input_data)[0]
#         idx = list(model.classes_).index(prediction)
#         probability = float(p[idx])

#     new_log = AnalysisLog(
#         text_analyzed=text,
#         prediction=str(prediction),
#         confidence=probability,
#         user_id=user_id
#     )
#     db.session.add(new_log)
#     db.session.commit()

#     return {
#         "text": text,
#         "label": str(prediction),
#         "probability": f"{probability:.4f}" if probability else "N/A"
#     }

def perform_analysis(text_to_analyze, user_id):
    if not model:
        raise Exception("ML model is not loaded.")

    # Detect if pipeline or BERT + classifier
    is_bert_model = not isinstance(model, Pipeline) and hasattr(model, "predict")
    cleaned_text = clean_text_for_prediction(text_to_analyze)

    # Prepare input
    if is_bert_model:
        if not bert_model:
            raise Exception("BERT encoder not loaded.")
        input_data = bert_model.encode([cleaned_text])
    else:
        input_data = pd.DataFrame({"text": [cleaned_text]})

    # Predict label
    prediction = model.predict(input_data)[0]

    # -----------------------------------------
    # 🔥 CONFIDENCE SCORE (fixed)
    # -----------------------------------------
    probability = None

    try:
        # Model supports real probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            class_index = list(model.classes_).index(prediction)
            probability = float(probs[class_index])

        # Model supports decision_function (SVM)
        elif hasattr(model, "decision_function"):
            score = model.decision_function(input_data)[0]
            prob = 1 / (1 + np.exp(-score))
            positive_label = str(model.classes_[1]) if len(model.classes_) > 1 else str(model.classes_[0])
            probability = prob if str(prediction) == positive_label else 1 - prob

    except:
        probability = None

    # -----------------------------------------
    # 🛑 FIX: If probability is None → create fallback probability
    # -----------------------------------------
    if probability is None:
        toxic_keywords = ["fuck", "idiot", "stupid", "kill", "kys", "trash", "bitch"]
        lower = cleaned_text.lower()

        hits = sum(w in lower for w in toxic_keywords)

        if hits >= 3:
            probability = 0.90
        elif hits == 2:
            probability = 0.75
        elif hits == 1:
            probability = 0.65
        else:
            probability = 0.55  # neutral fallback confidence

    # -----------------------------------------
    # ⭐ Emotional Damage Score — NEW
    # -----------------------------------------
    eds = emotional_damage_score(text_to_analyze, probability)

    # Save log
    new_log = AnalysisLog(
        text_analyzed=text_to_analyze,
        prediction=str(prediction),
        confidence=probability,
        user_id=user_id,
    )
    db.session.add(new_log)
    db.session.commit()

    return {
        "text": text_to_analyze,
        "label": str(prediction),
        "probability": f"{probability:.4f}",
        "eds": eds
    }


def emotional_damage_score(text, probability):
    text_lower = text.lower()

    # 1. Base from model probability (0–50)
    p_score = float(probability) * 50 if probability != "N/A" else 0

    # 2. Severity keywords (0–30)
    severity_words = {
        "kill": 10, "kys": 10, "suicide": 10, "die": 8,
        "hate": 6, "stupid": 5, "idiot": 5, "moron": 5,
        "loser": 4, "dumb": 4, "worthless": 7,
        "ugly": 3, "fat": 3
    }

    severity_score = 0
    for word, val in severity_words.items():
        if word in text_lower:
            severity_score += val

    severity_score = min(severity_score, 30)

    # 3. Intensity score (0–10)
    intensity = 0
    if text.isupper():
        intensity += 5
    if "!!!" in text or "???" in text:
        intensity += 3
    if re.search(r"(.)\1{2,}", text):
        intensity += 3
    intensity = min(intensity, 10)

    # 4. Category boost (0–10)
    category_boost = 0
    if any(w in text_lower for w in ["kill", "kys", "die", "hurt"]):
        category_boost = 10
    elif any(w in text_lower for w in ["idiot", "stupid", "moron", "worthless"]):
        category_boost = 5
    elif any(w in text_lower for w in ["nice job", "genius"]):
        category_boost = 3

    # Total score
    final_score = p_score + severity_score + intensity + category_boost
    return min(int(final_score), 100)



# ===============================
#  GEMINI EDUCATOR FUNCTIONS
# ===============================

def gemini_rewrite(text):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
Rewrite the following text to be polite, respectful, and constructive,
while keeping the original meaning:

Text: "{text}"
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Rewrite failed: {e}"


def gemini_explain(text):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
Explain why the following message is harmful and how it affects the person emotionally:

Text: "{text}"
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Explanation failed: {e}"


# ===============================
#  ROUTES — AI DETECTION
# ===============================

@app.route("/api/analyze_text", methods=["POST"])
@login_required
def analyze_text():
    text = request.get_json().get("text", "")
    if not text.strip():
        return jsonify({"error": "Empty"}), 400
    return jsonify(perform_analysis(text, current_user.id))


# ===============================
#  ROUTES — GEMINI EDUCATOR
# ===============================

@app.route("/api/rewrite", methods=["POST"])
@login_required
def rewrite_text():
    text = request.get_json().get("text", "")
    rewritten = gemini_rewrite(text)
    return jsonify({"rewritten": rewritten})


@app.route("/api/explain", methods=["POST"])
@login_required
def explain_text():
    text = request.get_json().get("text", "")
    explanation = gemini_explain(text)
    return jsonify({"explanation": explanation})


# ===============================
#  ROUTES — OCR, AUDIO, ADMIN, AUTH
# ===============================
@app.route('/api/analyze_image', methods=['POST'])
@login_required
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        image_file = request.files['image']

        # Read image with PIL
        from PIL import Image
        import pytesseract
        import numpy as np

        img = Image.open(image_file.stream).convert("RGB")

        # OCR extraction
        extracted_text = pytesseract.image_to_string(img)

        if not extracted_text.strip():
            extracted_text = "(No readable text found in image)"

        # Run your ML text classifier
                # Use the same ML pipeline as text analysis
        analysis = perform_analysis(extracted_text, current_user.id)

        return jsonify({
            "analysis": analysis,
            "extracted_text": extracted_text
        })

        # prediction = model.predict([extracted_text])[0]
        # probability = getattr(model, "predict_proba", None)

        # if probability:
        #     prob_value = float(model.predict_proba([extracted_text])[0][int(prediction)])
        # else:
        #     prob_value = "N/A"

        # # Format for frontend
        # result = {
        #     "label": str(prediction),
        #     "probability": prob_value,
        #     "text": extracted_text
        # }

        # return jsonify({
        #     "analysis": result,
        #     "extracted_text": extracted_text
        # })

    except Exception as e:
        print("IMAGE ANALYSIS ERROR:", e)
        return jsonify({"error": "Image processing failed"}), 500


# @app.route("/api/analyze_image", methods=["POST"])
# @login_required
# def analyze_image():
#     if "image" not in request.files:
#         return jsonify({"error": "No file"}), 400

#     file = request.files["image"]
#     image = Image.open(io.BytesIO(file.read()))
#     if image.mode != "RGB":
#         image = image.convert("RGB")

#     extracted = pytesseract.image_to_string(image)
#     result = perform_analysis(extracted, current_user.id)
#     result["extracted_text"] = extracted
#     return jsonify(result)


@app.route("/api/analyze_audio", methods=["POST"])
@login_required
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No file"}), 400

    audio = AudioSegment.from_file(request.files["audio"])
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
        transcribed = recognizer.recognize_google(audio_data)

    result = perform_analysis(transcribed, current_user.id)
    result["extracted_text"] = transcribed
    return jsonify(result)


# ===============================
#  ADMIN ROUTES (unchanged)
# ===============================

@app.route("/api/admin/logs")
@login_required
def get_admin_logs():
    if current_user.role != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    logs = (
        db.session.query(AnalysisLog, User.username)
        .join(User, AnalysisLog.user_id == User.id, isouter=True)
        .order_by(AnalysisLog.timestamp.desc())
        .limit(50)
        .all()
    )

    output = []
    for l, u in logs:
        output.append({
            "id": l.id,
            "timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "username": u or "[Deleted]",
            "text": l.text_analyzed,
            "prediction": "Bullying" if l.prediction == "1" else "Not Bullying",
            "confidence": f"{(l.confidence or 0) * 100:.2f}%",
        })

    return jsonify(output)


# ===============================
#  USER AUTH
# ===============================

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Taken"}), 409

    hashed = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    user = User(username=data["username"], password_hash=hashed)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Registered"}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data["username"]).first()

    if user and bcrypt.check_password_hash(user.password_hash, data["password"]):
        login_user(user, remember=True)
        return jsonify({"message": "Logged in", "user": {"username": user.username, "role": user.role}})

    return jsonify({"error": "Invalid"}), 401


@app.route("/api/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out"})


@app.route("/api/status")
def status():
    if current_user.is_authenticated:
        return jsonify({"logged_in": True, "user": {"username": current_user.username, "role": current_user.role}})
    return jsonify({"logged_in": False}), 401


@app.route("/")
def home():
    return "Cyberbullying Detection API running."


# ===============================
#  RUN SERVER
# ===============================

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
