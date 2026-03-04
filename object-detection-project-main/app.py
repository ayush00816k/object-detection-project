import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import io
import numpy as np
import cv2
import os
import joblib
from flask import Flask, render_template, request, jsonify
try:
    from flask_cors import CORS
except ImportError:
    print("✗ Required package 'flask-cors' not found. Run: python -m pip install -r requirements.txt")
    print("✗ Or install just it: python -m pip install flask-cors")
    import sys
    sys.exit(1)
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
from dotenv import load_dotenv
from bson.objectid import ObjectId
import sys

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ==================== FLASK APP SETUP ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
CORS(app)

# ==================== MONGODB CONNECTION ====================
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client['objectify_db']
    users_collection = db['users']
    print("✓ MongoDB connected successfully")
except Exception as e:
    print(f"✗ MongoDB connection failed: {e}")
    db = None
    users_collection = None

# ==================== JWT DECORATOR ====================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# ==================== MODEL IMPORTS ------------------
try:
    from model_resnet import get_resnet18_model
    from model_mobilenet import get_mobilenet_model
    from yolo_model import predict_yolo_single
    from ai_image_detector import AIImageDetector
except ImportError:
    from src.model_resnet import get_resnet18_model
    from src.model_mobilenet import get_mobilenet_model
    from src.yolo_model import predict_yolo_single
    from src.ai_image_detector import AIImageDetector

CLASS_NAMES = [
    "backpack", "bird", "book", "bottle", "car", "cat", "dog", "human",
    "keyboard", "laptop", "mobile", "mouse", "mug", "plant", "shoe", "watch"
]

TEMPERATURE = 2.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 16)
        )

    def forward(self, x):
        return self.network(x)

# ==================== LOAD MODELS ====================
print("Loading models...")
cnn_model = CNNModel()
cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device).eval()

resnet_model = get_resnet18_model(num_classes=16)
resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device).eval()

mobilenet_model = get_mobilenet_model(num_classes=16)
mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device))
mobilenet_model.to(device).eval()

resnet_feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_feature_extractor.to(device).eval()

decision_tree = joblib.load("checkpoints/decision_tree_model.pkl")
knn = joblib.load("checkpoints/knn_model.pkl")
random_forest = joblib.load("checkpoints/random_forest_model.pkl")
svm = joblib.load("checkpoints/svm_model.pkl")

ai_detector = None
# ==================== LOAD AI IMAGE DETECTOR ====================
try:
    print("Loading AI Image Detector...")
    # Sensitivity can be controlled by environment variable AI_DETECT_SENSITIVITY (low/medium/high)
    import os
    sensitivity = os.getenv('AI_DETECT_SENSITIVITY', 'medium')

    # Try HuggingFace model first (most accurate), fall back to hybrid or artifact if not available
    try:
        ai_detector = AIImageDetector(method='hybrid', sensitivity=sensitivity)
        print("✓ AI Image Detector (Hybrid) loaded successfully")
    except Exception:
        print("⚠️ HuggingFace model not available, trying hybrid approach...")
        try:
            ai_detector = AIImageDetector(method='hybrid', sensitivity=sensitivity)
            print("✓ AI Image Detector (Hybrid) loaded successfully")
        except Exception:
            print("⚠️ Hybrid detector failed, trying artifact-only approach...")
            try:
                ai_detector = AIImageDetector(method='artifact', sensitivity=sensitivity)
                print("✓ AI Image Detector (Artifact) loaded successfully")
            except Exception as e:
                print(f"⚠️ AI Image Detector failed to load: {e}")
                ai_detector = None
except Exception as e:
    print(f"⚠️ Error during AI detector initialization: {e}")
    ai_detector = None

# Report whether detector is available (do NOT overwrite a successfully loaded detector)
if ai_detector is None:
    print("Will continue without AI detection")
else:
    print("AI detection initialized and ready")

# ----------------- Checkpoint watcher: reload detector after retraining completes -----------------
import threading, time

_ai_detector_mtime = None

def _watch_checkpoints(interval=30):
    global ai_detector, _ai_detector_mtime
    ckpt = 'checkpoints/ai_detector.pth'
    while True:
        try:
            if os.path.exists(ckpt):
                m = os.path.getmtime(ckpt)
                if _ai_detector_mtime is None:
                    _ai_detector_mtime = m
                elif m > _ai_detector_mtime:
                    print('Detected updated ai detector checkpoint, reloading detector...')
                    try:
                        sensitivity = os.getenv('AI_DETECT_SENSITIVITY', 'high')
                        new_detector = AIImageDetector(method='hybrid', sensitivity=sensitivity)
                        ai_detector = new_detector
                        _ai_detector_mtime = m
                        print('AI detector reloaded from checkpoint.')
                    except Exception as e:
                        print('Error reloading detector:', e)
        except Exception as e:
            print('Checkpoint watcher error:', e)
        time.sleep(interval)

# Start watcher in background daemon thread
try:
    t = threading.Thread(target=_watch_checkpoints, args=(30,), daemon=True)
    t.start()
except Exception:
    pass

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def apply_temperature(probs, temperature=TEMPERATURE):
    logits = torch.log(torch.tensor(probs) + 1e-10)
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0).numpy()

def run_all_predictions_from_image(image: np.ndarray):
    try:
        image_pil = Image.fromarray(image).convert("RGB")
        tensor = transform(image_pil).unsqueeze(0).to(device)
        cv_img = image[:, :, ::-1]

        import concurrent.futures

        # Define tasks
        def task_dl_ml():
            local_scores = {}
            local_preds = {}
            
            # 1. Deep Learning Models
            with torch.no_grad():
                # CNN
                try:
                    cnn_output = cnn_model(tensor)
                    cnn_probs = torch.softmax(cnn_output, dim=1)[0].cpu().numpy()
                    cnn_probs = apply_temperature(cnn_probs)
                    local_scores["CNN"] = float(np.max(cnn_probs) * 100)
                    local_preds["CNN"] = CLASS_NAMES[int(np.argmax(cnn_probs))]
                    print(f"[CNN] Prediction: {local_preds['CNN']}, Confidence: {local_scores['CNN']:.2f}%")
                except Exception as e:
                    print(f"[ERROR] CNN failed: {str(e)}")
                    local_scores["CNN"] = 0.0
                    local_preds["CNN"] = "Unknown"

                # ResNet-18 (Optimized: Shared Backbone)
                try:
                    # Run backbone once
                    raw_features = resnet_feature_extractor(tensor)
                    full_features = raw_features.view(tensor.size(0), -1)
                    
                    # 1. Classification
                    res_output = resnet_model.fc(full_features)
                    res_probs = torch.softmax(res_output, dim=1)[0].cpu().numpy()
                    res_probs = apply_temperature(res_probs)
                    local_scores["ResNet-18"] = float(np.max(res_probs) * 100)
                    local_preds["ResNet-18"] = CLASS_NAMES[int(np.argmax(res_probs))]
                    print(f"[ResNet-18] Prediction: {local_preds['ResNet-18']}, Confidence: {local_scores['ResNet-18']:.2f}%")
                    
                    # 2. Prepare for ML models
                    full_features_np = full_features.cpu().numpy()
                    features_knn = full_features_np[:, :5]
                    features_rf = full_features_np[:, :10]
                    features_dt = full_features_np[:, :10]
                    features_svm = full_features_np
                except Exception as e:
                    print(f"[ERROR] ResNet/Features failed: {str(e)}")
                    local_scores["ResNet-18"] = 0.0
                    local_preds["ResNet-18"] = "Unknown"
                    features_knn = np.zeros((1, 5))
                    features_rf = np.zeros((1, 10))
                    features_dt = np.zeros((1, 10))
                    features_svm = np.zeros((1, 2048))

                # MobileNet
                try:
                    mob_output = mobilenet_model(tensor)
                    mob_probs = torch.softmax(mob_output, dim=1)[0].cpu().numpy()
                    mob_probs = apply_temperature(mob_probs)
                    local_scores["MobileNet"] = float(np.max(mob_probs) * 100)
                    local_preds["MobileNet"] = CLASS_NAMES[int(np.argmax(mob_probs))]
                    print(f"[MobileNet] Prediction: {local_preds['MobileNet']}, Confidence: {local_scores['MobileNet']:.2f}%")
                except Exception as e:
                    print(f"[ERROR] MobileNet failed: {str(e)}")
                    local_scores["MobileNet"] = 0.0
                    local_preds["MobileNet"] = "Unknown"

            # 2. ML Models (Depend on DL features)
            # KNN
            try:
                knn_pred = knn.predict(features_knn.reshape(1, -1))[0]
                knn_probs = knn.predict_proba(features_knn.reshape(1, -1))[0]
                knn_conf = float(np.max(knn_probs) * 100)
                local_scores["KNN"] = knn_conf
                local_preds["KNN"] = CLASS_NAMES[int(knn_pred)]
                print(f"[KNN] Prediction: {CLASS_NAMES[int(knn_pred)]}, Confidence: {knn_conf:.2f}%")
            except Exception as e:
                print(f"[ERROR] KNN failed: {str(e)}")
                local_scores["KNN"] = 0.0
                local_preds["KNN"] = "Unknown"

            # SVM
            try:
                svm_pred = svm.predict(features_svm.reshape(1, -1))[0]
                try:
                    svm_decision = svm.decision_function(features_svm.reshape(1, -1))[0]
                    svm_conf = float((np.mean(svm_decision) + 1) * 50)
                    svm_conf = min(100.0, max(0.0, svm_conf))
                except:
                    svm_conf = 50.0
                local_scores["SVM"] = svm_conf
                local_preds["SVM"] = CLASS_NAMES[int(svm_pred)]
                print(f"[SVM] Prediction: {CLASS_NAMES[int(svm_pred)]}, Confidence: {svm_conf:.2f}%")
            except Exception as e:
                print(f"[ERROR] SVM failed: {str(e)}")
                local_scores["SVM"] = 0.0
                local_preds["SVM"] = "Unknown"

            # Decision Tree
            try:
                dt_pred = decision_tree.predict(features_dt.reshape(1, -1))[0]
                dt_probs = decision_tree.predict_proba(features_dt.reshape(1, -1))[0]
                dt_conf = float(np.max(dt_probs) * 100)
                local_scores["Decision Tree"] = dt_conf
                local_preds["Decision Tree"] = CLASS_NAMES[int(dt_pred)]
                print(f"[Decision Tree] Prediction: {CLASS_NAMES[int(dt_pred)]}, Confidence: {dt_conf:.2f}%")
            except Exception as e:
                print(f"[ERROR] Decision Tree failed: {str(e)}")
                local_scores["Decision Tree"] = 0.0
                local_preds["Decision Tree"] = "Unknown"

            # Random Forest
            try:
                rf_pred = random_forest.predict(features_rf.reshape(1, -1))[0]
                rf_probs = random_forest.predict_proba(features_rf.reshape(1, -1))[0]
                rf_conf = float(np.max(rf_probs) * 100)
                local_scores["Random Forest"] = rf_conf
                local_preds["Random Forest"] = CLASS_NAMES[int(rf_pred)]
                print(f"[Random Forest] Prediction: {CLASS_NAMES[int(rf_pred)]}, Confidence: {rf_conf:.2f}%")
            except Exception as e:
                print(f"[ERROR] Random Forest failed: {str(e)}")
                local_scores["Random Forest"] = 0.0
                local_preds["Random Forest"] = "Unknown"
            
            return local_scores, local_preds

        def task_yolo():
            try:
                y_label, y_conf = predict_yolo_single(cv_img)
                print(f"[YOLO] Prediction: {y_label}, Confidence: {y_conf:.2f}%")
                return {"YOLO": y_conf}, {"YOLO": y_label}
            except Exception as e:
                print(f"[ERROR] YOLO failed: {str(e)}")
                return {"YOLO": 0.0}, {"YOLO": "Unknown"}

        def task_ai():
            if ai_detector is None:
                return None
            try:
                print("[AI DETECTOR] Running AI image detection...")
                import tempfile
                # Use a unique temp file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                    image_pil.save(tmp_path)
                
                ai_result = ai_detector.predict(tmp_path)
                
                try:
                    os.remove(tmp_path)
                except:
                    pass
                
                res = {
                    "is_ai_generated": ai_result.get('is_ai_generated', False),
                    "confidence": float(ai_result.get('confidence', 0)),
                    "label": ai_result.get('label', 'Unknown'),
                    "verdict": ai_result.get('verdict', ''),
                    "method": ai_result.get('method', 'unknown'),
                    "metrics": ai_result.get('metrics', {}),
                    "explanation": ai_result.get('explanation', '')
                }
                print(f"[AI DETECTOR] Result: {res['label']} ({res['confidence']:.2f}%)")
                return res
            except Exception as e:
                print(f"[ERROR] AI detection failed: {str(e)}")
                return {
                    "is_ai_generated": False,
                    "confidence": 0,
                    "label": "Error",
                    "verdict": f"Detection failed: {str(e)}",
                    "method": "error"
                }

        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_dl = executor.submit(task_dl_ml)
            future_yolo = executor.submit(task_yolo)
            future_ai = executor.submit(task_ai)

            # Gather results
            dl_scores, dl_preds = future_dl.result()
            yolo_scores, yolo_preds = future_yolo.result()
            ai_detection_result = future_ai.result()

        # Merge
        all_scores = {**dl_scores, **yolo_scores}
        all_preds = {**dl_preds, **yolo_preds}

        best_model = max(all_scores, key=all_scores.get)
        
        # Get top 3 models
        top3 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_models = [
            {
                "model": model,
                "confidence": score,
                "prediction": all_preds[model]
            }
            for model, score in top3
        ]
        
        print(f"\n[FINAL] Best Model: {best_model}, Confidence: {all_scores[best_model]:.2f}%")
        print(f"[FINAL] Top 3 Models: {[(m['model'], m['confidence']) for m in top3_models]}")
        print(f"[FINAL] All Scores: {all_scores}\n")

        return {
            "Final Prediction": all_preds[best_model],
            "Best Model": best_model,
            "Confidence (%)": round(all_scores[best_model], 2),
            "All Predictions": all_preds,
            "All Scores": all_scores,
            "Top 3 Models": top3_models,
            "AI Detection": ai_detection_result
        }
    except Exception as e:
        print(f"[CRITICAL ERROR] run_all_predictions_from_image failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "Final Prediction": "Error",
            "Best Model": "N/A",
            "Confidence (%)": 0,
            "All Predictions": {},
            "All Scores": {},
            "Top 3 Models": []
        }

print("✓ All models loaded successfully")

# ==================== PAGE ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/live-upload')
def live_upload():
    # Check if user is authenticated via JWT token in localStorage
    # This will be checked on the frontend with JavaScript
    return render_template('live_upload.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/saved-tests')
def saved_tests():
    """Display saved tests page (requires authentication via JS)"""
    return render_template('saved_tests.html')

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """Register a new user"""
    try:
        if db is None or users_collection is None:
            return jsonify({'message': 'Database not connected'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'message': 'No data provided'}), 400
        
        full_name = data.get('fullName', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        confirm_password = data.get('confirmPassword', '')
        
        # Validation
        if not all([full_name, email, password, confirm_password]):
            return jsonify({'message': 'All fields are required'}), 400
        
        if len(password) < 8:
            return jsonify({'message': 'Password must be at least 8 characters'}), 400
        
        if password != confirm_password:
            return jsonify({'message': 'Passwords do not match'}), 400
        
        if '@' not in email:
            return jsonify({'message': 'Invalid email format'}), 400
        
        # Check if user already exists
        if users_collection.find_one({'email': email}):
            return jsonify({'message': 'Email already registered'}), 400
        
        # Hash password and create user
        hashed_password = generate_password_hash(password)
        
        user = {
            'fullName': full_name,
            'email': email,
            'password': hashed_password,
            'createdAt': datetime.datetime.utcnow()
        }
        
        result = users_collection.insert_one(user)
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': str(result.inserted_id),
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': str(result.inserted_id),
                'email': email,
                'fullName': full_name
            }
        }), 201
    except Exception as e:
        print(f"Signup Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """Login user and return JWT token"""
    try:
        if db is None or users_collection is None:
            return jsonify({'message': 'Database not connected'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'message': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400
        
        # Find user
        user = users_collection.find_one({'email': email})
        
        if not user:
            return jsonify({'message': 'Invalid email or password'}), 401
        
        # Check password
        if not check_password_hash(user['password'], password):
            return jsonify({'message': 'Invalid email or password'}), 401
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': str(user['_id']),
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': str(user['_id']),
                'email': email,
                'fullName': user.get('fullName', '')
            }
        }), 200
    except Exception as e:
        print(f"Login Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500

@app.route('/api/verify-token', methods=['POST'])
def verify_token():
    """Verify JWT token"""
    token = None
    
    if 'Authorization' in request.headers:
        auth_header = request.headers['Authorization']
        try:
            token = auth_header.split(" ")[1]
        except IndexError:
            return jsonify({'message': 'Invalid token format'}), 401
    
    if not token:
        return jsonify({'message': 'Token is missing'}), 401
    
    try:
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return jsonify({
            'valid': True,
            'user_id': data['user_id'],
            'email': data['email']
        }), 200
    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token'}), 401

@app.route('/api/user/profile', methods=['GET'])
@token_required
def get_user_profile(current_user):
    """Get user profile (requires authentication)"""
    if db is None or users_collection is None:
        return jsonify({'message': 'Database not connected'}), 500
    
    user = users_collection.find_one({'_id': ObjectId(current_user)})
    
    if not user:
        return jsonify({'message': 'User not found'}), 404
    
    return jsonify({
        'id': str(user['_id']),
        'email': user['email'],
        'fullName': user.get('fullName', ''),
        'createdAt': user.get('createdAt', '').isoformat() if user.get('createdAt') else None
    }), 200

# ==================== DETECTION ROUTE ====================

@app.route('/api/detect', methods=['POST'])
@token_required
def api_detect(current_user):
    """Detect objects in image (requires authentication)"""
    if 'file' not in request.files:
        return jsonify({'message': 'No image provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
    
    try:
        image = Image.open(file).convert('RGB')
        image_array = np.array(image)
        results = run_all_predictions_from_image(image_array)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'message': f'Error processing image: {str(e)}'}), 500

# ==================== TEST RESULTS ROUTES ====================

@app.route('/api/save-test-result', methods=['POST'])
@token_required
def save_test_result(current_user):
    """Save test result to database (requires authentication)"""
    try:
        if db is None or users_collection is None:
            return jsonify({'message': 'Database not connected'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'message': 'No data provided'}), 400
        
        # Get or create test_results collection
        test_results_collection = db['test_results']
        
        # Extract primary object and confidence from the detection results
        results = data.get('results', {})
        primary_object = results.get('Final Prediction') or results.get('object') or 'Unknown'
        confidence = results.get('Confidence (%)') or results.get('confidence') or 0
        
        # Ensure confidence is a number
        try:
            confidence = float(confidence) if confidence else 0
        except (ValueError, TypeError):
            confidence = 0
        
        # Ensure primary_object is a string
        try:
            primary_object = str(primary_object).strip() if primary_object else 'Unknown'
        except:
            primary_object = 'Unknown'
        
        # Log debug info
        print(f"[SAVE] Debug - results keys: {list(results.keys())}")
        print(f"[SAVE] Debug - primary_object extracted: {primary_object}")
        print(f"[SAVE] Debug - confidence extracted: {confidence}")
        
        test_result = {
            'user_id': current_user,
            'image_data': data.get('image_data'),  # Base64 encoded image
            'detection_results': results,  # Detection results from all models
            'detection_method': data.get('method', 'upload'),  # 'upload' or 'webcam'
            'timestamp': datetime.datetime.utcnow(),
            'primary_object': primary_object,
            'confidence': confidence
        }
        
        print(f"[SAVE] User: {current_user}, Object: {primary_object}, Confidence: {confidence}")
        
        result = test_results_collection.insert_one(test_result)
        
        return jsonify({
            'message': 'Test result saved successfully',
            'result_id': str(result.inserted_id)
        }), 201
    
    except Exception as e:
        print(f"Save Test Result Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500

@app.route('/api/test-connection', methods=['GET'])
def test_connection():
    """Test database connection"""
    try:
        if db is None:
            return jsonify({'status': 'error', 'message': 'Database not connected'}), 500
        
        # Try to ping the database
        db.client.admin.command('ping')
        
        # Check collections
        collections = db.list_collection_names()
        
        return jsonify({
            'status': 'ok',
            'database': 'objectify_db',
            'collections': collections
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-saved-tests', methods=['GET'])
@token_required
def get_saved_tests(current_user):
    """Get all saved test results for current user (requires authentication)"""
    try:
        print(f"DEBUG: Getting saved tests for user: {current_user}")
        
        if db is None or users_collection is None:
            return jsonify({'message': 'Database not connected'}), 500
        
        test_results_collection = db['test_results']
        
        # Fetch all test results for the user, sorted by most recent first
        results = list(test_results_collection.find(
            {'user_id': current_user}
        ).sort('timestamp', -1).limit(100))
        
        # Convert ObjectId to string for JSON serialization
        for result in results:
            result['_id'] = str(result['_id'])
            result['timestamp'] = result['timestamp'].isoformat() if result.get('timestamp') else None
        
        return jsonify({
            'saved_tests': results,
            'count': len(results)
        }), 200
    
    except Exception as e:
        print(f"ERROR - Get Saved Tests Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500

@app.route('/api/delete-test-result/<test_id>', methods=['DELETE'])
@token_required
def delete_test_result(current_user, test_id):
    """Delete a specific test result (requires authentication)"""
    try:
        if db is None or users_collection is None:
            return jsonify({'message': 'Database not connected'}), 500
        
        test_results_collection = db['test_results']
        
        # Verify the test result belongs to the current user
        test_result = test_results_collection.find_one({
            '_id': ObjectId(test_id),
            'user_id': current_user
        })
        
        if not test_result:
            return jsonify({'message': 'Test result not found or unauthorized'}), 404
        
        # Delete the test result
        test_results_collection.delete_one({'_id': ObjectId(test_id)})
        
        return jsonify({'message': 'Test result deleted successfully'}), 200
    
    except Exception as e:
        print(f"Delete Test Result Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500

@app.route('/api/clear-all-tests', methods=['DELETE'])
@token_required
def clear_all_tests(current_user):
    """Clear all test results for current user (requires authentication)"""
    try:
        if db is None:
            return jsonify({'message': 'Database not connected'}), 500
        
        test_results_collection = db['test_results']
        
        # Delete all test results for the user
        result = test_results_collection.delete_many({'user_id': current_user})
        
        print(f"[DELETE] User: {current_user} - Deleted {result.deleted_count} tests")
        
        return jsonify({
            'message': f'All tests cleared successfully. Deleted {result.deleted_count} tests.',
            'deleted_count': result.deleted_count
        }), 200
    
    except Exception as e:
        print(f"Clear All Tests Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500


@app.route('/api/rerun-ai-detection/<test_id>', methods=['POST'])
@token_required
def rerun_ai_detection(current_user, test_id):
    """Re-run AI detection for a saved test and update record (requires auth)
    Optional query/body param: sensitivity (low|medium|high)
    """
    try:
        if db is None:
            return jsonify({'message': 'Database not connected'}), 500
        test_results_collection = db['test_results']
        test_result = test_results_collection.find_one({'_id': ObjectId(test_id), 'user_id': current_user})
        if not test_result:
            return jsonify({'message': 'Test result not found or unauthorized'}), 404

        # sensitivity param
        sensitivity = request.args.get('sensitivity') or (request.json.get('sensitivity') if request.json else None)
        sensitivity = sensitivity or os.getenv('AI_DETECT_SENSITIVITY', 'high')

        # Get image data (prefer stored file path or base64 image_data)
        image_data = test_result.get('image_path') or test_result.get('image_data')
        if not image_data:
            return jsonify({'message': 'No image available to re-run detection'}), 400

        # If image_data is base64 data URL, save to temp file
        import tempfile, base64, re
        tmp_file = None
        if isinstance(image_data, str) and image_data.startswith('data:'):
            header, encoded = image_data.split(',', 1)
            ext = 'jpg' if 'jpeg' in header or 'jpg' in header else 'png'
            decoded = base64.b64decode(encoded)
            tmp = tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False)
            tmp.write(decoded)
            tmp.flush()
            tmp.close()
            tmp_file = tmp.name
        elif isinstance(image_data, str) and os.path.exists(image_data):
            tmp_file = image_data
        else:
            # unsupported image storage
            return jsonify({'message': 'Unsupported image storage format'}), 400

        # Instantiate a detector with requested sensitivity; prefer local custom model if available
        from src.ai_image_detector import AIImageDetector
        detector = AIImageDetector(method='hybrid', sensitivity=sensitivity)
        new_result = detector.predict(tmp_file)

        # Map to DB shape
        ai_detection = {
            'is_ai_generated': new_result.get('is_ai_generated', new_result.get('is_ai', False)),
            'confidence': float(new_result.get('confidence', 0)),
            'label': new_result.get('label', 'Unknown') if new_result.get('label') else ('AI Generated' if new_result.get('is_ai') else 'Real Photo'),
            'method': new_result.get('method'),
            'verdict': new_result.get('verdict'),
            'explanation': new_result.get('explanation') or (' | '.join(new_result.get('explanations', [])) if new_result.get('explanations') else ''),
            'metrics': new_result.get('metrics', {})
        }

        # Update DB document
        test_results_collection.update_one({'_id': ObjectId(test_id)}, {'$set': {'detection_results.AI Detection': ai_detection}})

        # Clean up temp
        try:
            if tmp_file and tmp_file != image_data:
                os.remove(tmp_file)
        except:
            pass

        return jsonify({'message': 'AI detection re-run successfully', 'ai_detection': ai_detection}), 200
    except Exception as e:
        print(f"Re-run AI detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500


# -------------------- Misclassification reporting & retraining --------------------
@app.route('/api/report-misclassification/<test_id>', methods=['POST'])
@token_required
def report_misclassification(current_user, test_id):
    """Report a misclassified saved test to add to retraining queue"""
    try:
        if db is None:
            return jsonify({'message': 'Database not connected'}), 500

        data = request.json or {}
        correct_label = data.get('correct_label')
        if correct_label not in ['ai', 'real']:
            return jsonify({'message': 'Invalid correct_label'}), 400

        test_results_collection = db['test_results']
        retrain_collection = db['retrain_queue']

        test_result = test_results_collection.find_one({'_id': ObjectId(test_id), 'user_id': current_user})
        if not test_result:
            return jsonify({'message': 'Test result not found or unauthorized'}), 404

        # Prepare directories
        import pathlib, shutil, tempfile, base64
        dest_dir = pathlib.Path('data/ai_detector/train') / ('ai' if correct_label == 'ai' else 'real')
        dest_dir.mkdir(parents=True, exist_ok=True)

        image_data = test_result.get('image_path') or test_result.get('image_data')
        if not image_data:
            return jsonify({'message': 'No image available to save'}), 400

        # Save file
        if isinstance(image_data, str) and image_data.startswith('data:'):
            header, encoded = image_data.split(',', 1)
            ext = 'jpg' if 'jpeg' in header or 'jpg' in header else 'png'
            filename = f"reported_{test_id}_{int(datetime.datetime.utcnow().timestamp())}.{ext}"
            save_path = dest_dir / filename
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(encoded))
        elif isinstance(image_data, str) and os.path.exists(image_data):
            ext = pathlib.Path(image_data).suffix or '.jpg'
            filename = f"reported_{test_id}_{int(datetime.datetime.utcnow().timestamp())}{ext}"
            save_path = dest_dir / filename
            shutil.copy(image_data, save_path)
        else:
            return jsonify({'message': 'Unsupported image storage format'}), 400

        # Insert retrain queue document
        retrain_doc = {
            'test_id': test_id,
            'user_id': current_user,
            'label': correct_label,
            'saved_path': str(save_path),
            'timestamp': datetime.datetime.utcnow()
        }
        retrain_collection.insert_one(retrain_doc)

        # Mark test as reported
        test_results_collection.update_one({'_id': ObjectId(test_id)}, {'$set': {'reported': True, 'reported_label': correct_label}})

        # Trigger retraining if enough reports collected
        RETRAIN_TRIGGER = int(os.getenv('RETRAIN_TRIGGER', 20))
        queue_count = retrain_collection.count_documents({})
        if queue_count >= RETRAIN_TRIGGER:
            # start background training process
            import subprocess
            training_log = 'checkpoints/ai_retrain.log'
            os.makedirs('checkpoints', exist_ok=True)
            cmd = [sys.executable, 'src/train_ai_detector.py', '--data_dir', 'data/ai_detector', '--epochs', '15', '--batch_size', '32', '--save_path', 'checkpoints/ai_detector.pth']
            with open(training_log, 'ab') as out:
                subprocess.Popen(cmd, stdout=out, stderr=out)
            # clear retrain queue after scheduling
            retrain_collection.delete_many({})

        return jsonify({'message': 'Reported for retraining', 'saved_path': str(save_path)}), 200
    except Exception as e:
        print(f"Report error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500


@app.route('/api/run-retrain', methods=['POST'])
@token_required
def run_retrain(current_user):
    """Manually trigger a retraining job. Optional JSON body: {epochs:int, batch_size:int}
    Requires sending {"confirm": true} in body to avoid accidental runs.
    """
    try:
        data = request.json or {}
        if not data.get('confirm'):
            return jsonify({'message': 'Please confirm retrain by sending {"confirm": true} in the request body'}), 400

        epochs = int(data.get('epochs', 15))
        batch_size = int(data.get('batch_size', 32))

        # Start background training process
        import subprocess
        training_log = 'checkpoints/ai_retrain_manual.log'
        os.makedirs('checkpoints', exist_ok=True)
        cmd = [sys.executable, 'src/train_ai_detector.py', '--data_dir', 'data/ai_detector', '--epochs', str(epochs), '--batch_size', str(batch_size), '--save_path', 'checkpoints/ai_detector.pth']
        with open(training_log, 'ab') as out:
            subprocess.Popen(cmd, stdout=out, stderr=out)

        return jsonify({'message': 'Retraining started', 'log_file': training_log}), 200
    except Exception as e:
        print(f"Run retrain error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Route not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'message': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)