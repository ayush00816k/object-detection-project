import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib

from model_cnn import CNNModel
from model_resnet import get_resnet18_model
from model_mobilenet import get_mobilenet_model
from yolo_model import predict_yolo_single

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DL MODELS
# =========================
cnn_model = CNNModel(num_classes=16)
cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device).eval()

resnet_model = get_resnet18_model(num_classes=16)
resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device).eval()


mobilenet_model = get_mobilenet_model(num_classes=16)
mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device))
mobilenet_model.to(device).eval()

# =========================
# LOAD ML MODELS
# =========================
knn_model = joblib.load("checkpoints/knn_model.pkl")
svm_model = joblib.load("checkpoints/svm_model.pkl")
dt_model  = joblib.load("checkpoints/decision_tree_model.pkl")
rf_model  = joblib.load("checkpoints/random_forest_model.pkl")

# =========================
# RESNET FEATURE EXTRACTOR
# =========================
resnet_feature_extractor = torch.nn.Sequential(
    *list(resnet_model.children())[:-1]
)
resnet_feature_extractor.to(device).eval()

# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = ["backpack", "bird", "book", "bottle", "car", "cat", "dog", "human",
                "keyboard", "laptop", "mobile", "mouse", "mug", "plant", "shoe", "watch"]# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# WEBCAM SETUP
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Press 'q' to quit.")

FRAME_SKIP = 6
frame_count = 0

# CNN, ResNet, MobileNet, KNN, SVM, DT, RF, YOLO
pred_texts = [""] * 8

colors = [
    (0, 0, 255),       # CNN
    (255, 255, 0),     # ResNet
    (124, 252, 0),     # MobileNet
    (255, 0, 255),     # KNN
    (0, 165, 255),     # SVM
    (200, 200, 0),     # Decision Tree
    (0, 255, 255),     # Random Forest
    (50, 205, 50)      # YOLO
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    if frame_count % FRAME_SKIP == 0:
        with torch.no_grad():

            out = cnn_model(input_tensor)
            prob = torch.softmax(out, dim=1)
            conf, pred = prob.max(1)
            pred_texts[0] = f"CNN: {class_names[pred.item()]} ({conf.item()*100:.1f}%)"

            out = resnet_model(input_tensor)
            prob = torch.softmax(out, dim=1)
            conf, pred = prob.max(1)
            pred_texts[1] = f"ResNet: {class_names[pred.item()]} ({conf.item()*100:.1f}%)"

            out = mobilenet_model(input_tensor)
            prob = torch.softmax(out, dim=1)
            conf, pred = prob.max(1)
            pred_texts[2] = f"MobileNet: {class_names[pred.item()]} ({conf.item()*100:.1f}%)"

            features = resnet_feature_extractor(input_tensor)
            features = features.view(features.size(0), -1).cpu().numpy()

            f5 = features[:, :5]
            p = knn_model.predict(f5)[0]
            c = knn_model.predict_proba(f5)[0][p] * 100
            pred_texts[3] = f"KNN: {class_names[p]} ({c:.1f}%)"

            p = svm_model.predict(features)[0]
            c = svm_model.predict_proba(features)[0][p] * 100
            pred_texts[4] = f"SVM: {class_names[p]} ({c:.1f}%)"

            f10 = features[:, :10]
            p = dt_model.predict(f10)[0]
            c = dt_model.predict_proba(f10)[0][p] * 100
            pred_texts[5] = f"DT: {class_names[p]} ({c:.1f}%)"

            p = rf_model.predict(f10)[0]
            c = rf_model.predict_proba(f10)[0][p] * 100
            pred_texts[6] = f"RF: {class_names[p]} ({c:.1f}%)"

            # =========================
            # YOLO
            # =========================
            yolo_label, yolo_conf = predict_yolo_single(frame)
            pred_texts[7] = f"YOLO: {yolo_label} ({yolo_conf:.1f}%)"

    # ===== DISPLAY =====
    y = 35
    for text, color in zip(pred_texts, colors):
        cv2.putText(frame, text, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        y += 30

    cv2.imshow("Live Classification (Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
