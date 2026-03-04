from ultralytics import YOLO
import cv2

# =========================
# CONFIG
# =========================
# These must match your dataset folder names exactly
ALLOWED_CLASSES = [
    "backpack","bird","book","bottle","car","cat","dog","human","keyboard","laptop","mobile","mouse","mug","plant","shoe","watch"
]
CLASS_MAPPING = {
    "person": "human",
    "bird": "bird",
    "car": "car",
    "cat": "cat",
    "dog": "dog",
    "bottle": "bottle",
    "cup": "mug",          
    "wine glass": "mug",   
    "laptop": "laptop",
    "cell phone": "mobile",
    "book": "book",
    "clock": "watch",      
    "tie": "watch",        
    "skis": "shoe",        
    "snowboard": "shoe",   
    "suitcase": "shoe",  
    "backpack": "backpack",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "potted plant ": "plant"
}

# =========================
# LOAD YOLO MODEL (ONCE)
# =========================
model = YOLO("yolov8s.pt")

def predict_yolo_single(image):
    """
    YOLO single-object prediction logic.
    """
    if image is None:
        return "unknown", 0.0

    # YOLO inference
    results = model(image, conf=0.15, iou=0.45, max_det=10, verbose=False)
    boxes = results[0].boxes

    best_label = "unknown"
    best_conf = 0.0

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            conf = box.conf.item() * 100
            coco_label = model.names[class_id]

            # Map COCO label → dataset label
            label = CLASS_MAPPING.get(coco_label, "unknown")

            # Filter only for classes in your dataset
            if label in ALLOWED_CLASSES and conf > best_conf:
                best_label = label
                best_conf = conf

    return best_label, best_conf
