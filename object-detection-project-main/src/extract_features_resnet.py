import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_resnet import get_resnet18_model

# ===== PATHS =====
train_dir = os.path.join(os.getcwd(), "data", "images", "train")
test_dir = os.path.join(os.getcwd(), "data", "images", "test")
features_dir = "features"
os.makedirs(features_dir, exist_ok=True)

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== TRANSFORMS (same as testing) =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== LOAD DATA =====
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# ===== LOAD TRAINED RESNET MODEL =====
num_classes = len(train_data.classes)
model = get_resnet18_model(num_classes=num_classes)

model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
model.to(device)
model.eval()

# ===== REMOVE FINAL CLASSIFICATION LAYER =====
# We only want feature vectors, not class predictions
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# ===== FEATURE EXTRACTION FUNCTION =====
def extract_features(loader):
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            output = feature_extractor(images)
            output = output.view(output.size(0), -1)  # flatten

            features.append(output.cpu().numpy())
            labels.append(targets.numpy())

    return np.vstack(features), np.hstack(labels)

# ===== EXTRACT FEATURES =====
print("🚀 Extracting training features...")
X_train, y_train = extract_features(train_loader)

print("🚀 Extracting testing features...")
X_test, y_test = extract_features(test_loader)

# ===== SAVE FEATURES =====
np.save(os.path.join(features_dir, "X_train.npy"), X_train)
np.save(os.path.join(features_dir, "y_train.npy"), y_train)
np.save(os.path.join(features_dir, "X_test.npy"), X_test)
np.save(os.path.join(features_dir, "y_test.npy"), y_test)

print("✅ Feature extraction completed successfully")
print("Train features shape:", X_train.shape)
print("Test features shape :", X_test.shape)
