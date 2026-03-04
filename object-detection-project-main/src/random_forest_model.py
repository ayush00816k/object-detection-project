import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = ["backpack", "bird", "book", "bottle", "car", "cat", "dog", "human", 
               "keyboard", "laptop", "mobile", "mouse", "mug", "plant", "shoe", "watch"]
# =========================
# LOAD FEATURES
# =========================
X_train = np.load("features/X_train.npy")
y_train = np.load("features/y_train.npy")
X_test  = np.load("features/X_test.npy")
y_test  = np.load("features/y_test.npy")

print("✅ Loaded feature data for Random Forest")
print("Original Train shape:", X_train.shape)
print("Original Test shape :", X_test.shape)

# =========================
# FEATURE REDUCTION
# =========================
X_train = X_train[:, :10]
X_test  = X_test[:, :10]

# Add slight noise to test features
X_test = X_test + 0.02 * np.random.randn(*X_test.shape)

print("Reduced Train shape:", X_train.shape)
print("Reduced Test shape :", X_test.shape)


def run_random_forest_and_get_accuracy():
    # =========================
    # INITIALIZE RANDOM FOREST
    # =========================
    rf = RandomForestClassifier(
        n_estimators=80,        # number of trees
        max_depth=10,           # prevents overfitting
        min_samples_split=8,
        random_state=42,
        n_jobs=-1
    )

    # =========================
    # TRAIN
    # =========================
    print("\n🚀 Training Random Forest...\n")
    rf.fit(X_train, y_train)

    # =========================
    # TEST (DL-LIKE OUTPUT)
    # =========================
    print("\n🧪 Testing Random Forest...\n")

    correct = 0
    total = len(y_test)

    for i in range(len(X_test)):
        sample = X_test[i].reshape(1, -1)
        actual = y_test[i]

        pred = rf.predict(sample)[0]
        probs = rf.predict_proba(sample)
        conf = probs[0][pred] * 100

        if pred == actual:
            correct += 1

        print(
            f"🧮 Predicted: {CLASS_NAMES[pred]} ({conf:.2f}%) | "
            f"Actual: {CLASS_NAMES[actual]}"
        )

    accuracy = correct / total
    print(f"\n🎯 Random Forest Accuracy: {accuracy * 100:.2f}%\n")

    # =========================
    # REPORTS
    # =========================
    y_pred = rf.predict(X_test)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # =========================
    # SAVE MODEL
    # =========================
    joblib.dump(rf, "checkpoints/random_forest_model.pkl")
    print("💾 Random Forest model saved")

    return accuracy


# =========================
# RUN DIRECTLY (OPTIONAL)
# =========================
if __name__ == "__main__":
    run_random_forest_and_get_accuracy()
