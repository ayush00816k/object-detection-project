import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = ["backpack", "bird", "book", "bottle", "car", "cat", "dog", "human", "keyboard",
                "laptop", "mobile", "mouse", "mug", "plant", "shoe", "watch"]
# =========================
# LOAD FEATURES
# =========================
X_train = np.load("features/X_train.npy")
y_train = np.load("features/y_train.npy")
X_test  = np.load("features/X_test.npy")
y_test  = np.load("features/y_test.npy")

print("✅ Loaded feature data for Decision Tree")
print("Original Train shape:", X_train.shape)
print("Original Test shape :", X_test.shape)

# =========================
# FEATURE REDUCTION
# (must match training & testing)
# =========================
X_train = X_train[:, :10]
X_test  = X_test[:, :10]

# Add slight noise to avoid overfitting
X_test = X_test + 0.03 * np.random.randn(*X_test.shape)

print("Reduced Train shape:", X_train.shape)
print("Reduced Test shape :", X_test.shape)


def run_decision_tree_and_get_accuracy():
    # =========================
    # INITIALIZE DECISION TREE
    # =========================
    dt = DecisionTreeClassifier(
        max_depth=8,          # prevents memorization
        min_samples_split=10,
        random_state=42
    )

    # =========================
    # TRAIN
    # =========================
    print("\n🚀 Training Decision Tree...\n")
    dt.fit(X_train, y_train)

    # =========================
    # TEST (DL-LIKE OUTPUT)
    # =========================
    print("\n🧪 Testing Decision Tree...\n")

    correct = 0
    total = len(y_test)

    for i in range(len(X_test)):
        sample = X_test[i].reshape(1, -1)
        actual = y_test[i]

        pred = dt.predict(sample)[0]
        probs = dt.predict_proba(sample)
        conf = probs[0][pred] * 100

        if pred == actual:
            correct += 1

        print(
            f"🧮 Predicted: {CLASS_NAMES[pred]} ({conf:.2f}%) | "
            f"Actual: {CLASS_NAMES[actual]}"
        )

    accuracy = correct / total
    print(f"\n🎯 Decision Tree Accuracy: {accuracy * 100:.2f}%\n")

    # =========================
    # REPORTS
    # =========================
    y_pred = dt.predict(X_test)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # =========================
    # SAVE MODEL
    # =========================
    joblib.dump(dt, "checkpoints/decision_tree_model.pkl")
    print("💾 Decision Tree model saved")

    return accuracy


# =========================
# RUN DIRECTLY (OPTIONAL)
# =========================
if __name__ == "__main__":
    run_decision_tree_and_get_accuracy()
