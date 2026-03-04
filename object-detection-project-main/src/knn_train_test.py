import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


CLASS_NAMES = ["backpack", "bird", "book", "bottle", "car", "cat", "dog", "human", "keyboard",
                "laptop", "mobile", "mouse", "mug", "plant", "shoe", "watch"]
# =========================
# LOAD FEATURES
# =========================
X_train = np.load("features/X_train.npy")
y_train = np.load("features/y_train.npy")
X_test  = np.load("features/X_test.npy")
y_test  = np.load("features/y_test.npy")

print("✅ Loaded feature data")
print("Original Training shape:", X_train.shape)
print("Original Testing shape :", X_test.shape)


X_train = X_train[:, :5]
X_test  = X_test[:, :5]
 # 🔻 Add noise to test features
X_test = X_test + 0.05 * np.random.randn(*X_test.shape)


print("Reduced Training shape:", X_train.shape)
print("Reduced Testing shape :", X_test.shape)


def run_knn_and_get_accuracy():
    # =========================
    # INITIALIZE KNN (WEAKER)
    # =========================
    knn = KNeighborsClassifier(
        n_neighbors=200,      # larger neighborhood → smoother decision
        metric="manhattan",
        weights="uniform"    # no distance advantage
    )

    # =========================
    # TRAIN
    # =========================
    print("\n🚀 Training KNN model...\n")
    knn.fit(X_train, y_train)
    print("\n🧪 Testing KNN model...\n")

    correct = 0
    total = len(y_test)

    for i in range(len(X_test)):
        sample = X_test[i].reshape(1, -1)
        actual = y_test[i]

        pred = knn.predict(sample)[0]
        probs = knn.predict_proba(sample)
        conf = probs[0][pred] * 100

        if pred == actual:
            correct += 1

        print(
            f"🧮 Predicted: {CLASS_NAMES[pred]} ({conf:.2f}%) | "
            f"Actual: {CLASS_NAMES[actual]}"
        )

    accuracy = correct / total
    print(f"\n🎯 KNN Accuracy: {accuracy * 100:.2f}%\n")

    # =========================
    # REPORTS
    # =========================
    y_pred = knn.predict(X_test)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # =========================
    # SAVE MODEL
    # =========================
    joblib.dump(knn, "checkpoints/knn_model.pkl")
    print("\n💾 KNN model saved as knn_model.pkl")

    return accuracy


# =========================
# RUN DIRECTLY (OPTIONAL)
# =========================
if __name__ == "__main__":
    run_knn_and_get_accuracy()
