import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# =========================
# LOAD FEATURES
# =========================
X_train = np.load("features/X_train.npy")
y_train = np.load("features/y_train.npy")
X_test  = np.load("features/X_test.npy")
y_test  = np.load("features/y_test.npy")

def run_svm_and_get_accuracy():
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )

    print("\n🚀 Training SVM...\n")
    svm.fit(X_train, y_train)

    print("🧪 Testing SVM...\n")
    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 SVM Accuracy: {acc * 100:.2f}%")

    joblib.dump(svm, "checkpoints/svm_model.pkl")
    print("💾 SVM model saved")

    return acc


if __name__ == "__main__":
    run_svm_and_get_accuracy()
