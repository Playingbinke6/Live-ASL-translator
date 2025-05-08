import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from data_loader import train_loader, val_loader, test_loader
import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- Function to extract features (flatten images) and labels from DataLoader ---
def extract_features_labels(dataloader):
    all_features = []
    all_labels = []
    for images, labels in dataloader:
        # Flatten the images: (batch_size, channels, height, width) -> (batch_size, channels * height * width)
        features = images.view(images.size(0), -1).numpy()
        labels = labels.numpy()
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

# Extract features and labels
train_features, train_labels = extract_features_labels(train_loader)
val_features, val_labels = extract_features_labels(val_loader)
test_features, test_labels = extract_features_labels(test_loader)

print("Training features shape:", train_features.shape)
print("Training labels shape:", train_labels.shape)
print("Validation features shape:", val_features.shape)
print("Validation labels shape:", val_labels.shape)
print("Test features shape:", test_features.shape)
print("Test labels shape:", test_labels.shape)

# --- Train Logistic Regression ---
print("\n--- Training Logistic Regression ---")
logistic_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', multi_class='ovr')
logistic_model.fit(train_features, train_labels)

# Evaluate on validation set
val_preds_logistic = logistic_model.predict(val_features)
val_accuracy_logistic = accuracy_score(val_labels, val_preds_logistic)
print(f"Validation Accuracy (Logistic Regression): {val_accuracy_logistic:.4f}")
print("Validation Classification Report (Logistic Regression):\n", classification_report(val_labels, val_preds_logistic, zero_division=0))

# Evaluate on test set (note: small size)
test_preds_logistic = logistic_model.predict(test_features)
test_accuracy_logistic = accuracy_score(test_labels, test_preds_logistic)
print(f"Test Accuracy (Logistic Regression): {test_accuracy_logistic:.4f} (Note: Test set is very small)")
print("Test Classification Report (Logistic Regression):\n", classification_report(test_labels, test_preds_logistic, zero_division=0))

# --- Train Support Vector Machine (SVM) ---
print("\n--- Training Support Vector Machine (SVM) ---")
svm_model = SVC(random_state=42, kernel='linear') # Linear kernel for initial benchmark
svm_model.fit(train_features, train_labels)

# Evaluate on validation set
val_preds_svm = svm_model.predict(val_features)
val_accuracy_svm = accuracy_score(val_labels, val_preds_svm)
print(f"Validation Accuracy (SVM): {val_accuracy_svm:.4f}")
print("Validation Classification Report (SVM):\n", classification_report(val_labels, val_preds_svm, zero_division=0))

# Evaluate on test set (note: small size)
test_preds_svm = svm_model.predict(test_features)
test_accuracy_svm = accuracy_score(test_labels, test_preds_svm)
print(f"Test Accuracy (SVM): {test_accuracy_svm:.4f} (Note: Test set is very small)")
print("Test Classification Report (SVM):\n", classification_report(test_labels, test_preds_svm, zero_division=0))
