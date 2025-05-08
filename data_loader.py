import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

# Path to the main dataset directory
data_dir = r'DATA/ML-ASL-raw'  # Replace with the actual path
train_dir = os.path.join(data_dir, 'asl_alphabet_train/asl_alphabet_train')
test_dir = os.path.join(data_dir, 'asl_alphabet_test')
image_size = 64
random_seed = 42  # For reproducibility

def load_and_preprocess_image(img_path, target_size=(image_size, image_size)):
    """Loads, resizes, and normalizes an image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    return img_normalized

def create_dataset_from_dir(data_dir):
    """Creates lists of images and labels from a directory."""
    images = []
    labels = []
    label_to_index = {label: i for i, label in enumerate(sorted(os.listdir(data_dir)))}
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(label_dir):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(label_dir, img_file)
                img = load_and_preprocess_image(img_path)
                images.append(img)
                labels.append(label_to_index[label])
    return np.array(images), np.array(labels), label_to_index

# Load training data
train_images, train_labels, label_to_index = create_dataset_from_dir(train_dir)
index_to_label = {v: k for k, v in label_to_index.items()}

# Split training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=random_seed, stratify=train_labels
)

# Load test data
test_images, test_labels, _ = create_dataset_from_dir(test_dir) # We don't need label_to_index again

# Save the processed data
np.save(r'DATA/data_labels/train_images.npy', train_images)
np.save(r'DATA/data_labels/train_labels.npy', train_labels)
np.save(r'DATA/data_labels/val_images.npy', val_images)
np.save(r'DATA/data_labels/val_labels.npy', val_labels)
np.save(r'DATA/data_labels/test_images.npy', test_images)
np.save(r'DATA/data_labels/test_labels.npy', test_labels)

print("Processed data saved!")

# --- Custom Dataset Class for PyTorch (Load from saved arrays) ---
class ASLDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Load from saved arrays
loaded_train_images = np.load(r'DATA/data_labels/train_images.npy')
loaded_train_labels = np.load(r'DATA/data_labels/train_labels.npy')
loaded_val_images = np.load(r'DATA/data_labels/val_images.npy')
loaded_val_labels = np.load(r'DATA/data_labels/val_labels.npy')
loaded_test_images = np.load(r'DATA/data_labels/test_images.npy')
loaded_test_labels = np.load(r'DATA/data_labels/test_labels.npy')

# Create DataLoaders
batch_size = 64
train_dataset = ASLDataset(loaded_train_images, loaded_train_labels)
val_dataset = ASLDataset(loaded_val_images, loaded_val_labels)
test_dataset = ASLDataset(loaded_test_images, loaded_test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("DataLoaders created from saved arrays.")
print("Number of batches in train_loader:", len(train_loader))
print("Number of batches in val_loader:", len(val_loader))
print("Number of batches in test_loader:", len(test_loader))