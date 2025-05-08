import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from autoencoder import ConvAutoencoderV3
from data_loader import train_loader, val_loader # Ensure your DataLoaders are correctly set up
import numpy as np
import os

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the full autoencoder model
autoencoder = ConvAutoencoderV3().to(device)

# Load the state dictionary of the trained autoencoder
model_path = 'models/conv_autoencoder.pth'
if os.path.exists(model_path):
    autoencoder.load_state_dict(torch.load(model_path))
    print(f"Loaded trained autoencoder from {model_path}")
else:
    print(f"Error: Trained autoencoder not found at {model_path}. Make sure to run train_autoencoder.py first.")
    exit()

# Extract the encoder part of the model
encoder = autoencoder.encoder
flatten = autoencoder.flatten
fc_encode = autoencoder.fc_encode
encoder.eval() # Set encoder to evaluation mode

def extract_features(dataloader):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            encoded = encoder(images)
            batch_size = encoded.size(0)
            flattened = flatten(encoded)
            bottleneck_features = fc_encode(flattened).cpu().numpy()
            all_features.append(bottleneck_features)
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

# Extract features from the training set
train_bottleneck_features, train_labels = extract_features(train_loader)
print("Extracted bottleneck features from training set:", train_bottleneck_features.shape)
print("Training labels shape:", train_labels.shape)

# Extract features from the validation set
val_bottleneck_features, val_labels = extract_features(val_loader)
print("Extracted bottleneck features from validation set:", val_bottleneck_features.shape)
print("Validation labels shape:", val_labels.shape)

# Save the extracted features and labels
np.save(r'DATA/Bottleneck_features/train_bottleneck_features.npy', train_bottleneck_features)
np.save(r'DATA/Bottleneck_features/train_labels.npy', train_labels)
np.save(r'DATA/Bottleneck_features/val_bottleneck_features.npy', val_bottleneck_features)
np.save(r'DATA/Bottleneck_features/val_labels.npy', val_labels)

print("Bottleneck features and labels saved!")