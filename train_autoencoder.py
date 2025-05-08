import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from autoencoder import ConvAutoencoderV3
from data_loader import train_loader, val_loader
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model and move it to the device
model = ConvAutoencoderV3().to(device)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 60 # Adjustable

# Training loop
for epoch in range(num_epochs):
    train_loss = 0.0
    for batch_idx, (images, _) in enumerate(train_loader):
        # Move images to the device
        images = images.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(images)

        # Calculate the loss
        loss = criterion(outputs, images)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    # Calculate average training loss for the epoch
    train_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}")

    # Validation loop
    val_loss = 0.0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

torch.save(model.state_dict(), r'models/conv_autoencoder.pth')
print("Finished Training")


model.eval() # Set to evaluation mode

# Take a batch of images from the validation set
dataiter = iter(val_loader)
images, labels = next(dataiter)
images = images.to(device)

# Perform a forward pass to get the reconstructions
reconstructed, _ = model(images)

# Move the images and reconstructions back to the CPU and detach
images = images.cpu().numpy()
reconstructed = reconstructed.cpu().detach().numpy()

def show_numpy_image(ax, image_np):
    # Assuming image_np is (batch, C, H, W) - we'll take a single image
    img = np.transpose(image_np, (1, 2, 0)) # From CHW to HWC
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.axis('off')

n_samples = 5
fig, axes = plt.subplots(nrows=2, ncols=n_samples, figsize=(10, 4))

for i in range(n_samples):
    # Original images
    show_numpy_image(axes[0, i], images[i])
    axes[0, i].set_title("Original")

    # Reconstructed images
    show_numpy_image(axes[1, i], reconstructed[i])
    axes[1, i].set_title("Reconstructed")

plt.tight_layout()
plt.show()

print("Finished Visualization")