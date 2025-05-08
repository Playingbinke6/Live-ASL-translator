import os
import matplotlib.pyplot as plt
import random
import cv2

# Path to the dataset
data_dir = r'DATA/ML-ASL-raw/asl_alphabet_train/asl_alphabet_train'

# Get the list of labels (directories)
labels = sorted(os.listdir(data_dir))
print("Number of labels:", len(labels))
print("Labels:", labels)

# Count the number of images per label and display samples
num_samples = 5
plt.figure(figsize=(15, 15))
for i, label in enumerate(labels):
    label_dir = os.path.join(data_dir, label)
    image_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.jpg')]
    num_images = len(image_files)
    print(f"Label: {label}, Number of images: {num_images}")

    # Display a few random samples
    random_samples = random.sample(image_files, min(num_samples, num_images))
    for j, img_path in enumerate(random_samples):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
        plt.subplot(len(labels), num_samples, i * num_samples + j + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')

plt.tight_layout()
plt.show()