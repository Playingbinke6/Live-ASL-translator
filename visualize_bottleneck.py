import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap

# Load the bottleneck features and labels (using the validation set)
val_features = np.load('DATA/Bottleneck_features/val_bottleneck_features.npy')
val_labels = np.load('DATA/Bottleneck_features/val_labels.npy')

print("Validation features shape:", val_features.shape)
print("Validation labels shape:", val_labels.shape)

# Scale the features (important for PCA, MDS, Isomap)
scaler = StandardScaler()
scaled_val_features = scaler.fit_transform(val_features)

n_components = 2
n_neighbors_isomap = 10  # Hyperparameter for Isomap

# --- t-SNE Visualization ---
print("\n--- t-SNE ---")
tsne = TSNE(n_components=n_components, random_state=42, n_iter=300, verbose=1)
tsne_embeddings = tsne.fit_transform(scaled_val_features)
print("t-SNE embedding shape:", tsne_embeddings.shape)
plt.figure(figsize=(12, 10))
sns.scatterplot(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=val_labels, palette='viridis', legend='full')
plt.title('t-SNE Visualization of Bottleneck Features (Validation Set)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# --- PCA Visualization ---
print("\n--- PCA ---")
pca = PCA(n_components=n_components, random_state=42)
pca_embeddings = pca.fit_transform(scaled_val_features)
print("PCA embedding shape:", pca_embeddings.shape)
plt.figure(figsize=(12, 10))
sns.scatterplot(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1], hue=val_labels, palette='viridis', legend='full')
plt.title('PCA Visualization of Bottleneck Features (Validation Set)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# --- UMAP Visualization ---
print("\n--- UMAP ---")
reducer = umap.UMAP(n_components=n_components, random_state=42)
umap_embeddings = reducer.fit_transform(scaled_val_features)
print("UMAP embedding shape:", umap_embeddings.shape)
plt.figure(figsize=(12, 10))
sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=val_labels, palette='viridis', legend='full')
plt.title('UMAP Visualization of Bottleneck Features (Validation Set)')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()

# --- Isomap Visualization --- WARNING TAKES A LONG TIME
print("\n--- Isomap ---")
isomap = Isomap(n_neighbors=n_neighbors_isomap, n_components=n_components)
isomap_embeddings = isomap.fit_transform(scaled_val_features)
print("Isomap embedding shape:", isomap_embeddings.shape)
plt.figure(figsize=(12, 10))
sns.scatterplot(x=isomap_embeddings[:, 0], y=isomap_embeddings[:, 1], hue=val_labels, palette='viridis', legend='full')
plt.title('Isomap Visualization of Bottleneck Features (Validation Set)')
plt.xlabel('Isomap Dimension 1')
plt.ylabel('Isomap Dimension 2')
plt.show()

print("\nFinished Dimensionality Reduction and Visualization with t-SNE, PCA, UMAP, and Isomap")