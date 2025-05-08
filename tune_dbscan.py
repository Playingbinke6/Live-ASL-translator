import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.preprocessing import StandardScaler

# Load the bottleneck features and labels
val_features = np.load(r'DATA/Bottleneck_features/val_bottleneck_features.npy')
val_labels = np.load(r'DATA/Bottleneck_features/val_labels.npy')

# Scale the features
scaler = StandardScaler()
scaled_val_features = scaler.fit_transform(val_features)

def map_clusters_to_labels(cluster_labels, true_labels, n_clusters):
    cluster_label_map = {}
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        if cluster_id != -1:  # Ignore noise
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                most_common_label = Counter(true_labels[cluster_indices]).most_common(1)[0][0]
                cluster_label_map[cluster_id] = most_common_label
            else:
                cluster_label_map[cluster_id] = -1
    return cluster_label_map

# Example starting point based on your previous best
eps_start = 1 # Start at best eps
min_samples_start = 1 # Start at best min_samples

eps_range = np.arange(eps_start, 5.1, 0.5) # Explore larger eps values
min_samples_range = [min_samples_start, min_samples_start + 5, min_samples_start + 10, 20]

best_accuracy = 0
best_n_clusters = 8227 # Initialize with the high number
best_eps = None
best_min_samples = None

print("--- DBSCAN Hyperparameter Tuning (Reducing Clusters) ---")

for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_val_features)
        n_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        if n_clusters > 0:
            cluster_to_label = map_clusters_to_labels(cluster_labels, val_labels, n_clusters)
            predicted_labels = [cluster_to_label.get(cluster, -1) for cluster in cluster_labels]
            accuracy = accuracy_score(val_labels, predicted_labels)

            print(f"eps={eps:.1f}, min_samples={min_samples}: Accuracy={accuracy:.4f}, Clusters={n_clusters}")

            if accuracy > 0.5 and n_clusters < best_n_clusters: # Aim for reasonable accuracy and fewer clusters
                best_accuracy = accuracy
                best_n_clusters = n_clusters
                best_eps = eps
                best_min_samples = min_samples
            elif best_accuracy == 0 and accuracy > 0: # If no good result yet, take the first one with some accuracy
                best_accuracy = accuracy
                best_n_clusters = n_clusters
                best_eps = eps
                best_min_samples = min_samples


print("\n--- Best DBSCAN Hyperparameters Found (Attempting Fewer Clusters) ---")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Number of Clusters: {best_n_clusters}")
print(f"Best eps: {best_eps}")
print(f"Best min_samples: {best_min_samples}")