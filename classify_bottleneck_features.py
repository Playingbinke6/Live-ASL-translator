import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load the bottleneck features and labels
train_features = np.load(r'DATA/Bottleneck_features/train_bottleneck_features.npy')
train_labels = np.load(r'DATA/Bottleneck_features/train_labels.npy')
val_features = np.load(r'DATA/Bottleneck_features/val_bottleneck_features.npy')
val_labels = np.load(r'DATA/Bottleneck_features/val_labels.npy')

# Scale the features (important for distance-based algorithms)
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)

n_clusters = 29 # For algorithms that require it

def map_clusters_to_labels(cluster_labels, true_labels, n_clusters):
    cluster_label_map = {}
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) > 0:
            most_common_label = Counter(true_labels[cluster_indices]).most_common(1)[0][0]
            cluster_label_map[i] = most_common_label
        else:
            cluster_label_map[i] = -1 # Handle empty clusters
    return cluster_label_map

# --- K-means Clustering ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
kmeans_labels_val = kmeans.fit_predict(val_features_scaled)
kmeans_cluster_to_label = map_clusters_to_labels(kmeans_labels_val, val_labels, n_clusters)
kmeans_predicted_labels = [kmeans_cluster_to_label.get(cluster, -1) for cluster in kmeans_labels_val]
kmeans_accuracy = accuracy_score(val_labels, kmeans_predicted_labels)
print(f"\nK-means Validation Accuracy (after mapping): {kmeans_accuracy:.4f}")

# --- DBSCAN Clustering ---
dbscan = DBSCAN(eps=1., min_samples=1) # Initial hyperparameters, adjust as needed
dbscan_labels_val = dbscan.fit_predict(val_features_scaled)
n_dbscan_clusters = len(np.unique(dbscan_labels_val)) - (1 if -1 in dbscan_labels_val else 0) # Exclude noise
print(f"\nDBSCAN found {n_dbscan_clusters} clusters (excluding noise).")
dbscan_cluster_to_label = map_clusters_to_labels(dbscan_labels_val, val_labels, n_dbscan_clusters)
dbscan_predicted_labels = [dbscan_cluster_to_label.get(cluster, -1) for cluster in dbscan_labels_val]
dbscan_accuracy = accuracy_score(val_labels, dbscan_predicted_labels)
print(f"DBSCAN Validation Accuracy (after mapping): {dbscan_accuracy:.4f}")

# --- Agglomerative Clustering ---
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
agg_labels_val = agg_clustering.fit_predict(val_features_scaled)
agg_cluster_to_label = map_clusters_to_labels(agg_labels_val, val_labels, n_clusters)
agg_predicted_labels = [agg_cluster_to_label.get(cluster, -1) for cluster in agg_labels_val]
agg_accuracy = accuracy_score(val_labels, agg_predicted_labels)
print(f"\nAgglomerative Clustering Validation Accuracy (after mapping): {agg_accuracy:.4f}")

# --- Logistic Regression ---
# Initialize a Logistic Regression model
logistic_model = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')

# Train the Logistic Regression model
print("\n--- Training Logistic Regression on Bottleneck Features ---")
logistic_model.fit(train_features, train_labels)

# Make predictions on the validation set
val_preds = logistic_model.predict(val_features)

# Evaluate the performance
accuracy = accuracy_score(val_labels, val_preds)
print(f"Validation Accuracy: {accuracy:.4f}")


print("\nFinished Clustering with Multiple Methods")