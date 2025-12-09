import umap
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from collections import Counter
import datetime

# --- Helpers ---

def get_current_day_folder(base_dir=r"Z:\\TrackerData"):
    """
    Returns the folder path for today based on the Z:\\TrackerData\\YYYY\\MM\\DD structure.
    """
    now = datetime.datetime.now()
    
    # Matches Tracker.py logic: Year / Month (02d) / Day (02d)
    return os.path.join(
        base_dir, 
        str(now.year), 
        f"{now.month:02d}", 
        f"{now.day:02d}"
    )

def get_rgb_from_vectors(vectors):
    """Normalizes a (N, 3) numpy array of vectors to 0-255 integers."""
    coords = vectors.copy()
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1e-5 
    normalized = (coords - min_vals) / ranges
    return (normalized * 255).astype(int)

def generate_cluster_labels(art_data, n_clusters=8):
    """
    Groups points into clusters based on 3D position and names the cluster
    after the most common app used in that area.
    """
    if len(art_data) < n_clusters:
        n_clusters = len(art_data)

    # 1. Extract coordinates for clustering
    coords = np.array([[p['x'], p['y'], p['z']] for p in art_data])

    # 2. Perform K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(coords)

    # 3. Analyze each cluster to find the "Dominant App"
    cluster_info = {}
    
    for cluster_id in range(n_clusters):
        # Get indices of points in this cluster
        indices = np.where(labels == cluster_id)[0]
        
        # Gather all app names in this cluster
        labels_in_cluster = [art_data[i]['window_title'] for i in indices]
        
        # Find the mode (most common title/app)
        if labels_in_cluster:
            most_common = Counter(labels_in_cluster).most_common(1)[0][0]
        else:
            most_common = "Unknown"
            
        # Calculate centroid (average center) for label placement
        centroid = coords[indices].mean(axis=0)
        
        cluster_info[cluster_id] = {
            "label": most_common,
            "centroid": centroid.tolist(),
            "count": len(indices)
        }

    # 4. Attach cluster IDs back to the original points
    for i, point in enumerate(art_data):
        point['cluster_id'] = int(labels[i])

    return art_data, cluster_info

# --- Main Pipeline ---

def parse_journal_for_art(day_folder_path, n_dim=3):
    """
    Full pipeline: Load -> UMAP -> Normalize Colors -> Cluster -> Label.
    """
    if not os.path.exists(day_folder_path):
        return [], {}

    # 1. Load Data
    valid_events = []
    files = sorted([f for f in os.listdir(day_folder_path) if f.endswith('.json')])
    
    for filename in files:
        try:
            filepath = os.path.join(day_folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure we have both embeddings
                if (data.get("text_embedding") and len(data["text_embedding"]) > 0 and 
                    data.get("image_embedding") and len(data["image_embedding"]) > 0):
                    valid_events.append(data)
        except:
            continue

    if len(valid_events) < 5:
        print("Waiting for more data points (need > 5)...")
        return [], {}

    # 2. Extract Matrices
    text_matrix = np.array([e["text_embedding"] for e in valid_events])
    image_matrix = np.array([e["image_embedding"] for e in valid_events])

    # 3. UMAP Reduction
    # Position based on Text (Meaning)
    pos_reducer = umap.UMAP(n_components=n_dim)
    positions = pos_reducer.fit_transform(text_matrix)

    # Color based on Image (Vibe)
    color_reducer = umap.UMAP(n_components=3)
    raw_colors = color_reducer.fit_transform(image_matrix)
    rgb_values = get_rgb_from_vectors(raw_colors)

    # 4. Build Initial Points List
    art_data = []
    for i, event in enumerate(valid_events):
        intensity = event.get("keys_pressed", 0) + event.get("mouse_activity", 0)
        
        art_data.append({
            "x": float(positions[i][0]),
            "y": float(positions[i][1]),
            "z": float(positions[i][2]) if n_dim == 3 else 0,
            "r": int(rgb_values[i][0]),
            "g": int(rgb_values[i][1]),
            "b": int(rgb_values[i][2]),
            "intensity": intensity,
            "timestamp": event.get("timestamp"),
            "app_name": event.get("app_name", "Unknown"),
            "window_title": event.get("window_title", "Unknown")
        })

    # 5. Cluster and Label
    # We dynamically choose K based on data size (sqrt of N is a decent heuristic)
    k = int(np.sqrt(len(art_data))) 
    k = max(2, min(k, 10)) # Keep K between 2 and 10 for readability
    
    labeled_points, cluster_meta = generate_cluster_labels(art_data, n_clusters=k)
    
    return labeled_points, cluster_meta

# Path to today's data (or whatever you get from get_current_day_folder)
folder = r"Z:\\TrackerData\\2025\\12\\09" 

# Get the list of points
points = parse_journal_for_art(folder)

# Inspect the first point to verify
if points:
    print(points[:5])