import umap
import numpy as np
import os
import json
import datetime

# --- Helpers ---

def get_current_day_folder(base_dir=r"Z:\\TrackerData"):
    """
    Returns the folder path for today based on the Z:\\TrackerData\\YYYY\\MM\\DD structure.
    """
    now = datetime.datetime.now()
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

# --- Main Pipeline ---

def parse_journal_for_art(day_folder_path, n_dim=2):
    """
    Full pipeline: Load (JSONL or JSONs) -> UMAP -> Normalize Colors.
    
    Args:
        day_folder_path (str): Path to the day's data folder.
        n_dim (int): Dimensions to reduce text embeddings to (2 for maps, 3 for 3D viz).
    """
    if not os.path.exists(day_folder_path):
        print(f"Path does not exist: {day_folder_path}")
        return []

    valid_events = []
    
    # --- 1. Load Data (Dual Mode) ---
    jsonl_path = os.path.join(day_folder_path, "data.jsonl")
    
    if os.path.exists(jsonl_path):
        # MODE A: New "Single File" Format
        print(f"Reading from unified file: {jsonl_path}")
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if (data.get("text_embedding") and len(data["text_embedding"]) > 0 and 
                            data.get("image_embedding") and len(data["image_embedding"]) > 0):
                            valid_events.append(data)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading JSONL: {e}")
    else:
        print("JSONL file not found.")
        return []

    if len(valid_events) < 5:
        print(f"Not enough data points to generate art (Found {len(valid_events)}, need > 5).")
        return []

    print(f"Processing {len(valid_events)} valid data points...")

    # --- 2. Extract Matrices ---
    text_matrix = np.array([e["text_embedding"] for e in valid_events])
    image_matrix = np.array([e["image_embedding"] for e in valid_events])

    # --- 3. UMAP Reduction ---
    print("Running UMAP on text embeddings (Position)...")
    # Position based on Text (Meaning)
    pos_reducer = umap.UMAP(n_components=n_dim, random_state=42)
    positions = pos_reducer.fit_transform(text_matrix)

    print("Running UMAP on image embeddings (Color)...")
    # Color based on Image (Vibe) - Always 3D for RGB
    color_reducer = umap.UMAP(n_components=3, random_state=42)
    raw_colors = color_reducer.fit_transform(image_matrix)
    rgb_values = get_rgb_from_vectors(raw_colors)

    # --- 4. Build Initial Points List ---
    art_data = []
    for i, event in enumerate(valid_events):
        # Calculate Intensity (Keys + Mouse)
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

    return art_data

if __name__ == "__main__":
    # Test run
    test_folder = get_current_day_folder()
    points, _ = parse_journal_for_art(test_folder)
    if points:
        print(f"Success! Generated {len(points)} points.")
        print(f"Sample: {points[0]}")