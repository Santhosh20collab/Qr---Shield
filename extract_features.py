import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops

def extract_qr_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    h, w = binary.shape

    # --- Feature: Module density ---
    black_pixel_ratio = np.sum(binary == 0) / (h * w)

    # --- Features: Transitions ---
    horizontal_transitions = sum(np.sum(np.diff(row) != 0) for row in binary) / h
    vertical_transitions = sum(np.sum(np.diff(col) != 0) for col in binary.T) / w

    # --- Feature: Quadrant symmetry ---
    mid_h, mid_w = h // 2, w // 2
    quadrants = [binary[0:mid_h, 0:mid_w], binary[0:mid_h, mid_w:w],
                 binary[mid_h:h, 0:mid_w], binary[mid_h:h, mid_w:w]]
    quadrant_densities = [np.sum(q == 0) / q.size for q in quadrants]
    symmetry_std = np.std(quadrant_densities)

    # --- Features: Run-length statistics ---
    run_lengths = []
    for row in binary:
        changes = np.where(np.diff(row) != 0)[0]
        if len(changes) > 1:
            run_lengths.extend(np.diff(changes))
    avg_run_length = np.mean(run_lengths) if run_lengths else 0
    run_length_std = np.std(run_lengths) if run_lengths else 0

    # --- Feature: Number of distinct black regions ---
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_black_regions = len(contours)

    # --- Feature: Edge density ---
    edges = cv2.Canny(binary, 100, 200)
    edge_density = np.sum(edges > 0) / (h * w)

    # --- Features: GLCM texture analysis ---
    # Measures how neighboring pixel intensities relate to each other -
    # a standard image-forensics technique for detecting subtle pattern differences
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, "contrast")[0, 0]
    glcm_homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    glcm_energy = graycoprops(glcm, "energy")[0, 0]
    glcm_correlation = graycoprops(glcm, "correlation")[0, 0]

    return {
        "black_pixel_ratio": black_pixel_ratio,
        "horizontal_transitions": horizontal_transitions,
        "vertical_transitions": vertical_transitions,
        "symmetry_std": symmetry_std,
        "avg_run_length": avg_run_length,
        "run_length_std": run_length_std,
        "num_black_regions": num_black_regions,
        "edge_density": edge_density,
        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity,
        "glcm_energy": glcm_energy,
        "glcm_correlation": glcm_correlation,
    }

# --- Load metadata and process every image ---
metadata = pd.read_csv("dataset/metadata.csv")

feature_rows = []
for _, row in metadata.iterrows():
    image_path = os.path.join("dataset", row["label"], row["filename"])
    features = extract_qr_features(image_path)
    features["filename"] = row["filename"]
    features["label"] = row["label"]
    features["payload_type"] = row["payload_type"]
    features["qr_version"] = row["qr_version"]
    features["error_correction"] = row["error_correction"]
    feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)

# Convert error_correction letter (L/M/Q/H) into an ordinal number the model can use
ec_map = {"L": 0, "M": 1, "Q": 2, "H": 3}
features_df["error_correction_level"] = features_df["error_correction"].map(ec_map)

features_df.to_csv("dataset/features.csv", index=False)

print(features_df.head())
print(f"\nSaved {len(features_df)} rows to dataset/features.csv")