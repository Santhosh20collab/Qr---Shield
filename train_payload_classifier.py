import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("dataset/features.csv")

# Best-performing feature set for payload-type classification (GLCM DID help here)
feature_columns = [
    "black_pixel_ratio", "horizontal_transitions", "vertical_transitions",
    "symmetry_std", "avg_run_length", "run_length_std",
    "num_black_regions", "edge_density",
    "qr_version", "error_correction_level",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
]
X = df[feature_columns]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["payload_type"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"=== FINAL Module 2 Model: LightGBM ===")
print(f"Payload-type classification accuracy: {accuracy:.4f}")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\n--- Feature Importance ---")
print(importance_df.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance")
plt.title("Module 2: Payload-Type Classification - Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("module2_feature_importance.png")

joblib.dump(model, "payload_type_model.pkl")
joblib.dump(label_encoder, "payload_type_label_encoder.pkl")
print("\nFinal Module 2 model saved as payload_type_model.pkl")