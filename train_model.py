import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("dataset/features.csv")
df["label_num"] = df["label"].map({"legitimate": 0, "phishing": 1})

# Best-performing feature set for phishing detection (GLCM did NOT help here)
feature_columns = [
    "black_pixel_ratio", "horizontal_transitions", "vertical_transitions",
    "symmetry_std", "avg_run_length", "run_length_std",
    "num_black_regions", "edge_density",
    "qr_version", "error_correction_level",
]

X = df[feature_columns]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Best-performing model for this task
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss")
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

auc = roc_auc_score(y_test, probs)
print(f"=== FINAL Module 3 Model: XGBoost ===")
print(f"AUC Score: {auc:.4f}")
print(f"Base paper benchmark: 0.9133")
print(classification_report(y_test, preds, target_names=["legitimate", "phishing"]))

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
plt.title("Module 3: Phishing Detection - Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("module3_feature_importance.png")

joblib.dump(model, "qr_shield_baseline_model.pkl")
print("\nFinal Module 3 model saved as qr_shield_baseline_model.pkl")