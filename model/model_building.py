import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------
# Load dataset
# -------------------------
columns = [
    "ID", "Diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

dataset = pd.read_csv("wdbc.data", header=None, names=columns)

# Encode target
dataset["Diagnosis"] = dataset["Diagnosis"].map({"B": 0, "M": 1})

# -------------------------
# Select 5 features
# -------------------------
selected_features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
X = dataset[selected_features]
y = dataset["Diagnosis"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# Build ANN model
# -------------------------
ann_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# -------------------------
# Evaluate model
# -------------------------
pred_prob = ann_model.predict(X_test)
pred_class = (pred_prob > 0.5).astype(int)

metrics = {
    "Accuracy": accuracy_score(y_test, pred_class),
    "Precision": precision_score(y_test, pred_class),
    "Recall": recall_score(y_test, pred_class),
    "F1-score": f1_score(y_test, pred_class),
    "AUC": roc_auc_score(y_test, pred_prob)
}

print("ANN Model Metrics:", metrics)

# -------------------------
# Save model, scaler, and feature columns
# -------------------------
ann_model.save("model/breast_cancer_ann_model.h5")
joblib.dump(scaler, "model/breast_cancer_scaler.joblib")
joblib.dump(selected_features, "model/breast_cancer_columns.joblib")
print("Model, scaler, and feature columns saved successfully!")