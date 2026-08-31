import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load test data and model
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
model = keras.models.load_model("models/pneumonia_cnn.h5")

# Predict
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Flatten
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

# Metrics
test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
accuracy = accuracy_score(y_test_flat, y_pred_flat)
precision = precision_score(y_test_flat, y_pred_flat)
recall = recall_score(y_test_flat, y_pred_flat)
f1 = f1_score(y_test_flat, y_pred_flat)

# Results
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_flat, y_pred_flat, target_names=['Normal', 'Pneumonia'], digits=4))