(
echo import numpy as np
echo from tensorflow import keras
echo from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
echo import matplotlib.pyplot as plt
echo import seaborn as sns
echo.
echo def main^(^):
echo     print^(^"Loading test data...^"^)
echo     X_test = np.load^(^"data/X_test.npy^"^)
echo     y_test = np.load^(^"data/y_test.npy^"^)
echo.
echo     model = keras.models.load_model^(^"models/pneumonia_cnn.h5^"^)
echo.
echo     y_pred_proba = model.predict^(X_test, verbose=0^)
echo     y_pred = ^(y_pred_proba ^>= 0.5^).astype^(int^)
echo.
echo     y_test_flat = y_test.flatten^(^)
echo     y_pred_flat = y_pred.flatten^(^)
echo.
echo     accuracy = accuracy_score^(y_test_flat, y_pred_flat^)
echo     precision = precision_score^(y_test_flat, y_pred_flat^)
echo     recall = recall_score^(y_test_flat, y_pred_flat^)
echo     f1 = f1_score^(y_test_flat, y_pred_flat^)
echo.
echo     print^(f^"Accuracy: {accuracy:.4f}^"^)
echo     print^(f^"Precision: {precision:.4f}^"^)
echo     print^(f^"Recall: {recall:.4f}^"^)
echo     print^(f^"F1-Score: {f1:.4f}^"^)
echo.
echo     cm = confusion_matrix^(y_test_flat, y_pred_flat^)
echo     plt.figure^(figsize=^(8, 6^)^)
echo     sns.heatmap^(cm, annot=True, fmt=^"d^", cmap=^"Blues^", xticklabels=[^"Normal^", ^"Pneumonia^"], yticklabels=[^"Normal^", ^"Pneumonia^"]^)
echo     plt.title^(^"Confusion Matrix^"^)
echo     plt.xlabel^(^"Predicted^"^)
echo     plt.ylabel^(^"Actual^"^)
echo     plt.tight_layout^(^)
echo     plt.savefig^(^"plots/confusion_matrix.png^", dpi=150^)
echo     plt.show^(^)
echo.
echo if __name__ == ^"__main__^":
echo     main^(^)
) > evaluate.py