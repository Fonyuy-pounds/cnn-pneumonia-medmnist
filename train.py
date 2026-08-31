(
echo import os
echo import numpy as np
echo import tensorflow as tf
echo from tensorflow import keras
echo from model import create_pneumonia_cnn
echo.
echo np.random.seed^(42^)
echo tf.random.set_seed^(42^)
echo.
echo def main^(^):
echo     print^(^"Loading data...^"^)
echo     X_train = np.load^(^"data/X_train.npy^"^)
echo     y_train = np.load^(^"data/y_train.npy^"^)
echo     X_val = np.load^(^"data/X_val.npy^"^)
echo     y_val = np.load^(^"data/y_val.npy^"^)
echo.
echo     model = create_pneumonia_cnn^(^)
echo.
echo     callbacks = [
echo         keras.callbacks.EarlyStopping^(monitor=^"val_loss^", patience=5, restore_best_weights=True, verbose=1^),
echo         keras.callbacks.ReduceLROnPlateau^(monitor=^"val_loss^", factor=0.5, patience=3, min_lr=1e-6, verbose=1^)
echo     ]
echo.
echo     history = model.fit^(X_train, y_train, batch_size=32, epochs=35, validation_data=^(X_val, y_val^), callbacks=callbacks, verbose=1^)
echo.
echo     os.makedirs^(^"models^", exist_ok=True^)
echo     model.save^(^"models/pneumonia_cnn.h5^"^)
echo     print^(^"Model saved!^"^)
echo.
echo if __name__ == ^"__main__^":
echo     main^(^)
) > train.py