(
echo import os
echo import numpy as np
echo from medmnist import PneumoniaMNIST
echo.
echo np.random.seed^(42^)
echo.
echo def main^(^):
echo     print^(^"Loading dataset...^"^)
echo     train_dataset = PneumoniaMNIST^(split=^"train^", download=True^)
echo     val_dataset = PneumoniaMNIST^(split=^"val^", download=True^)
echo     test_dataset = PneumoniaMNIST^(split=^"test^", download=True^)
echo.
echo     X_train = train_dataset.imgs.reshape^(-1, 28, 28, 1^).astype^(np.float32^) / 255.0
echo     X_val = val_dataset.imgs.reshape^(-1, 28, 28, 1^).astype^(np.float32^) / 255.0
echo     X_test = test_dataset.imgs.reshape^(-1, 28, 28, 1^).astype^(np.float32^) / 255.0
echo.
echo     y_train = train_dataset.labels.astype^(np.float32^)
echo     y_val = val_dataset.labels.astype^(np.float32^)
echo     y_test = test_dataset.labels.astype^(np.float32^)
echo.
echo     os.makedirs^(^"data^", exist_ok=True^)
echo     np.save^(^"data/X_train.npy^", X_train^)
echo     np.save^(^"data/y_train.npy^", y_train^)
echo     np.save^(^"data/X_val.npy^", X_val^)
echo     np.save^(^"data/y_val.npy^", y_val^)
echo     np.save^(^"data/X_test.npy^", X_test^)
echo     np.save^(^"data/y_test.npy^", y_test^)
echo     print^(^"Data preparation complete!^"^)
echo.
echo if __name__ == ^"__main__^":
echo     main^(^)
) > data_prep.py