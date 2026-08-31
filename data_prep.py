import numpy as np
from medmnist import PneumoniaMNIST

np.random.seed(42)

print("=" * 60)
print("PNEUMONIAMNIST DATA PREPARATION")
print("=" * 60)

# Load dataset splits
print("\n[1/3] Loading PneumoniaMNIST dataset...")
train_dataset = PneumoniaMNIST(split="train", download=True)
val_dataset = PneumoniaMNIST(split="val", download=True)
test_dataset = PneumoniaMNIST(split="test", download=True)

X_train_raw = train_dataset.imgs
y_train_raw = train_dataset.labels
X_val_raw = val_dataset.imgs
y_val_raw = val_dataset.labels
X_test_raw = test_dataset.imgs
y_test_raw = test_dataset.labels

print("Dataset loaded successfully")

# Inspect dataset properties
print("\n[2/3] Dataset Inspection")
print("-" * 60)

print("Image Properties:")
print(f"  Dimensions: {X_train_raw.shape[1:3]}")
print(f"  Channels: {X_train_raw.shape[-1]}")
print(f"  Type: {'Grayscale' if X_train_raw.shape[-1] == 1 else 'RGB'}")

print("\nDataset Splits:")
print(f"  Training:   {X_train_raw.shape[0]:,} samples")
print(f"  Validation: {X_val_raw.shape[0]:,} samples")
print(f"  Test:       {X_test_raw.shape[0]:,} samples")

print("\nPixel Values:")
print(f"  Range: [{X_train_raw.min()}, {X_train_raw.max()}]")
print(f"  Dtype: {X_train_raw.dtype}")

print("\nLabel Structure:")
print(f"  Shape: {y_train_raw.shape}")
print(f"  Unique labels: {np.unique(y_train_raw)}")
print(f"  Dtype: {y_train_raw.dtype}")

print("\nClass Distribution (Training):")
unique, counts = np.unique(y_train_raw, return_counts=True)
for label, count in zip(unique, counts):
    percentage = (count / len(y_train_raw)) * 100
    class_name = "Pneumonia" if label == 1 else "Normal"
    print(f"  {class_name} ({label}): {count:,} samples ({percentage:.1f}%)")

# Data integrity check
print("\n[3/3] Data Integrity Check")
print("-" * 60)
print(f"  X_train: {X_train_raw.shape}")
print(f"  y_train: {y_train_raw.shape}")
print(f"  X_val:   {X_val_raw.shape}")
print(f"  y_val:   {y_val_raw.shape}")
print(f"  X_test:  {X_test_raw.shape}")
print(f"  y_test:  {y_test_raw.shape}")

print("\n" + "=" * 60)
print("DATA LOADING COMPLETE")
print("=" * 60)