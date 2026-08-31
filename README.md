```markdown
# CNN for PneumoniaMNIST — Lightweight Binary Classification

## Project Overview

This project implements a **lightweight Convolutional Neural Network (CNN)** for binary classification of chest X-ray images using the **PneumoniaMNIST** dataset from the MedMNIST collection. The model distinguishes between **Normal** and **Pneumonia** cases with a deliberate focus on **high recall (sensitivity)** to minimize missed pneumonia diagnoses.

The architecture is intentionally lightweight — only **5,601 total parameters** — making it efficient, fast to train, and suitable for deployment in resource-constrained medical environments.

---

## Dataset

**Source:** [MedMNIST — PneumoniaMNIST](https://medmnist.com/)

| Property | Value |
|----------|-------|
| Image size | 28 × 28 pixels |
| Channels | 1 (grayscale) |
| Classes | 2 (Normal, Pneumonia) |
| Training samples | 4,708 |
| Validation samples | 524 |
| Test samples | 624 |
| Pixel range (raw) | 0 – 255 |
| Pixel range (normalized) | 0 – 1 |

**Class Distribution (Training Set):**
- Normal: 1,214 samples (25.8%)
- Pneumonia: 3,494 samples (74.2%)

---

## Model Architecture

```

Input (28, 28, 1)
  ↓
Conv2D (16 filters, 3×3, ReLU, valid padding)
  ↓
MaxPooling2D (2×2)
  ↓
Conv2D (32 filters, 3×3, ReLU, valid padding)
  ↓
MaxPooling2D (2×2)
  ↓
Flatten
  ↓
Dense (1 unit, sigmoid)

```

### Layer Details

| Layer | Type | Output Shape | Parameters |
|-------|------|--------------|------------|
| Input | InputLayer | (28, 28, 1) | 0 |
| conv1 | Conv2D (16 filters, 3×3) | (26, 26, 16) | 160 |
| pool1 | MaxPooling2D (2×2) | (13, 13, 16) | 0 |
| conv2 | Conv2D (32 filters, 3×3) | (11, 11, 32) | 4,640 |
| pool2 | MaxPooling2D (2×2) | (5, 5, 32) | 0 |
| flatten | Flatten | (800,) | 0 |
| output | Dense (1, sigmoid) | (1,) | 801 |

**Total trainable parameters: 5,601**

### Final Dense Layer Calculation

```

Flatten output = 5 × 5 × 32 = 800 units
Dense weights = 800 × 1 = 800
Dense bias = 1
Total Dense parameters = 801

```

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (learning rate = 0.001) |
| Loss function | Binary cross-entropy |
| Metrics | Accuracy, Recall, Precision |
| Batch size | 32 |
| Epochs | Up to 35 |
| EarlyStopping | monitor = val_loss, patience = 5 |
| ReduceLROnPlateau | factor = 0.5, patience = 3, min_lr = 1e-6 |

---

## Results

### Hold-Out Test Set Performance

| Metric | Value |
|--------|-------|
| Test Loss | 0.5075 |
| Accuracy | 0.8526 |
| Precision | 0.8184 |
| **Recall (Sensitivity)** | **0.9821** |
| F1-Score | 0.8928 |

### Confusion Matrix

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | 149 | 85 |
| **Actual Pneumonia** | 7 | 383 |

### Clinical Implications

- **Missed pneumonia cases:** 7 out of 390 (1.8%)
- **False alarms:** 85 out of 234 (36.3%)

The model catches **98.2% of pneumonia cases**, prioritizing patient safety over reducing false alarms. False positives are clinically manageable through physician review and follow-up testing.

---

## Repository Structure

```

cnn-pneumonia-medmnist/
│
├── README.md               # Project documentation
├── data_prep.py            # Data preparation script
├── model.py                # CNN architecture definition
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore rules
│
├── data/                   # Generated .npy files (ignored by git)
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   ├── y_val.npy
│   ├── X_test.npy
│   └── y_test.npy
│
├── models/                 # Saved models (ignored by git)
│   └── pneumonia_cnn.h5
│
└── plots/                  # Generated visualizations (ignored by git)
    ├── sample_images.png
    ├── training_history.png
    ├── confusion_matrix.png
    ├── prediction_distribution.png
    └── metrics_report.txt

```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/cnn-pneumonia-medmnist.git
cd cnn-pneumonia-medmnist

# 2. Create and activate a virtual environment (recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1: Data Preparation

```bash
python data_prep.py
```

This script:

- Downloads the PneumoniaMNIST dataset
- Reshapes images to (28, 28, 1)
- Normalizes pixel values to [0, 1]
- Saves processed arrays as `.npy` files in `data/`
- Generates sample image visualization

### Step 2: Model Training

```bash
python train.py
```

This script:

- Loads the prepared `.npy` files
- Creates the CNN architecture from `model.py`
- Trains the model with EarlyStopping and ReduceLROnPlateau
- Saves the trained model to `models/pneumonia_cnn.h5`

### Step 3: Evaluation

```bash
python evaluate.py
```

This script:

- Loads the trained model and test data
- Calculates accuracy, precision, recall, and F1-score
- Generates confusion matrix
- Saves evaluation plots and metrics report

---

## Dependencies

```
tensorflow>=2.21.0
medmnist==3.0.2
numpy>=1.26.0
matplotlib>=3.9.0
scikit-learn>=1.9.0
pandas>=2.3.0
seaborn>=0.13.0
```

---

## Design Decisions

### Why Lightweight?

The model uses only 5,601 parameters because:

- Input images are small (28×28 grayscale)
- Binary classification is a relatively simple task
- Limited training data (4,708 samples) — larger models would overfit
- Assignment requirement for a lightweight CNN
- Practical for deployment in resource-constrained settings

### Why Recall Priority?

In pneumonia diagnosis:

- **False negative** = Missing pneumonia → patient sent home untreated → potentially life-threatening
- **False positive** = False alarm → additional testing → manageable inconvenience

The model is designed to minimize false negatives even at the cost of more false positives.

### Why No Horizontal Flipping?

Chest X-rays are **anatomically asymmetric** — the heart is on the left side. Horizontal flipping creates anatomically impossible images where the heart appears on the right. This would teach the model incorrect anatomical features and potentially reduce diagnostic accuracy.

### Why Separation of Data Preparation and Training?

The assignment explicitly requires this separation. `data_prep.py` handles all preprocessing and saves `.npy` files. `train.py` loads these files without any preprocessing logic. This ensures:

- Reproducibility — same data every time
- Efficiency — data prepared once, reused multiple times
- Modularity — can swap model architectures without redoing data prep
- Easier debugging — isolate data vs. model issues

---

## Training Curves

The training process showed:

- Training accuracy reached 97.05%
- Validation accuracy reached 96.37%
- Validation recall remained high at 97.94%
- Minimal overfitting (training and validation curves remained close)
- ReduceLROnPlateau activated at epochs 27 and 34
- Best weights restored from epoch 31

---

## Limitations

- **Low resolution** (28×28) — fine details are lost
- **Class imbalance** — pneumonia overrepresented (74.2%)
- **No data augmentation** — intentionally avoided due to medical imaging constraints
- **High false positive rate** (36.3%) — trade-off for high recall
- **Single dataset** — model may not generalize to X-rays from different sources

---

## Future Improvements

- Use higher resolution images if available
- Implement class weights to address imbalance
- Add carefully chosen medical augmentation (brightness, small shifts)
- Ensemble multiple models for robust predictions
- Explore transfer learning with pre-trained models
- Fine-tune decision threshold to balance precision/recall trade-off

---

## Author

Fonyuy Patrick

## License

This project is part of the Neural Networks & Computer Vision Assignment under instructor Gita.

---

## Acknowledgments

- MedMNIST for providing the PneumoniaMNIST dataset
- TensorFlow/Keras for the deep learning framework

```
