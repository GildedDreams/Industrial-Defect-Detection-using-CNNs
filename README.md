# Industrial Defect Detection with a CNN

A TensorFlow/Keras project that classifies images of industrial products as **defective** or **non-defective**. A single script, `defect_detection_cnn.py`, covers data loading, preprocessing, augmentation, training, evaluation, and prediction on new images.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Requirements and Installation](#requirements-and-installation)
4. [Preparing Your Dataset](#preparing-your-dataset)
5. [Quick Start](#quick-start)
6. [Command-Line Options](#command-line-options)
7. [How It Works](#how-it-works)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Output Files](#output-files)
10. [Predicting on New Images](#predicting-on-new-images)
11. [Customization Guide](#customization-guide)
12. [Tips for Better Results](#tips-for-better-results)
13. [Troubleshooting](#troubleshooting)
14. [Limitations](#limitations)

---

## Features

- Binary classification: `non_defective` (label 0) vs `defective` (label 1, the positive class)
- Image resizing and pixel normalization
- On-the-fly augmentation: flips, rotation, zoom, brightness, and contrast
- Stratified 70/15/15 train/validation/test split
- Four-block CNN with batch normalization, dropout, and global average pooling
- Class weighting for imbalanced datasets
- Early stopping, learning-rate scheduling, and best-model checkpointing
- Evaluation with accuracy, precision, recall, F1, ROC-AUC, a classification report, and a confusion matrix
- Decision threshold tuned on the validation set to hit a target recall, so few defects are missed
- Built-in synthetic data generator for testing the pipeline without a real dataset
- Command-line inference for new images

---

## Project Structure

```
.
├── defect_detection_cnn.py     # Entire pipeline: train, evaluate, predict
├── README.md
├── demo_data/                  # Created by --demo (synthetic images)
└── outputs/                    # Created on first run
    ├── defect_cnn.keras        # Best trained model
    ├── threshold.json          # Tuned decision threshold
    ├── training_curves.png     # Loss and accuracy per epoch
    └── confusion_matrix.png    # Test-set confusion matrix
```

---

## Requirements and Installation

- Python 3.9 or newer
- TensorFlow 2.13 or newer (needed for the `.keras` model format and the Keras augmentation layers used here)

```bash
# Optional but recommended: use a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install tensorflow scikit-learn matplotlib pillow numpy
```

A GPU is not required, but it speeds up training considerably. See the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu) if you want to enable one.

---

## Preparing Your Dataset

Organize your images into two folders named exactly `defective` and `non_defective`:

```
my_dataset/
├── defective/
│   ├── img_001.jpg
│   ├── img_002.png
│   └── ...
└── non_defective/
    ├── img_101.jpg
    ├── img_102.png
    └── ...
```

- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Images can be any size and will be resized automatically. Grayscale images are converted to 3 channels.
- Subfolders inside each class folder are searched recursively.
- Defects are usually rare. Class imbalance is handled with class weights, but more defect examples always help.

**Avoid data leakage.** If your dataset contains near-duplicates (consecutive frames from a camera, or the same part photographed several times), make sure they all end up in the same split. The script splits randomly by file, so duplicates can inflate test scores. Group such images before splitting if this applies to you.

---

## Quick Start

**1. Test the pipeline with synthetic data (a few minutes on CPU):**

```bash
python defect_detection_cnn.py --demo --epochs 10
```

This generates 300 synthetic images per class in `demo_data/`, trains the model, and prints the evaluation report.

**2. Train on your own data:**

```bash
python defect_detection_cnn.py --data_dir path/to/my_dataset --epochs 30
```

**3. Classify new images with the trained model:**

```bash
python defect_detection_cnn.py --predict part_a.jpg part_b.png
```

---

## Command-Line Options

| Option | Default | Description |
|---|---|---|
| `--data_dir PATH` | none | Folder containing `defective/` and `non_defective/` |
| `--demo` | off | Generate and use a synthetic dataset instead of `--data_dir` |
| `--epochs N` | 30 | Maximum training epochs (early stopping may end sooner) |
| `--min_recall R` | 0.95 | Recall target used when tuning the decision threshold |
| `--predict IMG [IMG ...]` | none | Classify one or more images using the saved model, then exit |

You must provide one of `--data_dir`, `--demo`, or `--predict`.

---

## How It Works

### 1. Data loading and splitting

The script collects image paths from the two class folders and splits them into **70% train, 15% validation, and 15% test**. The split is stratified, so each set keeps the same defective/non-defective ratio, and uses a fixed random seed (42) for reproducibility.

- **Train**: used to fit the model weights
- **Validation**: used for early stopping, learning-rate scheduling, and threshold tuning
- **Test**: touched only once, for the final evaluation

### 2. Preprocessing

| Step | Details |
|---|---|
| Resize | All images are resized to 128×128 (`IMG_SIZE`) |
| Channels | Forced to 3 (RGB) |
| Normalization | Pixel values scaled to 0–1 by a `Rescaling(1/255)` layer inside the model |

Normalization lives inside the model on purpose: the same preprocessing is then applied automatically at prediction time, so train and inference can never drift apart.

### 3. Data augmentation

Applied to **training images only**, with fresh random changes every epoch:

| Augmentation | Setting |
|---|---|
| Flip | Horizontal and vertical |
| Rotation | About ±29° |
| Zoom | ±10% |
| Brightness | ±20% |
| Contrast | ±20% |

This simulates variation in part placement and lighting, which reduces overfitting and improves robustness. If orientation matters for your product (for example, a printed label that must be upright), reduce the flips and rotation in the `augmenter` definition.

### 4. Model architecture

```
Input (128×128×3)
  └─ Rescaling (1/255)
  └─ Block 1: [Conv3×3(32)  → BatchNorm → ReLU] ×2 → MaxPool 2×2
  └─ Block 2: [Conv3×3(64)  → BatchNorm → ReLU] ×2 → MaxPool 2×2
  └─ Block 3: [Conv3×3(128) → BatchNorm → ReLU] ×2 → MaxPool 2×2
  └─ Block 4: [Conv3×3(256) → BatchNorm → ReLU] ×2 → MaxPool 2×2
  └─ GlobalAveragePooling2D
  └─ Dense(128, ReLU)
  └─ Dropout(0.5)
  └─ Dense(1, sigmoid)  →  P(defective)
```

- **Convolutional layers** extract features. Early layers detect edges and textures, and deeper layers detect shapes such as scratches, cracks, and dents.
- **Batch normalization** stabilizes and speeds up training.
- **Max pooling** halves the spatial size after each block, reducing computation and adding tolerance to small shifts.
- **Global average pooling** collapses each feature map to one number, which uses far fewer parameters than flattening and reduces overfitting.
- **Dense layers** combine the features and make the final decision.
- **Dropout (0.5)** randomly disables units during training to improve generalization.

### 5. Training

| Setting | Value |
|---|---|
| Loss | Binary cross-entropy |
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 32 |
| Class weights | "Balanced", computed from the training labels |
| EarlyStopping | Monitors `val_loss`, patience 8, restores best weights |
| ReduceLROnPlateau | Halves the learning rate after 3 epochs without improvement |
| ModelCheckpoint | Saves the best model by `val_loss` |

### 6. Threshold tuning

A sigmoid output is a probability, and the default cutoff of 0.5 is rarely optimal for inspection. After training, the script finds the threshold on the **validation set** that:

1. achieves recall of at least `--min_recall` (default 0.95), meaning at most about 5% of defects are missed, and
2. among those thresholds, gives the highest precision, meaning the fewest false alarms.

If the recall target cannot be reached, it falls back to 0.5. The threshold is saved to `outputs/threshold.json` and reused at prediction time.

---

## Evaluation Metrics

All metrics are computed on the held-out **test set**, with **defective as the positive class**.

| Metric | Meaning | Why it matters here |
|---|---|---|
| **Accuracy** | Share of all predictions that are correct | Can look high on imbalanced data even if defects are missed |
| **Precision** | Of items flagged defective, the share that truly are | Low precision means many good parts are rejected (false positives) |
| **Recall** | Of truly defective items, the share that were caught | Low recall means defects reach customers (false negatives) |
| **F1-score** | Harmonic mean of precision and recall | Single balanced summary of the two |
| **ROC-AUC** | Ranking quality across all thresholds | Threshold-independent measure of separability |

The script also prints:

- a **confusion matrix** (also saved as an image) with true negatives, false positives, false negatives, and true positives
- the **false-negative rate** (share of defects missed) and **false-positive rate** (share of good parts rejected)
- a full **classification report** with per-class precision, recall, F1, and support

### Reading the confusion matrix

```
                    Predicted
                non_defective   defective
Actual
non_defective        TN             FP   ← good parts wrongly rejected
defective            FN             TP   ← FN = defects that escaped
```

In most factories, a **false negative** (a defective part shipped) costs more than a **false positive** (a good part re-inspected). Raise `--min_recall` to push false negatives down, accepting more false positives in return.

### Interpreting results

- Look at **recall on the defective class** first, then precision.
- A large gap between validation and test results suggests overfitting, a small dataset, or leakage.
- Training accuracy far above validation accuracy means overfitting. Add augmentation, more data, or stronger dropout.
- Both low means underfitting. Train longer, enlarge the model, or increase `IMG_SIZE`.

---

## Output Files

All files are written to `outputs/`:

| File | Description |
|---|---|
| `defect_cnn.keras` | The best trained model (full architecture and weights) |
| `threshold.json` | The tuned decision threshold, for example `{"threshold": 0.37}` |
| `training_curves.png` | Training and validation loss and accuracy per epoch |
| `confusion_matrix.png` | Confusion matrix on the test set |

---

## Predicting on New Images

### From the command line

```bash
python defect_detection_cnn.py --predict path/to/image1.jpg path/to/image2.png
```

Example output (illustrative):

```
path/to/image1.jpg: NON_DEFECTIVE  (P(defective) = 0.041)
path/to/image2.png: DEFECTIVE  (P(defective) = 0.932)
```

### From Python

```python
from defect_detection_cnn import predict_images

results = predict_images(["part_a.jpg", "part_b.jpg"])
# results is a list of (path, label, probability) tuples
for path, label, prob in results:
    print(path, label, round(prob, 3))
```

The saved model and threshold in `outputs/` are loaded automatically. Images are resized the same way as in training, and normalization is built into the model.

---

## Customization Guide

All main settings are constants near the top of `defect_detection_cnn.py`.

| What to change | Where | Notes |
|---|---|---|
| Input resolution | `IMG_SIZE` | Increase (for example 224×224 or 256×256) if defects are tiny. Training gets slower and uses more memory. |
| Batch size | `BATCH_SIZE` | Lower it if you run out of memory. |
| Split ratios | `split_data()` | Edit the two `test_size` values. |
| Augmentation | `augmenter` | Add, remove, or tune layers to match how your products appear in practice. |
| Architecture | `build_model()` | Change the filter list `(32, 64, 128, 256)` or the dense layer size. |
| Learning rate | `build_model()` | The Adam learning rate argument. |
| Recall target | `--min_recall` | Higher means fewer missed defects and more false alarms. |
| Class names | `CLASS_NAMES` | Folder names must match. Index 1 is the positive class. |

### Using transfer learning

With fewer than roughly 1,000 images, a pretrained backbone often beats a from-scratch CNN. In `build_model()`, replace the convolutional blocks with something like:

```python
base = keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3),
                                      include_top=False, weights="imagenet")
base.trainable = False  # train only the new head first
x = keras.applications.mobilenet_v2.preprocess_input(inputs)  # replaces Rescaling
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
```

Use this in place of the `Rescaling` layer and the conv blocks, because MobileNetV2 expects inputs scaled to [-1, 1]. After the head converges, you can unfreeze the top layers of the backbone and fine-tune with a low learning rate (for example 1e-5).

---

## Tips for Better Results

- **More defect examples** is the single most effective improvement. Include every defect type you expect to see.
- **Consistent imaging** (fixed camera, lighting, and background) makes the task much easier than varied conditions.
- **Increase resolution** if defects occupy only a few pixels at 128×128.
- **Review mistakes.** Look at the false negatives and false positives directly. They often reveal labeling errors or a defect type missing from training data.
- **Check labels.** Even a few percent of mislabeled images can cap performance.
- **Run several seeds** and compare, since small datasets produce noisy results.

---

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `FileNotFoundError: Expected folder not found` | Folder names must be exactly `defective` and `non_defective` inside `--data_dir`. |
| `AttributeError` for `RandomBrightness` or similar layers | TensorFlow is too old. Upgrade to 2.13 or newer. |
| Out-of-memory error | Reduce `BATCH_SIZE` or `IMG_SIZE`. |
| Model predicts everything as one class | Check class balance and labels. Train longer, and confirm class weights are printed at the start of training. |
| Validation loss never improves | Lower the learning rate, check for mislabeled images, or add data. |
| Very high test scores that seem too good | Look for near-duplicate images across splits (data leakage). |
| Corrupt image error during training | Remove or re-save the unreadable file. |
| `--predict` fails to find the model | Train first so `outputs/defect_cnn.keras` exists, and run from the directory that contains `outputs/`. |

---

## Limitations

- The model classifies a **whole image** as defective or not. It does not locate or outline the defect. For localization, consider object detection (such as YOLO) or segmentation models, or add Grad-CAM heatmaps for visual explanations.
- It is trained for the **defect types and imaging conditions present in your data**. New defect types, different cameras, or changed lighting may need new training data.
- The synthetic `--demo` data is for testing the pipeline only. Its results say nothing about real-world performance.
- This code has not been validated on any specific industrial dataset. Always check performance on a representative test set before relying on it for quality decisions.
