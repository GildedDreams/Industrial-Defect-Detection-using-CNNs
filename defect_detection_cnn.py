"""
CNN-based Defect Detection for Industrial Product Images (TensorFlow / Keras)
==============================================================================

Binary classifier: 0 = non_defective, 1 = defective  (defective is the "positive" class).

Expected dataset layout:
    data_dir/
        defective/        *.jpg / *.png ...
        non_defective/    *.jpg / *.png ...

Usage:
    # Train + evaluate on your own data
    python defect_detection_cnn.py --data_dir path/to/data --epochs 30

    # Smoke-test the whole pipeline on auto-generated synthetic images
    python defect_detection_cnn.py --demo --epochs 10

    # Predict on new images with a previously trained model
    python defect_detection_cnn.py --predict img1.jpg img2.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures to disk; works on servers without a display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
IMG_SIZE = (128, 128)          # every image is resized to this (height, width)
BATCH_SIZE = 32
CLASS_NAMES = ["non_defective", "defective"]   # index = label
OUTPUT_DIR = Path("outputs")
MODEL_PATH = OUTPUT_DIR / "defect_cnn.keras"
THRESHOLD_PATH = OUTPUT_DIR / "threshold.json"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

tf.keras.utils.set_random_seed(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# 1. Data loading + train/validation/test split
# --------------------------------------------------------------------------
def load_image_paths(data_dir):
    """Collect image file paths and labels from the class sub-folders."""
    paths, labels = [], []
    for label, name in enumerate(CLASS_NAMES):
        folder = Path(data_dir) / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {folder}")
        files = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
        paths += [str(p) for p in files]
        labels += [label] * len(files)
        print(f"  {name:>14}: {len(files)} images")
    return np.array(paths), np.array(labels)


def split_data(paths, labels):
    """70% train / 15% validation / 15% test, stratified to keep class ratios."""
    p_train, p_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    p_val, p_test, y_val, y_test = train_test_split(
        p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return (p_train, y_train), (p_val, y_val), (p_test, y_test)


# --------------------------------------------------------------------------
# 2. Preprocessing + augmentation (tf.data pipeline)
# --------------------------------------------------------------------------
# Augmentation layers are only active when called with training=True, so
# validation/test images are never altered.  Pixel values are still 0-255 here
# (normalization to 0-1 happens inside the model via a Rescaling layer, which
# means new images at prediction time are normalized automatically).
#
# NOTE: if product orientation matters for your defect type (e.g. a label that
# must be upright), remove vertical flips / reduce rotation.
augmenter = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.08),                       # about +/- 29 degrees
    layers.RandomZoom(0.10),
    layers.RandomBrightness(0.2, value_range=(0, 255)),  # brightness jitter
    layers.RandomContrast(0.2),
], name="augmentation")


def _load_and_resize(path, label):
    """Read an image file, force 3 channels, resize to IMG_SIZE."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)               # float32, range 0-255
    img.set_shape((*IMG_SIZE, 3))
    return img, tf.cast(label, tf.float32)


def make_dataset(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0., 255.), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------
# 3. CNN architecture
# --------------------------------------------------------------------------
def build_model():
    """
    Input -> Rescaling(1/255)
          -> 4 x [Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool]
             (filters 32, 64, 128, 256  : low-level edges -> textures -> defect shapes)
          -> GlobalAveragePooling -> Dense(128) -> Dropout(0.5) -> Dense(1, sigmoid)
    """
    def conv_block(x, filters):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        return layers.MaxPooling2D(2)(x)

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)            # normalize pixels to [0, 1]
    for f in (32, 64, 128, 256):
        x = conv_block(x, f)
    x = layers.GlobalAveragePooling2D()(x)             # fewer params than Flatten
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)                         # regularization
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(defective)

    model = keras.Model(inputs, outputs, name="defect_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )
    return model


# --------------------------------------------------------------------------
# 4. Training
# --------------------------------------------------------------------------
def train(model, train_ds, val_ds, y_train, epochs):
    # Class weights counter class imbalance (defects are usually the minority).
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}
    print("Class weights:", class_weight)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weight, callbacks=callbacks)
    return history


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_title(metric.capitalize()); ax.set_xlabel("Epoch"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Evaluation
# --------------------------------------------------------------------------
def select_threshold(y_val, p_val, min_recall=0.95):
    """
    Pick the decision threshold on the VALIDATION set (never the test set).
    Among thresholds achieving recall >= min_recall (i.e. few missed defects),
    choose the one with the highest precision (i.e. fewest false alarms).
    Falls back to 0.5 if the recall target cannot be met.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    ok = recall[:-1] >= min_recall
    if not ok.any():
        return 0.5
    best = np.argmax(np.where(ok, precision[:-1], -1))
    return float(thresholds[best])


def evaluate(model, test_ds, y_test, threshold):
    probs = model.predict(test_ds, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    # Core metrics (defective = positive class)
    print(f"\n=== Test-set results (threshold = {threshold:.3f}) ===")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}"
          "  (of items flagged defective, how many truly were)")
    print(f"Recall   : {recall_score(y_test, preds):.4f}"
          "  (of truly defective items, how many were caught)")
    print(f"F1-score : {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")

    # Confusion matrix + error rates that matter in manufacturing
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Neg: {tn} | False Pos (false alarms): {fp} | "
          f"False Neg (MISSED defects): {fn} | True Pos: {tp}")
    print(f"False-negative rate (defects missed): {fn / max(fn + tp, 1):.2%}")
    print(f"False-positive rate (good parts rejected): {fp / max(fp + tn, 1):.2%}")

    print("\nClassification report:\n")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))

    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (test set)")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# 6. Prediction on new images
# --------------------------------------------------------------------------
def predict_images(image_paths, model=None, threshold=None):
    """Return (label, P(defective)) for each image path."""
    if model is None:
        model = keras.models.load_model(MODEL_PATH)
    if threshold is None:
        threshold = (json.loads(THRESHOLD_PATH.read_text())["threshold"]
                     if THRESHOLD_PATH.exists() else 0.5)
    results = []
    for path in image_paths:
        img, _ = _load_and_resize(str(path), 0)           # same resize as training
        prob = float(model.predict(tf.expand_dims(img, 0), verbose=0)[0, 0])
        label = CLASS_NAMES[int(prob >= threshold)]
        results.append((str(path), label, prob))
        print(f"{path}: {label.upper()}  (P(defective) = {prob:.3f})")
    return results


# --------------------------------------------------------------------------
# Optional: synthetic demo data so the pipeline can be tested without a dataset
# --------------------------------------------------------------------------
def make_demo_dataset(root, n_per_class=300, size=128):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(SEED)
    for cls in CLASS_NAMES:
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            base = int(rng.integers(110, 160))
            arr = np.clip(rng.normal(base, 8, (size, size, 3)), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            d = ImageDraw.Draw(img)
            cx, cy, r = (int(v) for v in (rng.integers(52, 76), rng.integers(52, 76), rng.integers(28, 38)))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(50, 50, 50),
                      width=4, fill=(base + 40,) * 3)         # a "part"
            if cls == "defective":                            # scratch and/or dark blotch
                if rng.random() < 0.7:
                    x0, y0 = (int(v) for v in rng.integers(cx - r, cx + r, 2))
                    d.line([x0, y0, x0 + int(rng.integers(-25, 25)), y0 + int(rng.integers(-25, 25))],
                           fill=(30, 30, 30), width=2)
                if rng.random() < 0.6:
                    bx, by = (int(v) for v in rng.integers(cx - r // 2, cx + r // 2, 2))
                    d.ellipse([bx - 4, by - 4, bx + 5, by + 5], fill=(40, 40, 40))
            img.save(root / cls / f"{cls}_{i:04d}.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="folder with defective/ and non_defective/")
    ap.add_argument("--demo", action="store_true", help="generate and use synthetic data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min_recall", type=float, default=0.95,
                    help="recall target used when tuning the decision threshold")
    ap.add_argument("--predict", nargs="+", help="image(s) to classify with the saved model")
    args = ap.parse_args()

    if args.predict:                                   # inference only
        predict_images(args.predict)
        return

    if args.demo:
        args.data_dir = "demo_data"
        print("Generating synthetic demo dataset ...")
        make_demo_dataset(Path(args.data_dir))
    if not args.data_dir:
        ap.error("provide --data_dir, --demo, or --predict")

    print("Loading dataset:")
    paths, labels = load_image_paths(args.data_dir)
    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = split_data(paths, labels)
    print(f"Train/Val/Test sizes: {len(p_tr)}/{len(p_va)}/{len(p_te)}")

    train_ds = make_dataset(p_tr, y_tr, training=True)
    val_ds = make_dataset(p_va, y_va, training=False)
    test_ds = make_dataset(p_te, y_te, training=False)   # unshuffled -> order matches y_te

    model = build_model()
    model.summary()
    history = train(model, train_ds, val_ds, y_tr, args.epochs)
    plot_history(history)

    # Tune threshold on validation data, then report once on the untouched test set
    p_val = model.predict(val_ds, verbose=0).ravel()
    threshold = select_threshold(y_va, p_val, args.min_recall)
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}))
    evaluate(model, test_ds, y_te, threshold)
    model.save(MODEL_PATH)
    print(f"\nSaved model, threshold and plots to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
