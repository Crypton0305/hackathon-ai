"""
train_models.py

This script creates the two deep learning models used by the app:

1. CNN model -> detects vehicle damage from an image (damaged / not damaged)
2. ANN model -> predicts claim risk score from tabular data (low / medium / high)

UPDATED VERSION:
The CNN is now trained on a REAL dataset (e.g. the Kaggle "Car Damage
Detection" dataset) instead of synthetic images, so it generalizes to real
uploaded car photos.

--------------------------------------------------------------------------
DATASET LOCATION (already set up for this project)
--------------------------------------------------------------------------
This script expects your extracted Kaggle dataset here:

    data/training/<damage-folder>/*.jpg
    data/training/<whole-folder>/*.jpg
    data/validation/<damage-folder>/*.jpg
    data/validation/<whole-folder>/*.jpg

It does NOT matter if the folders are named "damage"/"whole",
"00-damage"/"01-whole", etc — the script auto-detects them by looking
for DAMAGE_KEYWORD ("damage") and WHOLE_KEYWORD ("whole") inside the
folder names. Both training and validation splits are automatically
merged and re-split by this script, so no manual copying is needed.

Just run:
    python train_models.py
--------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob
from PIL import Image

# Make results repeatable
np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# CONFIG - update these to match your real dataset
# -------------------------------------------------------------------
IMG_SIZE = 128  # bumped up from 64 -> real photos need more detail than synthetic ones

# Your Kaggle dataset is already extracted as:
#   data/training/<class_folder>/*.jpg
#   data/validation/<class_folder>/*.jpg
# Both splits get merged together here and re-split by this script,
# so you don't need to manually copy/merge anything.
DATASET_SPLIT_DIRS = [
    os.path.join("data", "training"),
    os.path.join("data", "validation"),
]

# Keywords used to auto-detect which subfolder is which class, so it
# works whether your folders are named "damage"/"whole",
# "00-damage"/"01-whole", etc. Change these ONLY if auto-detection fails.
DAMAGE_KEYWORD = "damage"
WHOLE_KEYWORD = "whole"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# -------------------------------------------------------------------
# PART 1: CNN model for vehicle damage detection (image modality)
# -------------------------------------------------------------------

def _find_class_folder(split_dir, keyword):
    """
    Looks inside split_dir for a subfolder whose name contains `keyword`
    (case-insensitive). This lets the script work whether folders are
    named "damage", "00-damage", "Damage", etc.
    """
    if not os.path.isdir(split_dir):
        return None

    for name in os.listdir(split_dir):
        full_path = os.path.join(split_dir, name)
        if os.path.isdir(full_path) and keyword.lower() in name.lower():
            return full_path

    return None


def _load_images_from_folder(folder_path, label, img_size):
    """Loads every image in a folder, resizes it, and returns (images, labels)."""
    images = []
    labels = []

    if not folder_path or not os.path.isdir(folder_path):
        return images, labels  # caller handles the "nothing found" case

    filepaths = [
        p for p in glob.glob(os.path.join(folder_path, "*"))
        if p.lower().endswith(VALID_EXTENSIONS)
    ]

    for path in filepaths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to 0-1
            images.append(arr)
            labels.append(label)
        except Exception as e:
            print(f"Skipping unreadable image {path}: {e}")

    return images, labels


def load_real_image_dataset(img_size=IMG_SIZE):
    """
    Loads the real car-damage dataset from disk, merging every split
    listed in DATASET_SPLIT_DIRS (e.g. data/training + data/validation).

    For each split, it auto-detects the damage/whole subfolders by
    matching DAMAGE_KEYWORD / WHOLE_KEYWORD against the folder names.
    """
    damage_images, damage_labels = [], []
    whole_images, whole_labels = [], []

    for split_dir in DATASET_SPLIT_DIRS:
        if not os.path.isdir(split_dir):
            print(f"  (skipping {split_dir} - not found)")
            continue

        damage_folder = _find_class_folder(split_dir, DAMAGE_KEYWORD)
        whole_folder = _find_class_folder(split_dir, WHOLE_KEYWORD)

        print(f"Loading from {split_dir}:")
        print(f"  damage folder -> {damage_folder}")
        print(f"  whole  folder -> {whole_folder}")

        d_imgs, d_labels = _load_images_from_folder(damage_folder, label=1, img_size=img_size)
        w_imgs, w_labels = _load_images_from_folder(whole_folder, label=0, img_size=img_size)

        damage_images += d_imgs
        damage_labels += d_labels
        whole_images += w_imgs
        whole_labels += w_labels

    print(f"Total: {len(damage_images)} damaged images, {len(whole_images)} whole/undamaged images")

    if len(damage_images) == 0 or len(whole_images) == 0:
        raise ValueError(
            "Could not find images for one or both classes.\n"
            "Check DATASET_SPLIT_DIRS, DAMAGE_KEYWORD, and WHOLE_KEYWORD at the "
            "top of this file, and confirm the folder names under data/training "
            "and data/validation match what's expected."
        )

    X = np.array(damage_images + whole_images, dtype=np.float32)
    y = np.array(damage_labels + whole_labels, dtype=np.int32)

    # Shuffle so train_test_split's stratify works cleanly and classes are mixed
    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]

    return X, y


def build_cnn_model(img_size=IMG_SIZE):
    """
    A slightly deeper CNN than before (real photos are noisier and more
    varied than synthetic ones, so a bit more capacity + augmentation +
    dropout helps it generalize instead of memorizing).
    """
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    model = keras.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),  # output: probability of damage
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn():
    X, y = load_real_image_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training CNN damage detection model on real images...")
    model = build_cnn_model()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=25,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN test accuracy: {accuracy:.3f}")

    model_path = os.path.join(MODELS_DIR, "cnn_damage_model.keras")
    model.save(model_path)
    print(f"Saved CNN model to {model_path}")


# -------------------------------------------------------------------
# PART 2: ANN model for claim risk prediction (tabular modality)
# -------------------------------------------------------------------
# NOTE: still synthetic for now. If you want this trained on your real
# historical_claims_sample.csv instead, tell me and I'll wire that up too
# (column names/target need to match what's in that file).

def make_synthetic_tabular_data(num_samples=2000):
    """
    Creates a synthetic historical claims dataset.

    Features:
    - claim_amount: how much money is being claimed
    - vehicle_age: age of the vehicle in years
    - previous_claims: number of past claims by this customer
    - driver_age: age of the driver
    - damage_severity: 0-1 score coming from the CNN (simulated here for training)

    Target: risk_level (0 = low, 1 = medium, 2 = high)
    """
    claim_amount = np.random.uniform(500, 50000, num_samples)
    vehicle_age = np.random.uniform(0, 20, num_samples)
    previous_claims = np.random.poisson(1.2, num_samples)
    driver_age = np.random.uniform(18, 75, num_samples)
    damage_severity = np.random.uniform(0, 1, num_samples)

    # Simple rule-based risk score to generate realistic-looking labels
    risk_score = (
        (claim_amount / 50000) * 0.4
        + (previous_claims / 5) * 0.3
        + damage_severity * 0.2
        + ((20 - vehicle_age).clip(min=0) / 20) * -0.1
    )
    risk_score += np.random.normal(0, 0.05, num_samples)  # add noise

    risk_level = pd.cut(
        risk_score,
        bins=[-np.inf, 0.33, 0.6, np.inf],
        labels=[0, 1, 2],  # 0 = low, 1 = medium, 2 = high
    ).astype(int)

    data = pd.DataFrame({
        "claim_amount": claim_amount,
        "vehicle_age": vehicle_age,
        "previous_claims": previous_claims,
        "driver_age": driver_age,
        "damage_severity": damage_severity,
        "risk_level": risk_level,
    })
    return data


def build_ann_model(num_features):
    """A small ANN for tabular risk classification."""
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax"),  # 3 risk classes
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ann():
    print("Generating synthetic tabular dataset...")
    os.makedirs("data", exist_ok=True)
    data = make_synthetic_tabular_data()
    data.to_csv(os.path.join("data", "historical_claims.csv"), index=False)

    feature_columns = [
        "claim_amount", "vehicle_age", "previous_claims", "driver_age", "damage_severity"
    ]
    X = data[feature_columns].values
    y = data["risk_level"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training ANN risk prediction model...")
    model = build_ann_model(num_features=X.shape[1])
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"ANN test accuracy: {accuracy:.3f}")

    model_path = os.path.join(MODELS_DIR, "ann_risk_model.keras")
    model.save(model_path)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "ann_scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "ann_feature_columns.pkl"))
    print(f"Saved ANN model and scaler to {MODELS_DIR}/")


if __name__ == "__main__":
    train_cnn()
    print("-" * 60)
    train_ann()
    print("-" * 60)
    print("All models trained and saved successfully.")