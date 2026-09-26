"""
Eğitim (Training) Script'i
EMNIST verisiyle CNN modelini eğitir, en iyi modeli kaydeder.

Çalıştırma:
    python train.py

Not: Google Colab'da GPU ile çalıştırman önerilir - CPU'da EMNIST
(byclass split, ~700K örnek) eğitimi çok yavaş olabilir.
"""

import os
import tensorflow as tf
from config import EPOCHS, MODEL_SAVE_DIR, ACTIVE_LANGUAGE
from data_loader import load_emnist_dataset
from model import build_cnn_model


def train():
    print(f"[{ACTIVE_LANGUAGE}] için veri seti yükleniyor...")
    ds_train, ds_test, ds_info = load_emnist_dataset()

    print("Model kuruluyor...")
    model = build_cnn_model()
    model.summary()

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"{ACTIVE_LANGUAGE}_cnn_best.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
    ]

    print("Eğitim başlıyor...")
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print(f"Model kaydedildi: {checkpoint_path}")
    return history


if __name__ == "__main__":
    train()
