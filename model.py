"""
CNN Modeli (TensorFlow/Keras)
Karakter tanıma için evrişimli sinir ağı mimarisi.

MİMARİ NOTU: num_classes parametresiyle çalışır, bu sayede aynı mimari
İngilizce (62 sınıf), Türkçe (farklı sınıf sayısı) veya başka bir dil için
kod değişikliği yapmadan yeniden kullanılabilir.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE, LANGUAGE_CONFIGS, ACTIVE_LANGUAGE


def build_cnn_model(num_classes: int = None) -> tf.keras.Model:
    """
    Karakter tanıma için CNN mimarisi kurar.
    num_classes verilmezse config.py'deki aktif dilin sınıf sayısı kullanılır.
    """
    if num_classes is None:
        num_classes = LANGUAGE_CONFIGS[ACTIVE_LANGUAGE]["num_classes"]

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
