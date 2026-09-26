"""
Tahmin (Prediction / Inference) Script'i
Eğitilmiş CNN modelini gerçek bir görüntü üzerinde çalıştırır.

Akış: Görüntü -> Önişleme -> Segmentasyon -> Her karakter için CNN tahmini -> Metin

Çalıştırma:
    python predict.py ornek_gorsel.png
"""

import sys
import cv2
import numpy as np
import tensorflow as tf

from config import IMG_SIZE, LANGUAGE_CONFIGS, ACTIVE_LANGUAGE, MODEL_SAVE_DIR
from preprocessing import preprocess_pipeline
from segmentation import full_segmentation_pipeline

# Sınıf indeksini karaktere çeviren liste (config.py'deki charset ile birebir aynı sırada)
CHARSET = LANGUAGE_CONFIGS[ACTIVE_LANGUAGE]["charset"]


def load_trained_model():
    """Kaydedilmiş en iyi modeli diskten yükler."""
    model_path = f"{MODEL_SAVE_DIR}/{ACTIVE_LANGUAGE}_cnn_best.keras"
    print(f"Model yükleniyor: {model_path}")
    return tf.keras.models.load_model(model_path)


def prepare_character_image(char_img: np.ndarray) -> np.ndarray:
    """
    Segmentasyondan gelen tek bir karakter görüntüsünü modele uygun
    28x28 formatına getirir. En-boy oranını bozmadan ortalar (padding).
    """
    h, w = char_img.shape

    # En-boy oranını koruyarak 20x20'ye sığdır (EMNIST'teki karakter/kenar boşluğu oranına yakın)
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 28x28'lik siyah bir tuval oluştur, karakteri ortala
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y_offset = (IMG_SIZE - new_h) // 2
    x_offset = (IMG_SIZE - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Normalize et ve modele uygun boyuta getir: (1, 28, 28, 1)
    normalized = canvas.astype(np.float32) / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict_character(model, char_img: np.ndarray) -> tuple:
    """Tek bir karakter görüntüsü için tahmin yapar. (karakter, güven_skoru) döner."""
    prepared = prepare_character_image(char_img)
    predictions = model.predict(prepared, verbose=0)
    class_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_index])
    return CHARSET[class_index], confidence


def predict_text(image_path: str, model=None) -> str:
    """
    Tam pipeline: görüntüyü oku, önişle, segmentle, her karakteri tanı,
    satır/kelime yapısını koruyarak metni yeniden oluştur.
    """
    if model is None:
        model = load_trained_model()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")

    binary = preprocess_pipeline(image)
    lines = full_segmentation_pipeline(binary)

    result_lines = []
    for line_words in lines:
        words_text = []
        for word_chars in line_words:
            if len(word_chars) == 0:
                continue
            chars_text = ""
            for char_img in word_chars:
                if char_img.size == 0:
                    continue
                char, confidence = predict_character(model, char_img)
                chars_text += char
            words_text.append(chars_text)
        result_lines.append(" ".join(words_text))

    return "\n".join(result_lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <görüntü_yolu>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_trained_model()

    print(f"\n'{image_path}' işleniyor...\n")
    recognized_text = predict_text(image_path, model)

    print("=" * 40)
    print("TANINAN METİN:")
    print("=" * 40)
    print(recognized_text)
    print("=" * 40)
