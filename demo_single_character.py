"""
Tek Karakter Tanıma Demosu
Projenin en güvenilir bileşenini gösterir: eğitilmiş CNN modelinin
TEK bir el yazısı karakteri üzerindeki performansı.

İki mod:
1. Kendi çektiğin tek karakter fotoğrafını tanır (üst-3 tahmin + güven skoru)
2. EMNIST test setinden rastgele örneklerle modelin genel doğruluğunu ölçer
   (rapor için nicel/quantitative kanıt)

Çalıştırma:
    python demo_single_character.py harf_A.png
    python demo_single_character.py --emnist-sample 15
"""

import sys
import cv2
import numpy as np
import tensorflow as tf

from config import LANGUAGE_CONFIGS, ACTIVE_LANGUAGE
from preprocessing import preprocess_pipeline
from predict import prepare_character_image, load_trained_model

CHARSET = LANGUAGE_CONFIGS[ACTIVE_LANGUAGE]["charset"]


def confidence_bar(confidence: float, width: int = 30) -> str:
    """Terminalde basit bir görsel güven çubuğu çizer."""
    filled = int(confidence * width)
    return "#" * filled + "-" * (width - filled)


def predict_top_k(model, char_img: np.ndarray, k: int = 3):
    """Bir karakter görüntüsü için en olası k tahmini (karakter, güven) olarak döner."""
    prepared = prepare_character_image(char_img)
    predictions = model.predict(prepared, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:k]
    return [(CHARSET[i], float(predictions[i])) for i in top_indices]


def demo_from_image(image_path: str):
    """Kullanıcının kendi çektiği TEK KARAKTER fotoğrafını tanır."""
    model = load_trained_model()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")

    binary = preprocess_pipeline(image)
    top_predictions = predict_top_k(model, binary, k=3)

    print("\n" + "=" * 45)
    print(f"  GİRDİ: {image_path}")
    print("=" * 45)
    print("\n  EN OLASI 3 TAHMİN:\n")

    for rank, (char, confidence) in enumerate(top_predictions, start=1):
        bar = confidence_bar(confidence)
        print(f"  {rank}. '{char}'  {bar}  %{confidence * 100:.1f}")

    print("\n" + "=" * 45)
    best_char, best_conf = top_predictions[0]
    print(f"  SONUÇ: '{best_char}'  (%{best_conf * 100:.1f} güven)")
    print("=" * 45 + "\n")


def demo_from_emnist(num_samples: int = 10):
    """
    EMNIST test setinden rastgele örneklerle modelin genel doğruluğunu
    gösterir - raporun 'nicel sonuçlar' bölümü için kanıt niteliğinde.
    """
    import tensorflow_datasets as tfds

    model = load_trained_model()
    config = LANGUAGE_CONFIGS[ACTIVE_LANGUAGE]

    ds_test = tfds.load(config["dataset"], split="test", as_supervised=True)
    ds_test = ds_test.shuffle(1000).take(num_samples)

    correct = 0
    print("\n" + "=" * 50)
    print(f"  EMNIST TEST SETİNDEN {num_samples} RASTGELE ÖRNEK")
    print("=" * 50 + "\n")

    for i, (image, label) in enumerate(ds_test, start=1):
        image = tf.image.transpose(image).numpy().squeeze()
        image = image.astype(np.float32) / 255.0
        prepared = image.reshape(1, 28, 28, 1)

        predictions = model.predict(prepared, verbose=0)[0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class])

        true_char = CHARSET[int(label.numpy())]
        pred_char = CHARSET[predicted_class]
        is_correct = true_char == pred_char
        correct += int(is_correct)

        mark = "DOĞRU" if is_correct else "YANLIŞ"
        print(f"  {i:2d}. Gerçek: '{true_char}'  Tahmin: '{pred_char}'  "
              f"(%{confidence * 100:.1f})  [{mark}]")

    accuracy = correct / num_samples * 100
    print("\n" + "-" * 50)
    print(f"  DOĞRULUK: {correct}/{num_samples}  (%{accuracy:.1f})")
    print("-" * 50 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım:")
        print("  python demo_single_character.py <görüntü_yolu>")
        print("  python demo_single_character.py --emnist-sample [adet]")
        sys.exit(1)

    if sys.argv[1] == "--emnist-sample":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        demo_from_emnist(n)
    else:
        demo_from_image(sys.argv[1])
