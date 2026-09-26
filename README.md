# Handwriting AI - El Yazısı & Tipografi Analiz Sistemi

## Kurulum

```bash
pip install -r requirements.txt
```

## Proje Yapısı

```
handwriting_ai/
├── config.py           # Merkezi ayarlar - dil-bağımsız mimari
├── preprocessing.py     # OpenCV önişleme (grayscale, denoise, binarize, deskew)
├── segmentation.py      # Satır -> Kelime -> Karakter segmentasyonu
├── data_loader.py       # EMNIST veri yükleyici
├── model.py              # CNN mimarisi (TensorFlow/Keras)
├── train.py               # Eğitim script'i
└── requirements.txt
```

## Çalıştırma Sırası

1. **Önişleme testi** (opsiyonel, kendi görüntünle dene):
   ```bash
   python preprocessing.py ornek_gorsel.png
   ```

2. **Model mimarisini kontrol et:**
   ```bash
   python model.py
   ```

3. **Eğitimi başlat** (EMNIST otomatik indirilir, ilk çalıştırmada zaman alabilir):
   ```bash
   python train.py
   ```
   > Not: CPU'da EMNIST (byclass, ~700K örnek) eğitimi yavaş kalabilir.
   > Google Colab'da ücretsiz GPU ile çalıştırman önerilir.

## Genişletme (Türkçe ve Diğer Diller)

Mimari, yeni bir dil eklemek için şu şekilde tasarlandı:

1. `config.py` içindeki `LANGUAGE_CONFIGS` sözlüğüne yeni dil girdisi ekle
   (karakter seti + veri seti adı + sınıf sayısı).
2. `data_loader.py` içindeki `load_custom_dataset()` fonksiyonunu o dilin
   veri klasör yapısına göre doldur.
3. `config.py` içinde `ACTIVE_LANGUAGE` değerini değiştir.

`model.py` ve `train.py` içinde **hiçbir değişiklik gerekmez** - mimari
`num_classes` parametresiyle otomatik uyum sağlar.

## Sonraki Adımlar (Henüz Eklenmedi)

- Tipografik/Grafolojik analiz katmanı (eğim, satır düzgünlüğü, harf aralığı, baskı kalınlığı)
- Segmentasyon çıktısını modele besleyen tam uçtan uca (end-to-end) pipeline
- Türkçe karakter desteği
