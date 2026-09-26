"""
Proje genel yapılandırma dosyası.

MİMARİ NOTU:
Bu proje dil-bağımsız (language-agnostic) tasarlanmıştır. Yeni bir dil
(örn. Türkçe) eklemek için tek yapman gereken LANGUAGE_CONFIGS sözlüğüne
yeni bir girdi eklemek ve ACTIVE_LANGUAGE değerini değiştirmektir.
Diğer modüller (model.py, data_loader.py) bu dosyadan okuduğu için
kod değişikliği gerekmez.
"""

import string

# --- Genel ayarlar ---
IMG_SIZE = 28              # EMNIST formatına uygun (28x28 gri tonlama)
BATCH_SIZE = 128
EPOCHS = 15
MODEL_SAVE_DIR = "models"
DATA_DIR = "data"

# --- Dil bazlı karakter setleri ---
# Her dil için: kullanılacak karakter listesi, veri seti adı ve sınıf sayısı
LANGUAGE_CONFIGS = {
    "english": {
        "charset": list(string.digits + string.ascii_uppercase + string.ascii_lowercase),
        "dataset": "emnist/byclass",   # tensorflow_datasets kayıtlı adı
        "num_classes": 62,             # 10 rakam + 26 büyük + 26 küçük harf
    },

    # --- GELECEK GENİŞLETME: Türkçe buraya eklenecek ---
    # "turkish": {
    #     "charset": list("0123456789ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
    #                      "abcçdefgğhıijklmnoöprsştuüvyz"),
    #     "dataset": "custom_turkish_dataset",
    #     "num_classes": 88,  # örnek: rakam + Türkçe büyük/küçük harfler
    # },
}

ACTIVE_LANGUAGE = "english"    # şu an aktif olan dil
