"""
Veri Yükleme Modülü
EMNIST veri setini tensorflow_datasets üzerinden yükler.

GELECEK GENİŞLETME:
Türkçe (veya başka bir dil) eklemek için load_custom_dataset() fonksiyonunu
doldurman yeterli - train.py ve model.py hiç değişmeden çalışmaya devam eder.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from config import LANGUAGE_CONFIGS, ACTIVE_LANGUAGE, BATCH_SIZE


def _preprocess(image, label):
    """Piksel değerlerini normalize eder ve EMNIST'e özgü yönlendirmeyi düzeltir."""
    image = tf.cast(image, tf.float32) / 255.0
    # EMNIST görüntüleri kaynağında 90 derece döndürülmüş + ayna halinde gelir
    image = tf.image.transpose(image)
    return image, label


def load_emnist_dataset():
    """EMNIST (byclass split) train/test verisini tf.data.Dataset olarak döner."""
    config = LANGUAGE_CONFIGS[ACTIVE_LANGUAGE]
    dataset_name = config["dataset"]

    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name,
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info


def load_custom_dataset(data_dir: str):
    """
    GELECEKTE KULLANILACAK: Türkçe (veya başka dil) için klasör tabanlı
    veri yükleme. Beklenen klasör yapısı:

        data_dir/
            a/
                img1.png
                img2.png
            b/
                ...
            c_cedilla/   (örn. 'ç' için)
                ...

    Şimdilik implement edilmedi - placeholder olarak bırakıldı.
    """
    raise NotImplementedError(
        "Özel veri seti yükleyici henüz eklenmedi. "
        "Türkçe veri seti eklerken bu fonksiyonu doldur."
    )


if __name__ == "__main__":
    ds_train, ds_test, ds_info = load_emnist_dataset()
    print(ds_info)
