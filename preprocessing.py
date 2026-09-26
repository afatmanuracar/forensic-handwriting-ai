"""
Görüntü Önişleme Modülü (Image Preprocessing)
Kağıt/dijital ortamdaki el yazısı görüntülerini OCR öncesi temizler.

Adımlar:
1. Gri tonlama (grayscale)
2. Gürültü temizleme (Gaussian Blur)
3. Adaptive Thresholding (binarizasyon)
4. Eğiklik düzeltme (deskew)
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR görüntüyü gri tonlamaya çevirir."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Gaussian Blur ile gürültü azaltma."""
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)


def binarize(gray_image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Adaptive Threshold ile siyah-beyaz (binary) görüntü üretir.
    Sabit eşiklemeden farklı olarak, farklı ışıklandırma/gölge
    koşullarında daha tutarlı sonuç verir.
    """
    return cv2.adaptiveThreshold(
        gray_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c,
    )


def deskew(binary_image: np.ndarray) -> np.ndarray:
    """
    Görüntüdeki metnin eğikliğini tespit edip düzeltir.
    minAreaRect ile beyaz piksellerin en küçük çevreleyen dikdörtgeninin
    açısını bulur ve görüntüyü ters yönde döndürür.
    """
    coords = np.column_stack(np.where(binary_image > 0))
    if len(coords) == 0:
        return binary_image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary_image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def remove_small_noise(binary_image: np.ndarray, min_area: int = 25) -> np.ndarray:
    """
    Bağlı bileşen (connected component) analizi ile min_area'dan küçük
    beyaz lekeleri temizler. Defter çizgileri, kağıt lekeleri, nokta
    gürültüsü gibi el yazısı OLMAYAN küçük parçaları eler.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    cleaned = np.zeros_like(binary_image)
    for label_id in range(1, num_labels):  # 0 = arka plan, atla
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255

    return cleaned


def bridge_strokes(binary_image: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """
    Morfolojik 'closing' (kapama) ile aynı harfe ait, aralarında küçük
    boşluk olan vuruşları birleştirir (örn. 'T' harfinin dikey ve yatay
    çizgileri arasındaki ince kopukluk).

    kernel_size verilmezse görüntü yüksekliğine orantılı hesaplanır -
    yüksek çözünürlüklü fotoğraflarda harfler de büyük piksel boyutunda
    olacağından sabit küçük bir çekirdek yetersiz kalabilir.
    """
    if kernel_size is None:
        kernel_size = max(5, binary_image.shape[0] // 15)
        if kernel_size % 2 == 0:
            kernel_size += 1  # tek sayı olmalı

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


def preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Tüm önişleme adımlarını sırayla uygular ve temizlenmiş binary görüntüyü döner.
    Sıra önemli: gürültü temizliği deskew'den önce (açı hesabını bozmasın),
    vuruş birleştirme ise en son (harf içi kopuklukları kapatır).
    """
    gray = to_grayscale(image)
    denoised = denoise(gray)
    binary = binarize(denoised)
    noise_free = remove_small_noise(binary)
    deskewed = deskew(noise_free)
    bridged = bridge_strokes(deskewed)
    return bridged


if __name__ == "__main__":
    # Hızlı test: bir görüntü dosyası üzerinde pipeline'ı dene
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        result = preprocess_pipeline(img)
        cv2.imwrite("preprocessed_output.png", result)
        print("Önişleme tamamlandı: preprocessed_output.png")
    else:
        print("Kullanım: python preprocessing.py <görüntü_yolu>")
