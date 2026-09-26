"""
Gelişmiş Adli Önişleme Modülü (Forensic Preprocessing)
Bozuk, okunaksız veya gölgeli el yazılarını temizler ve adli analiz için hazırlar.
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR görüntüyü gri tonlamaya çevirir."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def remove_shadows_and_contrast(gray_image: np.ndarray) -> np.ndarray:
    """
    Kriminal incelemede kağıttaki gölgeleri ve lekeleri kaldırır, 
    yazı kontrastını artırır (CLAHE yöntemi).
    """
    dilated_img = cv2.dilate(gray_image, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray_image, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm_img)


def binarize_otsu(gray_image: np.ndarray) -> np.ndarray:
    """
    Otsu Binarization: Okunaksız ve düzensiz ışıklandırılmış yazılarda 
    en ideal siyah-beyaz eşiğini otomatik bulur.
    """
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def skeletonize(binary_image: np.ndarray) -> np.ndarray:
    """
    Morphological Skeletonization (İskeletleştirme)
    Harflerin kalınlığını 1 piksele indirir. Kalemin kağıt üzerindeki 
    gerçek hareket yönünü ve adli vuruş karakteristiklerini ortaya çıkarır.
    """
    skel = np.zeros(binary_image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = binary_image.copy()

    while True:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()

        if cv2.countNonZero(temp_img) == 0:
            break

    return skel


def forensic_preprocess_pipeline(image: np.ndarray) -> tuple:
    """
    Adli önişleme zinciri.
    Dönüş: (temizlenmiş_binary_resim, harf_iskelet_resmi)
    """
    gray = to_grayscale(image)
    enhanced = remove_shadows_and_contrast(gray)
    binary = binarize_otsu(enhanced)
    skeleton = skeletonize(binary)
    
    return binary, skeleton


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        binary, skel = forensic_preprocess_pipeline(img)
        cv2.imwrite("forensic_binary.png", binary)
        cv2.imwrite("forensic_skeleton.png", skel)
        print("Adli önişleme tamamlandı:")
        print("- Temizlenmiş Yazı: forensic_binary.png")
        print("- Yazı İskeleti (Kalem İzi): forensic_skeleton.png")
    else:
        print("Kullanım: python preprocessing.py <görüntü_yolu>")