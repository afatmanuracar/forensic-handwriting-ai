"""
Karakter & Satır Segmentasyon Modülü
Önişlenmiş (binary) görüntüden satırları, kelimeleri ve karakterleri ayırır.

Hiyerarşi: Görüntü -> Satırlar -> Kelimeler -> Karakterler
"""

import cv2
import numpy as np


def segment_lines(binary_image: np.ndarray, min_line_height: int = 10):
    """
    Yatay izdüşüm (horizontal projection) ile satırları bulur.
    Piksel yoğunluğunun sıfıra düştüğü noktalar satır aralarına işaret eder.
    """
    row_sums = np.sum(binary_image, axis=1)
    lines = []
    in_line = False
    start = 0

    for i, val in enumerate(row_sums):
        if val > 0 and not in_line:
            start = i
            in_line = True
        elif val == 0 and in_line:
            if i - start >= min_line_height:
                lines.append((start, i))
            in_line = False

    if in_line:
        lines.append((start, len(row_sums)))

    return [binary_image[y1:y2, :] for y1, y2 in lines]


def segment_words(line_image: np.ndarray, min_gap: int = None):
    """
    Dikey izdüşüm (vertical projection) ile bir satırdaki kelimeleri ayırır.

    min_gap verilmezse görüntü genişliğine ORANTILI olarak otomatik hesaplanır
    (%4'ü). Bu sayede farklı çözünürlükte/farklı kırpma boyutunda çekilmiş
    fotoğraflarda da tutarlı çalışır - sabit piksel değeri, farklı boyuttaki
    görüntülerde harfleri yanlışlıkla ayrı kelime sanabilirdi.
    """
    if min_gap is None:
        min_gap = max(15, int(line_image.shape[1] * 0.04))

    col_sums = np.sum(line_image, axis=0)
    words = []
    in_word = False
    start = 0
    gap_count = 0

    for i, val in enumerate(col_sums):
        if val > 0:
            if not in_word:
                start = i
                in_word = True
            gap_count = 0
        else:
            if in_word:
                gap_count += 1
                if gap_count >= min_gap:
                    words.append((start, i - gap_count + 1))
                    in_word = False

    if in_word:
        words.append((start, len(col_sums)))

    return [line_image[:, x1:x2] for x1, x2 in words]


def segment_characters(word_image: np.ndarray, min_char_width: int = 3,
                        min_char_height: int = 8, min_area: int = 40):
    """
    Contour Analysis (Kontur Analizi) ile bir kelimedeki tekil karakterleri bulur.
    Bounding box'lar soldan sağa sıralanır (okuma sırası korunur).

    min_char_height ve min_area, gürültüden (nokta, leke, küçük artık) kalan
    çok küçük konturları eler - gerçek bir harf genelde hem belirli bir
    yükseklikte hem de belirli bir piksel alanında olur.
    """
    contours, _ = cv2.findContours(
        word_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [
        b for b in boxes
        if b[2] >= min_char_width
        and b[3] >= min_char_height
        and (b[2] * b[3]) >= min_area
    ]
    boxes.sort(key=lambda b: b[0])  # x koordinatına göre soldan sağa sırala

    characters = []
    for (x, y, w, h) in boxes:
        char_img = word_image[y:y + h, x:x + w]
        characters.append(char_img)

    return characters


def full_segmentation_pipeline(binary_image: np.ndarray):
    """
    Tüm segmentasyon hiyerarşisini uygular: satır -> kelime -> karakter.
    Dönüş yapısı: [ [ [char_img, ...], [char_img, ...] ], ... ]
                    satır 1'in kelimeleri     satır 1'in kelimeleri
    """
    result = []
    lines = segment_lines(binary_image)

    for line in lines:
        line_words = []
        words = segment_words(line)
        for word in words:
            chars = segment_characters(word)
            line_words.append(chars)
        result.append(line_words)

    return result
