"""
Tipografik ve Grafolojik Analiz Modülü
Yazının fiziksel özelliklerinden anlamlandırma yapan 4 analiz:

1. Yazı Eğimi (Slant Analysis)      - karakterlerin dikey eksene göre açısı
2. Satır Düzgünlüğü (Baseline)      - satırın yatayda ne kadar dalgalandığı
3. Harf/Kelime Aralığı (Spacing)     - karakterler arası boşluk istatistikleri
4. Yazı Basıncı/Kalınlığı (Stroke)   - kalem izinin ortalama kalınlığı

Çalıştırma:
    python graphology.py ornek_gorsel.png
"""

import sys
import cv2
import numpy as np

from preprocessing import preprocess_pipeline
from segmentation import segment_lines


# ---------------------------------------------------------------------------
# 1. YAZI EĞİMİ (SLANT ANALYSIS)
# ---------------------------------------------------------------------------

def analyze_slant(char_img: np.ndarray) -> float:
    """
    Görüntü momentleri (image moments) ile bir karakterin dikey eksene göre
    kaç derece eğik olduğunu hesaplar. Bu, klasik bir OCR ön-işleme
    tekniğidir (aynı formül karakter deskew için de kullanılır).

    Pozitif değer sağa yatık, negatif değer sola yatık yazı demektir.
    """
    moments = cv2.moments(char_img)
    if abs(moments["mu02"]) < 1e-2:
        return 0.0
    skew = moments["mu11"] / moments["mu02"]
    angle_degrees = np.degrees(np.arctan(skew))
    return float(angle_degrees)


# ---------------------------------------------------------------------------
# 2. SATIR DÜZGÜNLÜĞÜ (BASELINE ANALYSIS)
# ---------------------------------------------------------------------------

def analyze_baseline(boxes: list) -> dict:
    """
    Bir satırdaki karakterlerin alt kenar (baseline) noktalarına doğrusal
    regresyon uygular. İki metrik döner:
    - egim_derece: satırın genel olarak yukarı/aşağı kayma açısı
    - dalgalanma_std: karakterlerin bu düz çizgiden ne kadar saptığı
      (yüksek değer = düzensiz/dalgalı satır, düşük değer = düzgün satır)
    """
    if len(boxes) < 2:
        return {"egim_derece": 0.0, "dalgalanma_std": 0.0}

    xs = np.array([b[0] + b[2] / 2 for b in boxes])
    ys_bottom = np.array([b[1] + b[3] for b in boxes])

    slope, intercept = np.polyfit(xs, ys_bottom, 1)
    predicted = slope * xs + intercept
    residuals = ys_bottom - predicted

    return {
        "egim_derece": float(np.degrees(np.arctan(slope))),
        "dalgalanma_std": float(residuals.std()),
    }


# ---------------------------------------------------------------------------
# 3. HARF ARALIĞI (SPACING ANALYSIS)
# ---------------------------------------------------------------------------

def analyze_spacing(boxes: list) -> dict:
    """
    Ardışık karakterler arasındaki yatay boşlukların ortalamasını ve
    standart sapmasını hesaplar. Yüksek std, düzensiz/tutarsız aralık
    kullanıldığını gösterir.
    """
    if len(boxes) < 2:
        return {"ortalama_bosluk": 0.0, "std_bosluk": 0.0}

    gaps = []
    for i in range(len(boxes) - 1):
        gap = boxes[i + 1][0] - (boxes[i][0] + boxes[i][2])
        gaps.append(max(0, gap))

    return {
        "ortalama_bosluk": float(np.mean(gaps)),
        "std_bosluk": float(np.std(gaps)),
    }


# ---------------------------------------------------------------------------
# 4. YAZI BASINCI / KALINLIĞI (STROKE WIDTH)
# ---------------------------------------------------------------------------

def analyze_stroke_width(char_img: np.ndarray) -> float:
    """
    Distance Transform ile her bir kalem izi pikselinin en yakın kenara
    olan uzaklığını hesaplar; ortalamanın 2 katı, yaklaşık kalem izi
    kalınlığını (basınç göstergesi) verir. Kalın/bastırarak yazanlarda
    bu değer yüksek, ince/hafif yazanlarda düşük çıkar.
    """
    if np.count_nonzero(char_img) == 0:
        return 0.0
    dist = cv2.distanceTransform(char_img, cv2.DIST_L2, 5)
    mean_radius = dist[char_img > 0].mean()
    return float(mean_radius * 2)


# ---------------------------------------------------------------------------
# KARAKTER TESPİTİ (satır görüntüsünden konum bilgili karakter listesi)
# ---------------------------------------------------------------------------

def _detect_characters_with_boxes(line_image: np.ndarray, min_width: int = 3,
                                   min_height: int = 8, min_area: int = 40):
    """Bir satır görüntüsündeki karakterleri, konumlarıyla (bbox) birlikte bulur."""
    contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [
        b for b in boxes
        if b[2] >= min_width and b[3] >= min_height and (b[2] * b[3]) >= min_area
    ]
    boxes.sort(key=lambda b: b[0])

    chars = []
    for (x, y, w, h) in boxes:
        char_img = line_image[y:y + h, x:x + w]
        chars.append((x, y, w, h, char_img))
    return chars


# ---------------------------------------------------------------------------
# ANA ANALİZ FONKSİYONU
# ---------------------------------------------------------------------------

def analyze_document(image_path: str) -> dict:
    """
    Bir görüntü dosyasını okur, önişler ve tüm grafolojik analizleri
    uygulayarak bir sonuç sözlüğü döner.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")

    binary = preprocess_pipeline(image)
    lines = segment_lines(binary)

    all_slants = []
    all_strokes = []
    line_results = []

    for line_img in lines:
        chars = _detect_characters_with_boxes(line_img)
        if not chars:
            continue

        boxes = [(x, y, w, h) for (x, y, w, h, _) in chars]

        line_slants = [analyze_slant(c) for (_, _, _, _, c) in chars]
        line_strokes = [analyze_stroke_width(c) for (_, _, _, _, c) in chars]

        all_slants.extend(line_slants)
        all_strokes.extend(line_strokes)

        line_results.append({
            "karakter_sayisi": len(chars),
            "baseline": analyze_baseline(boxes),
            "spacing": analyze_spacing(boxes),
            "ortalama_egim": float(np.mean(line_slants)) if line_slants else 0.0,
            "ortalama_kalinlik": float(np.mean(line_strokes)) if line_strokes else 0.0,
        })

    return {
        "satir_sayisi": len(line_results),
        "toplam_karakter": sum(r["karakter_sayisi"] for r in line_results),
        "genel_ortalama_egim": float(np.mean(all_slants)) if all_slants else 0.0,
        "genel_ortalama_kalinlik": float(np.mean(all_strokes)) if all_strokes else 0.0,
        "satirlar": line_results,
    }


def print_report(results: dict):
    """Analiz sonuçlarını okunabilir bir rapor olarak ekrana basar."""
    print("\n" + "=" * 50)
    print("  GRAFOLOJİK ANALİZ RAPORU")
    print("=" * 50)
    print(f"\n  Toplam satır: {results['satir_sayisi']}")
    print(f"  Toplam karakter: {results['toplam_karakter']}")
    print(f"\n  Genel ortalama yazı eğimi: {results['genel_ortalama_egim']:.1f} derece")
    print(f"  Genel ortalama kalem kalınlığı: {results['genel_ortalama_kalinlik']:.2f} piksel")

    for i, line in enumerate(results["satirlar"], start=1):
        print(f"\n  --- Satır {i} ---")
        print(f"    Karakter sayısı: {line['karakter_sayisi']}")
        print(f"    Ortalama eğim: {line['ortalama_egim']:.1f} derece")
        print(f"    Ortalama kalınlık: {line['ortalama_kalinlik']:.2f} piksel")
        print(f"    Satır eğimi (baseline): {line['baseline']['egim_derece']:.1f} derece")
        print(f"    Satır dalgalanması (std): {line['baseline']['dalgalanma_std']:.2f} piksel")
        print(f"    Ortalama harf boşluğu: {line['spacing']['ortalama_bosluk']:.1f} piksel")
        print(f"    Boşluk tutarlılığı (std): {line['spacing']['std_bosluk']:.2f} piksel")

    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python graphology.py <görüntü_yolu>")
        sys.exit(1)

    results = analyze_document(sys.argv[1])
    print_report(results)
