"""
Yorumlama Katmanı: Adli Karşılaştırma + Geleneksel Grafoloji Yorumu

İKİ FARKLI ÖZELLİK, İKİ FARKLI GÜVENİLİRLİK SEVİYESİ:

1. compare_documents()  -> ADLİ/FORENSIC KARŞILAŞTIRMA
   İki yazı örneğinin (eğim, basınç, aralık, satır düzgünlüğü) sayısal
   olarak ne kadar benzer/farklı olduğunu ölçer. Bu, adli belge inceleme
   biliminin (questioned document examination) kullandığı türden nesnel,
   ölçülebilir bir karşılaştırma yöntemidir.

2. interpret_personality() -> GELENEKSEL GRAFOLOJİ YORUMU
   *** BİLİMSEL GEÇERLİLİĞİ YOKTUR ***
   Yazı özelliklerinden kişilik çıkarımı (graphology), akademik psikolojide
   pseudoscience (sözde bilim) olarak kabul edilir - kontrollü çalışmalarda
   güvenilirliği/geçerliliği kanıtlanamamıştır. Burada sadece popüler
   kültürde/geleneksel grafoloji literatüründe sıkça tekrarlanan
   yorumları gösteriyoruz - bilimsel bir değerlendirme OLARAK SUNULMAMALIDIR.

Çalıştırma:
    python interpretation.py ornek1.png                  # tek örnek yorumu
    python interpretation.py ornek1.png ornek2.png        # iki örneği karşılaştır
"""

import sys
from graphology import analyze_document


# ---------------------------------------------------------------------------
# 1. ADLİ / FORENSIC KARŞILAŞTIRMA
# ---------------------------------------------------------------------------

def compare_documents(metrics_a: dict, metrics_b: dict) -> dict:
    """
    İki analiz sonucunu karşılaştırır. Her özellik için mutlak farkı hesaplar
    ve genel bir 'tutarlılık skoru' üretir (0-100, yüksek = daha benzer).

    NOT: Bu basit bir istatistiksel karşılaştırmadır, resmi bir adli rapor
    yerine geçmez - gerçek adli vakalarda uzman bir belge incelemecisi
    (questioned document examiner) tarafından değerlendirilmelidir.
    """
    slant_diff = abs(metrics_a["genel_ortalama_egim"] - metrics_b["genel_ortalama_egim"])
    stroke_diff = abs(metrics_a["genel_ortalama_kalinlik"] - metrics_b["genel_ortalama_kalinlik"])

    def _avg_line_metric(result, key, subkey=None):
        lines = result["satirlar"]
        if not lines:
            return 0.0
        if subkey:
            return sum(l[key][subkey] for l in lines) / len(lines)
        return sum(l[key] for l in lines) / len(lines)

    spacing_a = _avg_line_metric(metrics_a, "spacing", "ortalama_bosluk")
    spacing_b = _avg_line_metric(metrics_b, "spacing", "ortalama_bosluk")
    spacing_diff = abs(spacing_a - spacing_b)

    baseline_a = _avg_line_metric(metrics_a, "baseline", "dalgalanma_std")
    baseline_b = _avg_line_metric(metrics_b, "baseline", "dalgalanma_std")
    baseline_diff = abs(baseline_a - baseline_b)

    # Basit bir normalize edilmiş benzerlik skoru (ne kadar küçük fark, o kadar yüksek skor)
    # Eşik değerleri kabaca kalibre edilmiştir, kesin bilimsel referans değildir.
    slant_score = max(0, 100 - slant_diff * 10)
    stroke_score = max(0, 100 - stroke_diff * 8)
    spacing_score = max(0, 100 - spacing_diff * 1.5)
    baseline_score = max(0, 100 - baseline_diff * 0.5)

    overall_score = (slant_score + stroke_score + spacing_score + baseline_score) / 4

    return {
        "egim_farki": round(slant_diff, 2),
        "kalinlik_farki": round(stroke_diff, 2),
        "bosluk_farki": round(spacing_diff, 2),
        "dalgalanma_farki": round(baseline_diff, 2),
        "tutarlilik_skoru": round(overall_score, 1),
    }


def print_comparison_report(comparison: dict, path_a: str, path_b: str):
    print("\n" + "=" * 55)
    print("  ADLİ YAZI KARŞILAŞTIRMA RAPORU")
    print("=" * 55)
    print(f"\n  Örnek A: {path_a}")
    print(f"  Örnek B: {path_b}\n")
    print(f"  Eğim farkı:          {comparison['egim_farki']} derece")
    print(f"  Kalem kalınlığı farkı: {comparison['kalinlik_farki']} piksel")
    print(f"  Harf boşluğu farkı:   {comparison['bosluk_farki']} piksel")
    print(f"  Satır dalgalanma farkı: {comparison['dalgalanma_farki']} piksel")
    print(f"\n  TUTARLILIK SKORU: {comparison['tutarlilik_skoru']} / 100")

    score = comparison["tutarlilik_skoru"]
    if score >= 75:
        yorum = "Yüksek tutarlılık - aynı kişiden çıkmış olma ihtimali güçlü."
    elif score >= 50:
        yorum = "Orta düzey tutarlılık - kesin sonuç için ek inceleme gerekir."
    else:
        yorum = "Düşük tutarlılık - farklı kişilerden çıkmış olabilir."
    print(f"  Yorum: {yorum}")

    print("\n  *** NOT: Bu, basit istatistiksel bir karşılaştırmadır. Resmi bir")
    print("  adli rapor yerine geçmez; gerçek vakalarda uzman bir belge")
    print("  incelemecisi tarafından değerlendirilmelidir. ***")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# 2. GELENEKSEL GRAFOLOJİ YORUMU (BİLİMSEL GEÇERLİLİĞİ YOKTUR)
# ---------------------------------------------------------------------------

def interpret_personality(metrics: dict) -> list:
    """
    Geleneksel/popüler grafoloji literatüründe sıkça tekrarlanan yorumları
    döner. *** BUNLAR BİLİMSEL OLARAK KANITLANMAMIŞTIR. ***
    Akademik psikoloji camiası, yazıdan kişilik çıkarımını (graphology)
    kontrollü çalışmalarda geçerliliği gösterilememiş bir yöntem olarak
    kabul eder. Burada sadece kültürel/geleneksel bir referans olarak
    sunuluyor.
    """
    slant = metrics["genel_ortalama_egim"]
    stroke = metrics["genel_ortalama_kalinlik"]

    avg_waviness = 0.0
    avg_spacing = 0.0
    lines = metrics["satirlar"]
    if lines:
        avg_waviness = sum(l["baseline"]["dalgalanma_std"] for l in lines) / len(lines)
        avg_spacing = sum(l["spacing"]["ortalama_bosluk"] for l in lines) / len(lines)

    yorumlar = []

    if slant > 5:
        yorumlar.append("Sağa yatık eğim: Geleneksel grafolojide dışa dönüklük/sosyallik ile ilişkilendirilir.")
    elif slant < -5:
        yorumlar.append("Sola yatık eğim: Geleneksel grafolojide içe dönüklük/temkinlilik ile ilişkilendirilir.")
    else:
        yorumlar.append("Dik (nötr) eğim: Geleneksel grafolojide duygusal denge/mantıksal yaklaşım ile ilişkilendirilir.")

    if stroke > 8:
        yorumlar.append("Kalın kalem izi: Geleneksel grafolojide güçlü duygusal yoğunluk ile ilişkilendirilir.")
    else:
        yorumlar.append("İnce kalem izi: Geleneksel grafolojide hassasiyet/detaycılık ile ilişkilendirilir.")

    if avg_waviness > 15:
        yorumlar.append("Dalgalı satır: Geleneksel grafolojide spontanelik/değişkenlik ile ilişkilendirilir.")
    else:
        yorumlar.append("Düz satır: Geleneksel grafolojide disiplin/kararlılık ile ilişkilendirilir.")

    if avg_spacing > 20:
        yorumlar.append("Geniş harf aralığı: Geleneksel grafolojide bağımsızlık ile ilişkilendirilir.")
    else:
        yorumlar.append("Dar harf aralığı: Geleneksel grafolojide sosyal yakınlık ihtiyacı ile ilişkilendirilir.")

    return yorumlar


def print_personality_report(yorumlar: list, path: str):
    print("\n" + "=" * 55)
    print("  GELENEKSEL GRAFOLOJİ YORUMU")
    print("=" * 55)
    print(f"\n  Örnek: {path}\n")
    for y in yorumlar:
        print(f"  - {y}")
    print("\n  " + "!" * 51)
    print("  ! ÖNEMLİ UYARI: Bu yorumlar BİLİMSEL DEĞİLDİR.")
    print("  ! Akademik psikolojide yazıdan kişilik çıkarımı")
    print("  ! (graphology) geçerliliği kanıtlanmamış bir yöntemdir.")
    print("  ! Sadece geleneksel/kültürel referans amaçlıdır,")
    print("  ! gerçek psikolojik değerlendirme YERİNE GEÇMEZ.")
    print("  " + "!" * 51)
    print("=" * 55 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım:")
        print("  python interpretation.py <görüntü>                 # kişilik yorumu")
        print("  python interpretation.py <görüntü1> <görüntü2>     # adli karşılaştırma")
        sys.exit(1)

    if len(sys.argv) == 2:
        metrics = analyze_document(sys.argv[1])
        yorumlar = interpret_personality(metrics)
        print_personality_report(yorumlar, sys.argv[1])
    else:
        metrics_a = analyze_document(sys.argv[1])
        metrics_b = analyze_document(sys.argv[2])
        comparison = compare_documents(metrics_a, metrics_b)
        print_comparison_report(comparison, sys.argv[1], sys.argv[2])
