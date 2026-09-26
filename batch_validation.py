"""
Toplu Adli Karşılaştırma Doğrulama Script'i

Bir veri setindeki GERÇEK-GERÇEK çiftlerinin (aynı kişi) ve GERÇEK-SAHTE
çiftlerinin (taklit) ortalama tutarlılık skorlarını karşılaştırarak,
yöntemin gerçekten ayırt edici olup olmadığını istatistiksel olarak test eder.

Beklenen kullanım (örn. Kaggle Signature Verification Dataset ile):
    python batch_validation.py <gercek_klasoru> <sahte_klasoru> [limit]

Örnek:
    python batch_validation.py sign_data/001 sign_data/001_forg 5
"""

import sys
import os
import glob
import itertools

from graphology import analyze_document
from interpretation import compare_documents


def load_images_from_folder(folder: str, limit: int = None) -> list:
    extensions = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()
    if limit:
        files = files[:limit]
    return files


def average_score_for_pairs(paths_a: list, paths_b: list, max_pairs: int = 10) -> list:
    """
    paths_a ve paths_b listelerinden çiftler oluşturup her biri için
    tutarlılık skorunu hesaplar. Aynı liste verilirse (gerçek-gerçek),
    kendisiyle eşleşmeyen tüm ikili kombinasyonları dener.
    """
    scores = []

    if paths_a is paths_b:
        pairs = list(itertools.combinations(paths_a, 2))
    else:
        pairs = list(itertools.product(paths_a, paths_b))

    for path_a, path_b in pairs[:max_pairs]:
        try:
            metrics_a = analyze_document(path_a)
            metrics_b = analyze_document(path_b)
            comparison = compare_documents(metrics_a, metrics_b)
            scores.append(comparison["tutarlilik_skoru"])
            print(f"  {os.path.basename(path_a)} vs {os.path.basename(path_b)}: "
                  f"{comparison['tutarlilik_skoru']:.1f}")
        except Exception as e:
            print(f"  [Atlandı] {os.path.basename(path_a)} / {os.path.basename(path_b)}: {e}")

    return scores


def main():
    if len(sys.argv) < 3:
        print("Kullanım: python batch_validation.py <gercek_klasoru> <sahte_klasoru> [limit]")
        sys.exit(1)

    genuine_dir = sys.argv[1]
    forged_dir = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    genuine_files = load_images_from_folder(genuine_dir, limit=limit)
    forged_files = load_images_from_folder(forged_dir, limit=limit)

    print(f"\nBulunan gerçek imza sayısı: {len(genuine_files)}")
    print(f"Bulunan sahte imza sayısı: {len(forged_files)}\n")

    if len(genuine_files) < 2:
        print("En az 2 gerçek imza görüntüsü gerekiyor (aynı kişinin birden fazla örneği).")
        sys.exit(1)

    print("=" * 55)
    print("  GERÇEK - GERÇEK KARŞILAŞTIRMALARI (aynı kişi)")
    print("=" * 55)
    gg_scores = average_score_for_pairs(genuine_files, genuine_files, max_pairs=limit)

    print("\n" + "=" * 55)
    print("  GERÇEK - SAHTE KARŞILAŞTIRMALARI (taklit)")
    print("=" * 55)
    gf_scores = average_score_for_pairs(genuine_files, forged_files, max_pairs=limit)

    print("\n" + "=" * 55)
    print("  ÖZET SONUÇLAR")
    print("=" * 55)

    gg_avg = sum(gg_scores) / len(gg_scores) if gg_scores else None
    gf_avg = sum(gf_scores) / len(gf_scores) if gf_scores else None

    if gg_avg is not None:
        print(f"  Gerçek-Gerçek ortalama tutarlılık: {gg_avg:.1f} / 100  (n={len(gg_scores)})")
    else:
        print("  Gerçek-Gerçek: yeterli veri yok")

    if gf_avg is not None:
        print(f"  Gerçek-Sahte ortalama tutarlılık:  {gf_avg:.1f} / 100  (n={len(gf_scores)})")
    else:
        print("  Gerçek-Sahte: yeterli veri yok")

    if gg_avg is not None and gf_avg is not None:
        print(f"\n  FARK: {gg_avg - gf_avg:.1f} puan")
        if gg_avg > gf_avg:
            print("  -> Yöntem beklenen yönde ayırt edici: gerçek çiftler daha tutarlı çıktı.")
        else:
            print("  -> UYARI: Beklenen yönün tersi çıktı - bu veri setinde/örneklemde")
            print("     yöntem ayırt edici olamadı. Rapor için dürüstçe belirtilmeli.")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
