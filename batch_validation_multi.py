"""
Çoklu Kişi Doğrulama Script'i
batch_validation.py'nin genişletilmiş hali - tek kişi yerine BİRDEN FAZLA
kişinin klasörlerini otomatik tarayıp genel istatistik çıkarır. Bu, tek
kişilik sonuçların şansa bağlı olup olmadığını görmek için önemlidir.

Beklenen klasör yapısı (Kaggle Signature Verification Dataset ile uyumlu):
    sign_data/train/001, sign_data/train/001_forg
    sign_data/train/002, sign_data/train/002_forg
    ...

Çalıştırma:
    python batch_validation_multi.py sign_data\\train 001,002,003,004,005 3
    (üçüncü parametre: kişi başına kaç örnek/çift test edilsin)
"""

import sys
import os

from batch_validation import load_images_from_folder, average_score_for_pairs


def main():
    if len(sys.argv) < 3:
        print("Kullanım: python batch_validation_multi.py <ana_klasor> <kisi_id_listesi> [limit]")
        print("Örnek:    python batch_validation_multi.py sign_data\\train 001,002,003,004,005 3")
        sys.exit(1)

    base_dir = sys.argv[1]
    person_ids = sys.argv[2].split(",")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    all_gg_scores = []
    all_gf_scores = []

    for person_id in person_ids:
        genuine_dir = os.path.join(base_dir, person_id)
        forged_dir = os.path.join(base_dir, f"{person_id}_forg")

        if not os.path.isdir(genuine_dir) or not os.path.isdir(forged_dir):
            print(f"[Atlandı] Kişi {person_id}: klasör bulunamadı")
            print(f"           Aranan gerçek yol: '{genuine_dir}'")
            print(f"           Aranan sahte yol:  '{forged_dir}'")
            print(f"           Mevcut çalışma dizini: '{os.getcwd()}'")
            continue

        print(f"\n{'#' * 55}")
        print(f"  KİŞİ: {person_id}")
        print(f"{'#' * 55}")

        genuine_files = load_images_from_folder(genuine_dir, limit=limit)
        forged_files = load_images_from_folder(forged_dir, limit=limit)

        if len(genuine_files) < 2:
            print(f"  [Atlandı] Yetersiz gerçek örnek")
            continue

        print("\n  -- Gerçek-Gerçek --")
        gg_scores = average_score_for_pairs(genuine_files, genuine_files, max_pairs=limit)
        all_gg_scores.extend(gg_scores)

        print("\n  -- Gerçek-Sahte --")
        gf_scores = average_score_for_pairs(genuine_files, forged_files, max_pairs=limit)
        all_gf_scores.extend(gf_scores)

    print("\n\n" + "=" * 55)
    print("  GENEL İSTATİSTİK ÖZETİ")
    print("=" * 55)
    print(f"  Test edilen kişi sayısı: {len(person_ids)}")

    if all_gg_scores:
        gg_avg = sum(all_gg_scores) / len(all_gg_scores)
        print(f"\n  Gerçek-Gerçek ortalama: {gg_avg:.1f} / 100  (n={len(all_gg_scores)} çift)")
    else:
        gg_avg = None
        print("\n  Gerçek-Gerçek: veri yok")

    if all_gf_scores:
        gf_avg = sum(all_gf_scores) / len(all_gf_scores)
        print(f"  Gerçek-Sahte ortalama:  {gf_avg:.1f} / 100  (n={len(all_gf_scores)} çift)")
    else:
        gf_avg = None
        print("  Gerçek-Sahte: veri yok")

    if gg_avg is not None and gf_avg is not None:
        print(f"\n  FARK: {gg_avg - gf_avg:.1f} puan")
        if gg_avg > gf_avg:
            print("  -> Yöntem TUTARLI şekilde beklenen yönde ayırt edici.")
        else:
            print("  -> UYARI: Genel örneklemde beklenen yön tutmuyor.")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
