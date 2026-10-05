"""
Çoklu Kişi Doğrulama Script'i (AUC + güven aralığı)

Birden fazla kişinin klasörlerini otomatik tarar; GERÇEK-GERÇEK ve
GERÇEK-SAHTE çiftlerinin tutarlılık skorlarını karşılaştırır ve şunları üretir:
  - Ortalama ve standart sapma (grup bazında)
  - AUC: rastgele seçilen bir gerçek-gerçek çiftin, rastgele seçilen bir
    gerçek-sahte çiftten daha yüksek skor alma olasılığı (0.5 = şans, 1.0 = mükemmel)
  - AUC için %95 güven aralığı (kişi bazlı bootstrap - aynı kişiden gelen çiftler
    bağımsız olmadığı için kişiler yeniden örneklenir)
  - Tüm çift skorlarını içeren validation_results.csv dosyası

Çalıştırma (PowerShell'de kişi listesini TIRNAK İÇİNE al):
    python batch_validation_multi.py train "001,002,003,004,006,009,012,013,014,015" 3
"""

import sys
import os
import csv
import random

from batch_validation import load_images_from_folder, average_score_for_pairs


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def auc(positives, negatives):
    """Mann-Whitney yaklaşımıyla AUC. Eşit skorlar yarım puan sayılır."""
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def cluster_bootstrap_auc(records, n_boot=1000, seed=42):
    """Kişileri yeniden örnekleyerek AUC için %95 güven aralığı hesaplar."""
    by_person = {}
    for r in records:
        pos_neg = by_person.setdefault(r["kisi"], ([], []))
        if r["tur"] == "gercek-gercek":
            pos_neg[0].append(r["skor"])
        else:
            pos_neg[1].append(r["skor"])

    persons = sorted(by_person)
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        pos, neg = [], []
        for pid in [rng.choice(persons) for _ in persons]:
            pos.extend(by_person[pid][0])
            neg.extend(by_person[pid][1])
        a = auc(pos, neg)
        if a is not None:
            values.append(a)

    values.sort()
    low = values[int(0.025 * len(values))]
    high = values[int(0.975 * len(values)) - 1]
    return low, high


def save_csv(records, path="validation_results.csv"):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["kisi", "tur", "ornek_a", "ornek_b", "skor"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Tüm çift skorları kaydedildi: {path}")


def main():
    if len(sys.argv) < 3:
        print("Kullanım: python batch_validation_multi.py <ana_klasor> \"<kisi_id_listesi>\" [limit]")
        print("Örnek:    python batch_validation_multi.py train \"001,002,003,004\" 3")
        sys.exit(1)

    base_dir = sys.argv[1]
    person_ids = sys.argv[2].split(",")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    records = []
    tested_persons = 0

    for person_id in person_ids:
        genuine_dir = os.path.join(base_dir, person_id)
        forged_dir = os.path.join(base_dir, f"{person_id}_forg")

        if not os.path.isdir(genuine_dir) or not os.path.isdir(forged_dir):
            print(f"[Atlandı] Kişi {person_id}: klasör bulunamadı ({genuine_dir})")
            continue

        genuine_files = load_images_from_folder(genuine_dir, limit=limit)
        forged_files = load_images_from_folder(forged_dir, limit=limit)

        if len(genuine_files) < 2:
            print(f"[Atlandı] Kişi {person_id}: yetersiz gerçek örnek")
            continue

        print(f"\n{'#' * 55}\n  KİŞİ: {person_id}\n{'#' * 55}")
        tested_persons += 1

        print("\n  -- Gerçek-Gerçek --")
        average_score_for_pairs(genuine_files, genuine_files, max_pairs=limit,
                                records=records, person=person_id, pair_type="gercek-gercek")
        print("\n  -- Gerçek-Sahte --")
        average_score_for_pairs(genuine_files, forged_files, max_pairs=limit,
                                records=records, person=person_id, pair_type="gercek-sahte")

    gg = [r["skor"] for r in records if r["tur"] == "gercek-gercek"]
    gf = [r["skor"] for r in records if r["tur"] == "gercek-sahte"]

    print("\n\n" + "=" * 55)
    print("  GENEL İSTATİSTİK ÖZETİ")
    print("=" * 55)
    print(f"  Gerçekten test edilen kişi sayısı: {tested_persons}")

    if not gg or not gf:
        print("  Yeterli veri yok.")
        return

    print(f"\n  Gerçek-Gerçek: ort {mean(gg):.1f}  std {std(gg):.1f}  (n={len(gg)} çift)")
    print(f"  Gerçek-Sahte:  ort {mean(gf):.1f}  std {std(gf):.1f}  (n={len(gf)} çift)")
    print(f"  Ortalama farkı: {mean(gg) - mean(gf):.1f} puan")

    a = auc(gg, gf)
    low, high = cluster_bootstrap_auc(records)
    print(f"\n  AUC: {a:.3f}   (%95 güven aralığı: {low:.3f} - {high:.3f})")
    print("  (0.5 = şans düzeyi, 1.0 = mükemmel ayrım)")

    if low > 0.5:
        print("  -> Güven aralığı 0.5'in üzerinde: ayrım şans düzeyinden ayırt edilebiliyor,")
        print("     ancak AUC değeri bireysel karar için yeterli olup olmadığını belirler.")
    else:
        print("  -> Güven aralığı 0.5'i içeriyor: bu örneklemde şans düzeyinden")
        print("     ayrım kanıtlanamadı.")

    print()
    save_csv(records)
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
