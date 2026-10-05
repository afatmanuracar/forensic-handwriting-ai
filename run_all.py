"""
Tek Komutla Tam Pipeline

signature_verifier.py, writer_verifier.py ve domain_router.py'nin 'fit'
komutlarını SIRAYLA çalıştırır (kodlarını kopyalamaz, doğrudan fonksiyonlarını
çağırır), sonunda üç modülün de test AUC'sini tek bir özet tabloda gösterir.

Üretilen dosyalar:
    signature_model.json     - imza karşılaştırma modeli
    writer_model.json        - genel el yazısı (yazar) karşılaştırma modeli
    domain_classifier.json   - hangi modelin kullanılacağını otomatik seçen sınıflandırıcı

Kullanım:
    python run_all.py sign_data\\train sign_data\\test "eng_data\\...\\English Handwritten Pages Dataset"
    python run_all.py sign_data\\train sign_data\\test "eng_data\\..." --sig-limit 8 --writer-limit 6 --with-baseline

Bittikten sonra:
    python domain_router.py compare <gorselA> <gorselB>
"""

import argparse
import csv
from types import SimpleNamespace

import signature_verifier
import writer_verifier
import domain_router
from batch_validation_multi import auc


def read_csv_auc(path: str, pos_label: str = "gercek-gercek"):
    """Bir test sonucu CSV'sinden (skor ya da yeni_skor sütunu) AUC hesaplar."""
    pos, neg = [], []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = row.get("skor")
            if raw in (None, ""):
                raw = row.get("yeni_skor")
            if raw in (None, ""):
                continue
            score = float(raw)
            if row["tur"] == pos_label:
                pos.append(score)
            else:
                neg.append(score)
    return auc(pos, neg)


def main():
    ap = argparse.ArgumentParser(description="İmza + genel yazı + alan yönlendirici - tek komutla tam pipeline")
    ap.add_argument("sig_train_dir")
    ap.add_argument("sig_test_dir")
    ap.add_argument("writer_dir")
    ap.add_argument("--sig-limit", type=int, default=8)
    ap.add_argument("--writer-limit", type=int, default=6)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--with-baseline", action="store_true",
                     help="İmza modülünde eski (4 özellikli) yöntemi de test et - yavaştır")
    args = ap.parse_args()

    print("\n" + "#" * 78)
    print("# ADIM 1/3 - İMZA MODELİ (signature_verifier.py)")
    print("#" * 78)
    signature_verifier.cmd_fit(SimpleNamespace(
        train_dir=args.sig_train_dir, test_dir=args.sig_test_dir,
        limit=args.sig_limit, no_baseline=not args.with_baseline,
        model="signature_model.json",
    ))

    print("\n" + "#" * 78)
    print("# ADIM 2/3 - GENEL EL YAZISI (YAZAR) MODELİ (writer_verifier.py)")
    print("#" * 78)
    writer_verifier.cmd_fit(SimpleNamespace(
        base_dir=args.writer_dir, limit=args.writer_limit,
        test_frac=args.test_frac, model="writer_model.json",
    ))

    print("\n" + "#" * 78)
    print("# ADIM 3/3 - ALAN YÖNLENDİRİCİ (domain_router.py)")
    print("#" * 78)
    domain_router.cmd_fit(SimpleNamespace(
        sig_train_dir=args.sig_train_dir, sig_test_dir=args.sig_test_dir,
        writer_dir=args.writer_dir, sig_limit=args.sig_limit,
        writer_limit=args.writer_limit, test_frac=args.test_frac,
        classifier="domain_classifier.json",
    ))

    print("\n" + "#" * 78)
    print("# GENEL ÖZET")
    print("#" * 78)
    try:
        sig_auc = read_csv_auc("signature_test_results.csv")
        print(f"  İmza modeli        test AUC: {sig_auc:.3f}")
    except Exception as e:
        print(f"  İmza AUC okunamadı: {e}")

    try:
        writer_auc = read_csv_auc("writer_test_results.csv", pos_label="ayni-yazar")
        print(f"  Genel yazı modeli  test AUC: {writer_auc:.3f}")
    except Exception as e:
        print(f"  Genel yazı AUC okunamadı: {e}")

    print("\n  Üretilen dosyalar: signature_model.json, writer_model.json, domain_classifier.json")
    print("  Artık iki görüntüyü otomatik alan tespitiyle karşılaştırabilirsin:")
    print("    python domain_router.py compare <gorselA> <gorselB>")
    print("#" * 78 + "\n")


if __name__ == "__main__":
    main()
