"""
Alan Yönlendirici (Domain Router)

unified_verifier.py, imza ve genel el yazısı verisini TEK bir modelde
birleştirmeyi denedi; sonuç her iki alanda da bir miktar performans kaybı
oldu (imza: AUC 0.852->0.794, genel yazı: 0.826->0.750). Bu script farklı
bir çözüm dener: iki alana özel modeli (daha yüksek başarılı) SAKLAR, ama
önce görüntünün hangi alana ait olduğunu (kısa/imza mı, çok satırlı not mu)
otomatik tahmin eden küçük bir sınıflandırıcı ekler, sonra doğru modeli
çağırır. Kullanıcı elle "imza mı genel yazı mı" seçmek zorunda kalmaz.

Önkoşul - bu iki dosyanın önceden üretilmiş olması gerekir:
    signature_model.json  (python signature_verifier.py fit ...)
    writer_model.json     (python writer_verifier.py fit ...)

Kullanım:
    python domain_router.py fit sign_data\\train sign_data\\test "eng_data\\...\\English Handwritten Pages Dataset"
    python domain_router.py compare ornekA.png ornekB.png
    python domain_router.py compare ornekA.png ornekB.png --domain imza   (otomatik tespiti geçersiz kıl)
"""

import os
import sys
import json
import argparse

import numpy as np

from signature_verifier import load_person_data, extract_features, fit_logistic
from writer_verifier import load_writer_features, split_writers
from batch_validation_multi import auc

CLASSIFIER_PATH_DEFAULT = "domain_classifier.json"
SIGNATURE_MODEL_PATH = "signature_model.json"
WRITER_MODEL_PATH = "writer_model.json"


# ---------------------------------------------------------------------------
# HAM ÖZELLİK VEKTÖRLERİNİ TOPLAMA (çift değil, TEK görüntü bazında)
# ---------------------------------------------------------------------------

def flatten_signature_images(data: dict) -> list:
    feats = []
    for entry in data.values():
        for _, f in entry["gercek"] + entry["sahte"]:
            feats.append(f)
    return feats


def flatten_writer_images(data: dict) -> list:
    feats = []
    for items in data.values():
        for _, f in items:
            feats.append(f)
    return feats


# ---------------------------------------------------------------------------
# ALAN SINIFLANDIRICI: bir görüntü imza mı, genel el yazısı mı?
# ---------------------------------------------------------------------------

def fit_domain_classifier(imza_feats: list, yazi_feats: list) -> dict:
    X = np.array(imza_feats + yazi_feats)
    y = np.array([1.0] * len(imza_feats) + [0.0] * len(yazi_feats))

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    w, b = fit_logistic(Xn, y, l2=5.0)
    return {"mean": mean, "std": std, "w": w, "b": b}


def predict_domain_prob(clf: dict, feat: np.ndarray) -> float:
    """P(imza) döner; 1'e yakınsa imza, 0'a yakınsa genel el yazısı."""
    x = (feat - clf["mean"]) / clf["std"]
    z = float(np.dot(x, clf["w"]) + clf["b"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def save_classifier(clf: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mean": clf["mean"].tolist(), "std": clf["std"].tolist(),
                   "w": clf["w"].tolist(), "b": clf["b"]}, f)


def load_classifier(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {"mean": np.array(raw["mean"]), "std": np.array(raw["std"]),
            "w": np.array(raw["w"]), "b": raw["b"]}


# ---------------------------------------------------------------------------
# ALANA ÖZEL MODELLERİ YÜKLEME / KULLANMA (signature_model.json, writer_model.json)
# ---------------------------------------------------------------------------

def load_pair_model(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {"std": np.array(raw["std"]), "w": np.array(raw["w"]), "b": raw["b"],
            "mask": np.array(raw["mask"], dtype=bool)}


def score_pair(model: dict, fa: np.ndarray, fb: np.ndarray) -> float:
    x = (np.abs(fa - fb) / model["std"])[model["mask"]]
    z = float(np.dot(x, model["w"]) + model["b"])
    return 100.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------------------
# KOMUTLAR
# ---------------------------------------------------------------------------

def cmd_fit(args):
    print("\n[1/3] İmza verisi yükleniyor (train + test)...")
    sig_train = load_person_data(args.sig_train_dir, args.sig_limit, "imza-train")
    sig_test = load_person_data(args.sig_test_dir, args.sig_limit, "imza-test")

    print("[2/3] Genel el yazısı verisi yükleniyor ve train/test ayrılıyor...")
    writer_all = load_writer_features(args.writer_dir, args.writer_limit, "yazı-tümü")
    writer_train, writer_test = split_writers(writer_all, args.test_frac)

    imza_train_feats = flatten_signature_images(sig_train)
    yazi_train_feats = flatten_writer_images(writer_train)
    imza_test_feats = flatten_signature_images(sig_test)
    yazi_test_feats = flatten_writer_images(writer_test)

    print(f"\n  Eğitim görüntüsü: {len(imza_train_feats)} imza, {len(yazi_train_feats)} genel yazı")
    print(f"  Test görüntüsü:   {len(imza_test_feats)} imza, {len(yazi_test_feats)} genel yazı")

    print("\n[3/3] Alan sınıflandırıcısı eğitiliyor...")
    clf = fit_domain_classifier(imza_train_feats, yazi_train_feats)
    save_classifier(clf, args.classifier)

    test_imza_probs = [predict_domain_prob(clf, f) for f in imza_test_feats]
    test_yazi_probs = [predict_domain_prob(clf, f) for f in yazi_test_feats]
    correct = sum(p >= 0.5 for p in test_imza_probs) + sum(p < 0.5 for p in test_yazi_probs)
    total = len(test_imza_probs) + len(test_yazi_probs)
    domain_auc = auc(test_imza_probs, test_yazi_probs)

    print("\n" + "=" * 70)
    print("  ALAN SINIFLANDIRICI SONUÇLARI (hiç görülmemiş imza + yazı test görüntüleri)")
    print("=" * 70)
    print(f"  Doğruluk: {correct}/{total}  (%{100 * correct / total:.1f})")
    print(f"  AUC (imza vs genel yazı ayrımı): {domain_auc:.3f}")
    print("=" * 70)
    print(f"\n  Alan sınıflandırıcı kaydedildi: {args.classifier}")
    print(f"  (signature_model.json ve writer_model.json'ın önceden var olması gerekir)\n")


def cmd_compare(args):
    missing = [p for p in (args.classifier, SIGNATURE_MODEL_PATH, WRITER_MODEL_PATH) if not os.path.exists(p)]
    if missing:
        print(f"\nEksik dosya(lar): {', '.join(missing)}")
        print("Önce 'python domain_router.py fit ...' ile alan sınıflandırıcısını,")
        print("ve signature_verifier.py / writer_verifier.py ile ilgili modelleri eğitmen gerekiyor.\n")
        return

    clf = load_classifier(args.classifier)
    sig_model = load_pair_model(SIGNATURE_MODEL_PATH)
    writer_model = load_pair_model(WRITER_MODEL_PATH)

    fa = extract_features(args.a)
    fb = extract_features(args.b)
    if fa is None or fb is None:
        print("Görüntülerden biri okunamadı.")
        return

    prob_a = predict_domain_prob(clf, fa)
    prob_b = predict_domain_prob(clf, fb)
    label_a = "imza" if prob_a >= 0.5 else "genel yazı"
    label_b = "imza" if prob_b >= 0.5 else "genel yazı"

    print(f"\n  Alan tespiti: '{os.path.basename(args.a)}' -> {label_a} (%{max(prob_a, 1-prob_a)*100:.0f} emin)")
    print(f"                '{os.path.basename(args.b)}' -> {label_b} (%{max(prob_b, 1-prob_b)*100:.0f} emin)")

    if args.domain:
        domain = args.domain
        print(f"\n  (Manuel geçersiz kılma: '{domain}' zorlanıyor)")
        model = sig_model if domain == "imza" else writer_model
        score = score_pair(model, fa, fb)
        print(f"  Benzerlik skoru ({domain} modeli): {score:.1f} / 100\n")
        return

    if label_a == label_b:
        domain = "imza" if label_a == "imza" else "yazi"
        model = sig_model if domain == "imza" else writer_model
        score = score_pair(model, fa, fb)
        print(f"\n  -> İkisi de '{label_a}' olarak tespit edildi, {label_a} modeli kullanılıyor.")
        print(f"  Benzerlik skoru: {score:.1f} / 100")
    else:
        print("\n  UYARI: İki görüntü farklı alanlardan görünüyor, otomatik seçim belirsiz.")
        print("  Her iki modelin sonucu da gösteriliyor - hangisinin doğru olduğuna")
        print("  görüntülere bakarak sen karar ver, veya --domain imza / --domain yazi")
        print("  ile zorla:")
        print(f"    İmza modeliyle:      {score_pair(sig_model, fa, fb):.1f} / 100")
        print(f"    Genel yazı modeliyle: {score_pair(writer_model, fa, fb):.1f} / 100")

    print("\n  Not: Bu skorlar ortalamada anlamlı ama TEK bir örnek için güvenilir")
    print("  karar vermeye yetmez; resmi bir adli görüş yerine geçmez.\n")


def main():
    ap = argparse.ArgumentParser(description="İmza / genel el yazısı alan yönlendiricisi")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("sig_train_dir")
    f.add_argument("sig_test_dir")
    f.add_argument("writer_dir")
    f.add_argument("--sig-limit", type=int, default=8)
    f.add_argument("--writer-limit", type=int, default=6)
    f.add_argument("--test-frac", type=float, default=0.25)
    f.add_argument("--classifier", default=CLASSIFIER_PATH_DEFAULT)
    f.set_defaults(func=cmd_fit)

    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--classifier", default=CLASSIFIER_PATH_DEFAULT)
    c.add_argument("--domain", choices=["imza", "yazi"], default=None)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
