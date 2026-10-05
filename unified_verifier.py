"""
Birleşik (Unified) El Yazısı Karşılaştırma Modeli

İki ayrı modelin ağırlıklarını birleştirmek (örn. ortalamasını almak)
istatistiksel olarak anlamsızdır. Doğrusu: imza verisi (sign_data) ve genel
el yazısı verisi (eng_data) ile oluşturulan eğitim çiftlerini TEK bir
havuzda birleştirip TEK bir lojistik regresyon modelini bu havuzda eğitmek.

Bu script üç modeli AYNI test setlerinde karşılaştırır:
  - yalnızca imza verisiyle eğitilmiş model
  - yalnızca genel el yazısı verisiyle eğitilmiş model
  - ikisinin birleşiminde eğitilmiş TEK model (unified)

Amaç: birleştirmenin performansı ne kadar etkilediğini görmek. Birleşik
model her iki test setinde de alana-özel modellere yakın kalıyorsa, "her
türlü el yazısı örneğinde kullanılabilir TEK model" iddiası kanıtlanmış olur.

Kullanım:
    python unified_verifier.py fit sign_data\\train sign_data\\test ^
        "eng_data\\English Handwritten Pages Dataset\\English Handwritten Pages Dataset" ^
        --sig-limit 8 --writer-limit 6 --test-frac 0.25

    python unified_verifier.py compare ornekA.png ornekB.png
"""

import os
import sys
import csv
import json
import argparse

import numpy as np

from signature_verifier import (
    load_person_data, build_pairs as build_sig_pairs,
    extract_features, fit_logistic, make_mask, FEATURE_NAMES,
)
from writer_verifier import (
    load_writer_features, build_pairs as build_writer_pairs, split_writers,
)
from batch_validation_multi import auc, cluster_bootstrap_auc

MODEL_PATH_DEFAULT = "unified_model.json"


# ---------------------------------------------------------------------------
# İKİ ALANI ORTAK BİR KAYIT FORMATINA ÇEVİRME
# ---------------------------------------------------------------------------

def collect_signature_records(data: dict) -> list:
    """sign_data kişi verisini {kisi, domain, label, fa, fb} kayıtlarına çevirir."""
    records = []
    for person, entry in data.items():
        for tur, pa, pb, fa, fb in build_sig_pairs(entry):
            records.append({
                "kisi": f"imza-{person}", "domain": "imza",
                "label": 1 if tur == "gercek-gercek" else 0, "fa": fa, "fb": fb,
            })
    return records


def collect_writer_records(data: dict, seed: int) -> list:
    """eng_data yazar verisini aynı ortak kayıt formatına çevirir."""
    records = []
    for tur, pa, pb, fa, fb in build_writer_pairs(data, seed=seed):
        wid = os.path.basename(os.path.dirname(pa))
        records.append({
            "kisi": f"yazi-{wid}", "domain": "yazi",
            "label": 1 if tur == "ayni-yazar" else 0, "fa": fa, "fb": fb,
        })
    return records


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def fit_from_records(records: list, mask=None) -> dict:
    if mask is None:
        mask = make_mask()
    all_feats = [r["fa"] for r in records] + [r["fb"] for r in records]
    std = np.std(all_feats, axis=0)
    std[std < 1e-6] = 1.0

    X = np.array([np.abs(r["fa"] - r["fb"]) / std for r in records])[:, mask]
    y = np.array([r["label"] for r in records], dtype=float)
    w, b = fit_logistic(X, y)
    return {"std": std, "w": w, "b": b, "mask": mask}


def score_pair(model: dict, fa: np.ndarray, fb: np.ndarray) -> float:
    x = (np.abs(fa - fb) / model["std"])[model["mask"]]
    z = float(np.dot(x, model["w"]) + model["b"])
    return 100.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def eval_records(model: dict, records: list) -> list:
    out = []
    for r in records:
        s = score_pair(model, r["fa"], r["fb"])
        out.append({"kisi": r["kisi"], "tur": "gercek-gercek" if r["label"] == 1 else "gercek-sahte", "skor": s})
    return out


def summarize(name: str, records: list) -> float:
    pos = [r["skor"] for r in records if r["tur"] == "gercek-gercek"]
    neg = [r["skor"] for r in records if r["tur"] != "gercek-gercek"]
    a = auc(pos, neg)
    low, high = cluster_bootstrap_auc(records)
    print(f"  {name:<42} AUC {a:.3f}   %95 GA: {low:.3f} - {high:.3f}   "
          f"(n_pos={len(pos)}, n_neg={len(neg)})")
    return a


def save_model(model: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURE_NAMES, "std": model["std"].tolist(),
                   "w": model["w"].tolist(), "b": model["b"],
                   "mask": model["mask"].tolist()}, f)


def load_model(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {"std": np.array(raw["std"]), "w": np.array(raw["w"]), "b": raw["b"],
            "mask": np.array(raw["mask"], dtype=bool)}


# ---------------------------------------------------------------------------
# KOMUTLAR
# ---------------------------------------------------------------------------

def cmd_fit(args):
    print("\n[1/4] İmza verisi yükleniyor (train + test)...")
    sig_train_data = load_person_data(args.sig_train_dir, args.sig_limit, "imza-train")
    sig_test_data = load_person_data(args.sig_test_dir, args.sig_limit, "imza-test")

    print("[2/4] Genel el yazısı verisi yükleniyor ve train/test ayrılıyor...")
    writer_data = load_writer_features(args.writer_dir, args.writer_limit, "yazı-tümü")
    writer_train_data, writer_test_data = split_writers(writer_data, args.test_frac)
    print(f"  Yazı - Train: {len(writer_train_data)} yazar   Test: {len(writer_test_data)} yazar")

    sig_train = collect_signature_records(sig_train_data)
    sig_test = collect_signature_records(sig_test_data)
    writer_train = collect_writer_records(writer_train_data, seed=1)
    writer_test = collect_writer_records(writer_test_data, seed=2)

    print("\n[3/4] Üç model eğitiliyor: yalnızca imza / yalnızca yazı / BİRLEŞİK...")
    model_sig_only = fit_from_records(sig_train)
    model_writer_only = fit_from_records(writer_train)
    model_unified = fit_from_records(sig_train + writer_train)
    print(f"  İmza eğitim çifti: {len(sig_train)}   Yazı eğitim çifti: {len(writer_train)}   "
          f"Birleşik: {len(sig_train) + len(writer_train)}")

    print("\n[4/4] Değerlendiriliyor (her üç model, her iki test setinde)...\n")
    print("=" * 90)
    print("  İMZA TEST SETİNDE")
    print("=" * 90)
    summarize("yalnızca imza modeli", eval_records(model_sig_only, sig_test))
    summarize("yalnızca yazı modeli (alan dışı)", eval_records(model_writer_only, sig_test))
    unified_on_sig = summarize("BİRLEŞİK model", eval_records(model_unified, sig_test))

    print("\n" + "=" * 90)
    print("  GENEL EL YAZISI TEST SETİNDE")
    print("=" * 90)
    summarize("yalnızca imza modeli (alan dışı)", eval_records(model_sig_only, writer_test))
    summarize("yalnızca yazı modeli", eval_records(model_writer_only, writer_test))
    unified_on_writer = summarize("BİRLEŞİK model", eval_records(model_unified, writer_test))

    print("\n" + "=" * 90)
    print(f"  ÖZET: Birleşik model  ->  imza AUC {unified_on_sig:.3f}  |  genel yazı AUC {unified_on_writer:.3f}")
    print("  Yorum: Alan-dışı satırlar (örn. yalnızca imza modelinin genel yazıda test")
    print("  edilmesi) genelde düşük çıkar - bu beklenir. Asıl soru, BİRLEŞİK modelin")
    print("  alana-özel modellere ne kadar yakın kaldığıdır.")
    print("=" * 90)

    save_model(model_unified, args.model)

    names = [n for n, keep in zip(FEATURE_NAMES, model_unified["mask"]) if keep]
    order = np.argsort(-np.abs(model_unified["w"]))[:6]
    print("\n  Birleşik modelde en etkili özellikler:")
    for i in order:
        print(f"    {names[i]:<22} ağırlık {model_unified['w'][i]:+.2f}")

    out_csv = "unified_test_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["kisi", "tur", "skor", "alan"])
        writer.writeheader()
        for r in eval_records(model_unified, sig_test):
            r["alan"] = "imza"; writer.writerow(r)
        for r in eval_records(model_unified, writer_test):
            r["alan"] = "yazi"; writer.writerow(r)

    print(f"\n  Birleşik model: {args.model}   |   Test skorları: {out_csv}\n")


def cmd_compare(args):
    model = load_model(args.model)
    fa, fb = extract_features(args.a), extract_features(args.b)
    if fa is None or fb is None:
        print("Görüntülerden biri okunamadı.")
        return
    s = score_pair(model, fa, fb)
    print(f"\n  '{os.path.basename(args.a)}' ve '{os.path.basename(args.b)}' için")
    print(f"  benzerlik skoru (birleşik model): {s:.1f} / 100")
    print("  Bu, ortalamada anlamlı ama tek başına karar için yeterli olmayan bir")
    print("  sinyaldir; resmi bir adli görüş yerine geçmez.\n")


def main():
    ap = argparse.ArgumentParser(description="İmza + genel el yazısı - birleşik model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("sig_train_dir")
    f.add_argument("sig_test_dir")
    f.add_argument("writer_dir")
    f.add_argument("--sig-limit", type=int, default=8)
    f.add_argument("--writer-limit", type=int, default=6)
    f.add_argument("--test-frac", type=float, default=0.25)
    f.add_argument("--model", default=MODEL_PATH_DEFAULT)
    f.set_defaults(func=cmd_fit)

    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--model", default=MODEL_PATH_DEFAULT)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
