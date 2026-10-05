"""
Genel El Yazısı - Yazar Doğrulama (Writer Verification)

signature_verifier.py'deki özellik çıkarımını (extract_features) ve model
altyapısını (logistic regression, AUC, cache) OLDUĞU GİBİ kullanır. Farkı
veri yükleme mantığıdır: bu veri setinde "sahte" (_forg) klasörü YOK, çünkü
imza taklidi değil, genel el yazısı örnekleri söz konusu. Bu yüzden:
  - AYNI YAZAR çiftleri (pozitif): bir kişinin klasöründeki sayfalar arası
  - FARKLI YAZAR çiftleri (negatif): rastgele iki farklı kişinin sayfaları arası

Beklenen klasör yapısı:
    base_dir/S01/*.png (veya jpg)
    base_dir/S02/*.png
    ...

Kullanım:
    python writer_verifier.py fit "eng_data\\English Handwritten Pages Dataset\\English Handwritten Pages Dataset" --limit 6 --test-frac 0.25
    python writer_verifier.py compare notA.png notB.png
"""

import os
import sys
import csv
import random
import argparse
import itertools

import numpy as np

from signature_verifier import (
    extract_features, cached_features, _load_cache, _save_cache,
    fit_logistic, make_mask, FEATURE_NAMES,
)
from batch_validation_multi import auc, cluster_bootstrap_auc

MODEL_PATH_DEFAULT = "writer_model.json"


# ---------------------------------------------------------------------------
# VERİ YÜKLEME
# ---------------------------------------------------------------------------

def discover_writers(base_dir: str) -> list:
    """Görüntü dosyası içeren her alt klasörü bir 'yazar' sayar."""
    exts = (".png", ".jpg", ".jpeg")
    writers = []
    for name in sorted(os.listdir(base_dir)):
        folder = os.path.join(base_dir, name)
        if os.path.isdir(folder) and any(f.lower().endswith(exts) for f in os.listdir(folder)):
            writers.append(name)
    return writers


def list_images(folder: str, limit: int) -> list:
    exts = (".png", ".jpg", ".jpeg")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    return [os.path.join(folder, f) for f in files[:limit]]


def load_writer_features(base_dir: str, limit: int, label: str = "") -> dict:
    """Her yazar için {yazar_id: [(yol, özellik), ...]} döner."""
    cache = _load_cache()
    data = {}
    failed = 0
    writers = discover_writers(base_dir)
    for i, writer in enumerate(writers, start=1):
        items = []
        for path in list_images(os.path.join(base_dir, writer), limit):
            feats = cached_features(path, cache)
            if feats is None:
                failed += 1
                continue
            items.append((path, feats))
        if len(items) >= 2:
            data[writer] = items
        print(f"  [{label}] {i}/{len(writers)} yazar işlendi", end="\r")
    _save_cache(cache)
    print(f"  [{label}] {len(data)} yazar kullanılabilir (>=2 görüntü), {failed} görüntü okunamadı" + " " * 10)
    return data


# ---------------------------------------------------------------------------
# ÇİFT OLUŞTURMA: aynı yazar (pozitif) / farklı yazar (negatif, örneklenmiş)
# ---------------------------------------------------------------------------

def build_pairs(data: dict, neg_per_writer: int = None, seed: int = 0) -> list:
    """
    (tur, yol_a, yol_b, özellik_a, özellik_b) çiftleri üretir.
    Negatif (farklı yazar) çiftleri, pozitif sayısına yakın olacak şekilde
    RASTGELE örneklenir - aksi halde negatifler kombinatorik patlar.
    """
    rng = random.Random(seed)
    writers = list(data)
    pairs = []

    pos_count_by_writer = {}
    for writer, items in data.items():
        for (pa, fa), (pb, fb) in itertools.combinations(items, 2):
            pairs.append(("ayni-yazar", pa, pb, fa, fb))
        pos_count_by_writer[writer] = len(items) * (len(items) - 1) // 2

    for writer, items in data.items():
        n_neg = neg_per_writer if neg_per_writer is not None else max(1, pos_count_by_writer[writer])
        others = [w for w in writers if w != writer]
        if not others:
            continue
        for _ in range(n_neg):
            other = rng.choice(others)
            pa, fa = rng.choice(items)
            pb, fb = rng.choice(data[other])
            pairs.append(("farkli-yazar", pa, pb, fa, fb))

    return pairs


def split_writers(data: dict, test_frac: float, seed: int = 0) -> tuple:
    writers = sorted(data)
    random.Random(seed).shuffle(writers)
    n_test = max(1, int(len(writers) * test_frac))
    test_writers = set(writers[:n_test])
    train = {w: v for w, v in data.items() if w not in test_writers}
    test = {w: v for w, v in data.items() if w in test_writers}
    return train, test


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def fit_from_pairs(pairs: list, mask=None) -> dict:
    if mask is None:
        mask = make_mask()
    all_feats = [fa for _, _, _, fa, _ in pairs] + [fb for _, _, _, _, fb in pairs]
    std = np.std(all_feats, axis=0)
    std[std < 1e-6] = 1.0

    X = np.array([np.abs(fa - fb) / std for _, _, _, fa, fb in pairs])
    y = np.array([1.0 if tur == "ayni-yazar" else 0.0 for tur, _, _, _, _ in pairs])

    w, b = fit_logistic(X[:, mask], y)
    print(f"  Eğitim çifti sayısı: {len(y)}  (aynı yazar: {int(sum(y))}, farklı yazar: {int(len(y) - sum(y))})")
    return {"std": std, "w": w, "b": b, "mask": mask}


def score_pair(model: dict, fa: np.ndarray, fb: np.ndarray) -> float:
    x = (np.abs(fa - fb) / model["std"])[model["mask"]]
    z = float(np.dot(x, model["w"]) + model["b"])
    return 100.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def evaluate(model: dict, pairs: list) -> list:
    records = []
    for tur, pa, pb, fa, fb in pairs:
        s = score_pair(model, fa, fb)
        records.append({"kisi": os.path.basename(os.path.dirname(pa)), "tur": tur, "skor": s})
    return records


def summarize(name: str, records: list):
    pos = [r["skor"] for r in records if r["tur"] == "ayni-yazar"]
    neg = [r["skor"] for r in records if r["tur"] != "ayni-yazar"]
    a = auc(pos, neg)
    renamed = [{"kisi": r["kisi"], "tur": "gercek-gercek" if r["tur"] == "ayni-yazar" else "gercek-sahte",
               "skor": r["skor"]} for r in records]
    low, high = cluster_bootstrap_auc(renamed)
    print(f"  {name:<28} AUC {a:.3f}   %95 GA: {low:.3f} - {high:.3f}   "
          f"(aynı yazar n={len(pos)}, farklı yazar n={len(neg)})")


# ---------------------------------------------------------------------------
# KOMUTLAR
# ---------------------------------------------------------------------------

def cmd_fit(args):
    print("\n[1/3] Veri yükleniyor...")
    data = load_writer_features(args.base_dir, args.limit, "tümü")
    if len(data) < 6:
        print("Yeterli yazar bulunamadı.")
        return

    print(f"[2/3] Yazarlar train/test olarak ayrılıyor (%{int(args.test_frac*100)} test)...")
    train_data, test_data = split_writers(data, args.test_frac)
    print(f"  Train: {len(train_data)} yazar   Test: {len(test_data)} yazar")

    print("[3/3] Model eğitiliyor (yalnızca train yazarlarıyla)...\n")
    train_pairs = build_pairs(train_data, seed=1)
    model = fit_from_pairs(train_pairs)

    import json
    with open(args.model, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURE_NAMES, "std": model["std"].tolist(),
                   "w": model["w"].tolist(), "b": model["b"],
                   "mask": model["mask"].tolist()}, f)

    test_pairs = build_pairs(test_data, seed=2)
    test_records = evaluate(model, test_pairs)
    train_records = evaluate(model, train_pairs)

    print("\n" + "=" * 78)
    print(f"  SONUÇLAR  (train: {len(train_data)} yazar, test: {len(test_data)} yazar, "
          f"yazar başına ≤{args.limit} sayfa)")
    print("=" * 78)
    summarize("TEST - yazar doğrulama", test_records)
    summarize("TRAIN - yazar doğrulama (iyimser)", train_records)
    print("\n  Not: Bu veri seti İMZA DEĞİL, genel el yazısı notlarıdır. 'Farklı yazar'")
    print("  çiftleri rastgele örneklenmiştir, kasıtlı taklit içermez - bu yüzden")
    print("  sonuç imza-taklit senaryosuyla birebir kıyaslanmamalıdır.")

    out_csv = "writer_test_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["kisi", "tur", "skor"])
        writer.writeheader()
        writer.writerows(test_records)
    print(f"\n  Test çift skorları: {out_csv}   |   Model: {args.model}")
    print("=" * 78 + "\n")


def cmd_compare(args):
    import json
    with open(args.model, encoding="utf-8") as f:
        raw = json.load(f)
    model = {"std": np.array(raw["std"]), "w": np.array(raw["w"]), "b": raw["b"],
             "mask": np.array(raw["mask"], dtype=bool)}

    fa, fb = extract_features(args.a), extract_features(args.b)
    if fa is None or fb is None:
        print("Görüntülerden biri okunamadı.")
        return
    s = score_pair(model, fa, fb)
    print(f"\n  '{os.path.basename(args.a)}' ve '{os.path.basename(args.b)}' için")
    print(f"  benzerlik skoru: {s:.1f} / 100\n")


def main():
    ap = argparse.ArgumentParser(description="Genel el yazısı - yazar doğrulama")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("base_dir")
    f.add_argument("--limit", type=int, default=6, help="yazar başına en fazla sayfa")
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
