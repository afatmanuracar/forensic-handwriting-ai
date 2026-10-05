"""
İmza Doğrulama: Zengin Özellikler + Öğrenilmiş Karşılaştırma

Eski yöntem (interpretation.compare_documents) 4 elle seçilmiş özelliğe ve
elle ayarlanmış eşiklere dayanıyordu. Bu modül:

  1. İmzaya özel ön işleme yapar (eğim düzeltmesi YOK - eğim bilgisi korunur)
  2. ~29 ölçekten bağımsız özellik çıkarır
  3. İki imzanın özellik FARKLARINDAN "aynı kişi mi" olasılığını tahmin eden
     bir lojistik regresyon modelini TRAIN klasöründeki kişilerle eğitir
  4. Modeli hiç görmediği TEST klasöründeki kişilerde ölçer ve eski
     yöntemle aynı test verisinde karşılaştırır

ÖNEMLİ (bilimsel dürüstlük): test sonucunu gördükten sonra özellikleri veya
ayarları değiştirirsen test verisi "kirlenir" ve sonuç iyimser olur. Ayarları
yalnızca train üzerinde yap, test'i EN SON ve mümkünse tek sefer çalıştır.

Kullanım:
    python signature_verifier.py fit train test --limit 8
    python signature_verifier.py compare imza1.png imza2.png
"""

import os
import sys
import csv
import json
import pickle
import argparse
import itertools
import random

import numpy as np
import cv2

from batch_validation_multi import auc, cluster_bootstrap_auc

N_BINS = 8

FEATURE_NAMES = (
    ["en_boy_orani_log", "murekkep_yogunlugu", "kalem_kalinligi", "bilesen_sayisi_log",
     "egim_derece", "agirlik_merkezi_x", "agirlik_merkezi_y", "karmasiklik", "cevre_orani"]
    + [f"x_profil_{i}" for i in range(N_BINS)]
    + [f"y_profil_{i}" for i in range(N_BINS)]
    + [f"hu_{i + 1}" for i in range(4)]
)


# ---------------------------------------------------------------------------
# 1. ÖN İŞLEME + ÖZELLİK ÇIKARIMI
# ---------------------------------------------------------------------------

def load_signature_binary(path: str):
    """
    İmza görüntüsünü okur, Otsu ile ikili (mürekkep=255) hale getirir,
    küçük gürültü bileşenlerini temizler ve imzayı sıkıca kırpar.
    Eğim düzeltmesi yapılmaz (eğim, ayırt edici bir özelliktir).
    """
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(15, int(0.0002 * binary.shape[0] * binary.shape[1]))
    keep_ids = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep_ids[0] = False  # 0 = arka plan
    mask = keep_ids[labels]
    if not mask.any():
        return None

    cleaned = (mask * 255).astype(np.uint8)
    ys, xs = np.nonzero(cleaned)
    crop = cleaned[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return crop, int(keep_ids.sum())


def _profile(values: np.ndarray) -> list:
    total = values.sum()
    if total == 0:
        return [0.0] * N_BINS
    return [float(chunk.sum() / total) for chunk in np.array_split(values, N_BINS)]


def extract_features(path: str):
    """Bir imza görüntüsünden ölçekten bağımsız özellik vektörü çıkarır."""
    loaded = load_signature_binary(path)
    if loaded is None:
        return None
    crop, n_components = loaded

    h, w = crop.shape
    scale = float(np.sqrt(w * h))
    area = float(np.count_nonzero(crop))

    dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    stroke_width = float(2.0 * dist[crop > 0].mean())

    m = cv2.moments(crop, binaryImage=True)
    slant = 0.0 if abs(m["mu02"]) < 1e-6 else float(np.degrees(np.arctan(m["mu11"] / m["mu02"])))
    cx = m["m10"] / m["m00"] / w
    cy = m["m01"] / m["m00"] / h

    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))

    binary01 = (crop > 0).astype(np.float64)
    x_profile = _profile(binary01.sum(axis=0))
    y_profile = _profile(binary01.sum(axis=1))

    hu = cv2.HuMoments(m).flatten()[:4]
    hu_log = np.clip(-np.sign(hu) * np.log10(np.abs(hu) + 1e-30), -15, 15)

    features = [
        np.log(w / h),
        area / (w * h),
        stroke_width / scale,
        np.log1p(n_components),
        slant,
        cx,
        cy,
        (area / max(stroke_width, 1e-6)) / scale,
        perimeter / scale,
    ] + x_profile + y_profile + list(hu_log)

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. VERİ SETİ: KİŞİ KEŞFİ, ÇİFT OLUŞTURMA
# ---------------------------------------------------------------------------

CACHE_PATH = "feature_cache.pkl"
# Özellik çıkarım kodunu değiştirirsen bu sayıyı artır; eski önbellek otomatik geçersiz olur.
FEATURE_VERSION = 1


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict):
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"  [Uyarı] Önbellek yazılamadı: {e}")


def cached_features(path: str, cache: dict):
    """Dosya değişmediyse önbellekten, değiştiyse yeniden hesaplayarak özellik döner."""
    st = os.stat(path)
    key = os.path.abspath(path)
    stamp = (st.st_size, int(st.st_mtime), FEATURE_VERSION)
    hit = cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    feats = extract_features(path)
    cache[key] = (stamp, feats)
    return feats


def discover_persons(base_dir: str) -> list:
    """<kisi> ve <kisi>_forg klasör çiftlerini otomatik bulur."""
    persons = []
    for name in sorted(os.listdir(base_dir)):
        if name.endswith("_forg"):
            continue
        if os.path.isdir(os.path.join(base_dir, name)) and \
                os.path.isdir(os.path.join(base_dir, name + "_forg")):
            persons.append(name)
    return persons


def list_images(folder: str, limit: int) -> list:
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return [os.path.join(folder, f) for f in files[:limit]]


def load_person_data(base_dir: str, limit: int, label: str = "") -> dict:
    """Her kişi için gerçek ve sahte imzaların özelliklerini (önbellekleyerek) çıkarır."""
    data = {}
    failed = 0
    cache = _load_cache()
    persons = discover_persons(base_dir)
    for i, person in enumerate(persons, start=1):
        entry = {"gercek": [], "sahte": []}
        for key, folder in (("gercek", person), ("sahte", person + "_forg")):
            for path in list_images(os.path.join(base_dir, folder), limit):
                feats = cached_features(path, cache)
                if feats is None:
                    failed += 1
                    continue
                entry[key].append((path, feats))
        if len(entry["gercek"]) >= 2 and len(entry["sahte"]) >= 1:
            data[person] = entry
        print(f"  [{label}] {i}/{len(persons)} kişi işlendi", end="\r")
    _save_cache(cache)
    print(f"  [{label}] {len(data)} kişi kullanılabilir, {failed} görüntü okunamadı" + " " * 10)
    return data


def build_pairs(entry: dict) -> list:
    """(tür, yol_a, yol_b, özellik_a, özellik_b) çiftleri üretir."""
    pairs = []
    for (pa, fa), (pb, fb) in itertools.combinations(entry["gercek"], 2):
        pairs.append(("gercek-gercek", pa, pb, fa, fb))
    for (pa, fa) in entry["gercek"]:
        for (pb, fb) in entry["sahte"]:
            pairs.append(("gercek-sahte", pa, pb, fa, fb))
    return pairs


# ---------------------------------------------------------------------------
# 3. MODEL: L2 düzenlemeli, sınıf-dengeli lojistik regresyon (yalnızca numpy)
# ---------------------------------------------------------------------------

def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 10.0, lr: float = 0.2, iters: int = 2500):
    n, d = X.shape
    pos_rate = y.mean()
    if pos_rate in (0.0, 1.0):
        raise ValueError("Eğitim verisinde iki sınıf da bulunmalı.")
    sample_w = np.where(y == 1, 0.5 / pos_rate, 0.5 / (1 - pos_rate))

    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = (p - y) * sample_w
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


PEN_FEATURES = ["kalem_kalinligi", "murekkep_yogunlugu", "karmasiklik", "cevre_orani"]


def make_mask(keep_names=None, drop_names=None) -> np.ndarray:
    """Özellik adlarına göre bir kullan/kullanma maskesi üretir."""
    if keep_names is not None:
        return np.array([n in keep_names for n in FEATURE_NAMES])
    return np.array([n not in (drop_names or []) for n in FEATURE_NAMES])


def fit_model(train_data: dict, mask=None, verbose: bool = True) -> dict:
    if mask is None:
        mask = np.ones(len(FEATURE_NAMES), dtype=bool)
    all_feats = [f for entry in train_data.values() for _, f in entry["gercek"] + entry["sahte"]]
    std = np.std(all_feats, axis=0)
    std[std < 1e-6] = 1.0

    X, y = [], []
    for entry in train_data.values():
        for tur, _, _, fa, fb in build_pairs(entry):
            X.append(np.abs(fa - fb) / std)
            y.append(1.0 if tur == "gercek-gercek" else 0.0)

    w, b = fit_logistic(np.array(X)[:, mask], np.array(y))
    if verbose:
        print(f"  Eğitim çifti sayısı: {len(y)}  (aynı kişi: {int(sum(y))}, farklı/sahte: {int(len(y) - sum(y))})")
    return {"std": std, "w": w, "b": b, "mask": mask}


def score_pair(model: dict, fa: np.ndarray, fb: np.ndarray) -> float:
    """0-100 arası 'aynı kişi' skoru (yüksek = daha benzer)."""
    x = (np.abs(fa - fb) / model["std"])[model["mask"]]
    z = float(np.dot(x, model["w"]) + model["b"])
    return 100.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def save_model(model: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURE_NAMES, "std": model["std"].tolist(),
                   "w": model["w"].tolist(), "b": model["b"],
                   "mask": model["mask"].tolist()}, f)


def load_model(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    mask = np.array(raw.get("mask", [True] * len(raw["std"])), dtype=bool)
    return {"std": np.array(raw["std"]), "w": np.array(raw["w"]), "b": raw["b"], "mask": mask}


# ---------------------------------------------------------------------------
# 4. DEĞERLENDİRME
# ---------------------------------------------------------------------------

def evaluate_new(model: dict, data: dict):
    records, rows = [], []
    for person, entry in data.items():
        for tur, pa, pb, fa, fb in build_pairs(entry):
            s = score_pair(model, fa, fb)
            records.append({"kisi": person, "tur": tur, "skor": s})
            rows.append({"kisi": person, "tur": tur, "ornek_a": os.path.basename(pa),
                         "ornek_b": os.path.basename(pb), "yeni_skor": round(s, 2)})
    return records, rows


def evaluate_baseline(data: dict, rows: list):
    """Eski yöntemi (graphology + compare_documents) aynı çiftlerde çalıştırır."""
    from graphology import analyze_document
    from interpretation import compare_documents

    cache = {}

    def metrics(path):
        if path not in cache:
            try:
                cache[path] = analyze_document(path)
            except Exception:
                cache[path] = None
        return cache[path]

    records = []
    lookup = {}
    for person, entry in data.items():
        for tur, pa, pb, _, _ in build_pairs(entry):
            ma, mb = metrics(pa), metrics(pb)
            if ma is None or mb is None:
                continue
            try:
                s = compare_documents(ma, mb)["tutarlilik_skoru"]
            except Exception:
                continue
            records.append({"kisi": person, "tur": tur, "skor": s})
            lookup[(person, os.path.basename(pa), os.path.basename(pb))] = s

    for row in rows:
        row["eski_skor"] = lookup.get((row["kisi"], row["ornek_a"], row["ornek_b"]), "")
    return records


def summarize(name: str, records: list) -> tuple:
    pos = [r["skor"] for r in records if r["tur"] == "gercek-gercek"]
    neg = [r["skor"] for r in records if r["tur"] != "gercek-gercek"]
    a = auc(pos, neg)
    low, high = cluster_bootstrap_auc(records)
    print(f"  {name:<28} AUC {a:.3f}   %95 GA: {low:.3f} - {high:.3f}   "
          f"(aynı kişi n={len(pos)}, sahte n={len(neg)})")
    return a, low, high


# ---------------------------------------------------------------------------
# 5. KOMUTLAR
# ---------------------------------------------------------------------------

def cmd_fit(args):
    print("\n[1/4] Train verisi yükleniyor...")
    train = load_person_data(args.train_dir, args.limit, "train")
    print("[2/4] Test verisi yükleniyor...")
    test = load_person_data(args.test_dir, args.limit, "test")
    if not train or not test:
        print("Yeterli veri bulunamadı (klasör yapısını kontrol et).")
        return

    print("[3/4] Model eğitiliyor (yalnızca train kişileriyle)...")
    model = fit_model(train)
    save_model(model, args.model)

    print("[4/4] Değerlendiriliyor...\n")
    train_records, _ = evaluate_new(model, train)
    test_records, rows = evaluate_new(model, test)

    baseline_records = None
    if not args.no_baseline:
        print("  Eski yöntem test verisinde çalıştırılıyor (biraz sürebilir)...")
        baseline_records = evaluate_baseline(test, rows)

    print("\n" + "=" * 78)
    print(f"  SONUÇLAR  (train: {len(train)} kişi, test: {len(test)} kişi, kişi başına ≤{args.limit} görüntü)")
    print("=" * 78)
    if baseline_records:
        summarize("TEST - eski yöntem (4 özellik)", baseline_records)
    summarize("TEST - yeni yöntem", test_records)
    summarize("TRAIN - yeni yöntem (iyimser)", train_records)
    print("\n  Not: TRAIN sonucu modelin eğitildiği veridir, güvenilir ölçüt DEĞİLDİR.")
    print("  Gerçek performans TEST satırlarıdır (model bu kişileri hiç görmedi).")

    names = [n for n, keep in zip(FEATURE_NAMES, model["mask"]) if keep]
    order = np.argsort(-np.abs(model["w"]))[:6]
    print("\n  Yeni modelde en etkili özellikler (|ağırlık| sırasıyla):")
    for i in order:
        print(f"    {names[i]:<22} ağırlık {model['w'][i]:+.2f}")
    print("  (Birbirine bağlı özelliklerin - kalınlık, yoğunluk, çevre - işaretleri tek başına yorumlanmamalı.)")

    out_csv = "signature_test_results.csv"
    fields = ["kisi", "tur", "ornek_a", "ornek_b", "yeni_skor", "eski_skor"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row.setdefault("eski_skor", "")
            writer.writerow(row)
    print(f"\n  Test çift skorları: {out_csv}   |   Model: {args.model}")
    print("=" * 78 + "\n")


def cmd_ablate(args):
    """
    Kişi bazlı çapraz doğrulama ile özellik gruplarının katkısını ölçer.
    YALNIZCA train verisini kullanır (test'e dokunmaz).
    Soru: model gerçekten imzanın ŞEKLİNİ mi öğreniyor, yoksa ağırlıklı olarak
    kalem izi/mürekkep özelliklerine (kalınlık, yoğunluk) mi yaslanıyor?
    """
    print("\n[1/2] Train verisi yükleniyor...")
    train = load_person_data(args.train_dir, args.limit, "train")
    if len(train) < args.folds * 2:
        print("Çapraz doğrulama için yeterli kişi yok.")
        return

    persons = sorted(train)
    random.Random(0).shuffle(persons)
    folds = [persons[i::args.folds] for i in range(args.folds)]

    variants = {
        "tümü (29 özellik)": make_mask(),
        "kalem-izi ÇIKARILDI": make_mask(drop_names=PEN_FEATURES),
        "yalnız kalem-izi (4)": make_mask(keep_names=PEN_FEATURES),
        "yalnız kalem kalınlığı": make_mask(keep_names=["kalem_kalinligi"]),
    }

    print(f"[2/2] {args.folds}-katlı, kişi bazlı çapraz doğrulama ({len(train)} kişi)...\n")
    print("=" * 78)
    print("  ÖZELLİK GRUBU KATKISI (yalnızca train, hiç görülmemiş kişilerde ölçüm)")
    print("=" * 78)
    for name, mask in variants.items():
        records = []
        for k in range(args.folds):
            held = set(folds[k])
            tr = {p: train[p] for p in persons if p not in held}
            te = {p: train[p] for p in held}
            model = fit_model(tr, mask, verbose=False)
            recs, _ = evaluate_new(model, te)
            records.extend(recs)
        summarize(name, records)
    print("=" * 78 + "\n")


def cmd_compare(args):
    model = load_model(args.model)
    fa, fb = extract_features(args.a), extract_features(args.b)
    if fa is None or fb is None:
        print("Görüntülerden biri okunamadı veya içinde imza bulunamadı.")
        return
    s = score_pair(model, fa, fb)
    print(f"\n  '{os.path.basename(args.a)}' ve '{os.path.basename(args.b)}' için")
    print(f"  benzerlik skoru: {s:.1f} / 100")
    print("  Bu skor ortalamada anlamlı ama bireysel karar için güvenilir değildir;")
    print("  resmi bir adli görüş yerine geçmez.\n")


def main():
    ap = argparse.ArgumentParser(description="İmza doğrulama (zengin özellik + öğrenilmiş model)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="train ile eğit, test ile değerlendir")
    f.add_argument("train_dir")
    f.add_argument("test_dir")
    f.add_argument("--limit", type=int, default=8, help="klasör başına en fazla görüntü")
    f.add_argument("--no-baseline", action="store_true", help="eski yöntemle karşılaştırmayı atla")
    f.add_argument("--model", default="signature_model.json")
    f.set_defaults(func=cmd_fit)

    a = sub.add_parser("ablate", help="özellik gruplarının katkısını train'de çapraz doğrulamayla ölç")
    a.add_argument("train_dir")
    a.add_argument("--limit", type=int, default=8)
    a.add_argument("--folds", type=int, default=5)
    a.set_defaults(func=cmd_ablate)

    c = sub.add_parser("compare", help="eğitilmiş modelle iki imzayı karşılaştır")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--model", default="signature_model.json")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
