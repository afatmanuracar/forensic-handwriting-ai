"""
El Yazısı Karşılaştırma Uygulaması (Masaüstü GUI) - v2, otomatik alan tespitli

Eski app_signature_compare.py'nin yerini alır. Fark: bu sürüm artık
domain_router.py'yi kullanıyor - yani sadece imza değil, imza VEYA genel
el yazısı notu, hangisi olursa olsun otomatik tespit edip doğru modeli
(signature_model.json ya da writer_model.json) kendisi seçiyor. Kullanıcı
"bu imza mı yazı mı" diye elle belirtmek zorunda değil.

ÖNKOŞUL: Şu üç dosyanın da üretilmiş olması gerekir (en kolayı tek komut):
    python run_all.py sign_data\\train sign_data\\test "eng_data\\...\\English Handwritten Pages Dataset"
Bu, signature_model.json, writer_model.json ve domain_classifier.json'ı üretir.

ÖNEMLİ UYARI (uygulama ekranında da sürekli görünür):
İmza alanında test AUC'si ~0.85, genel el yazısında ~0.83 (rastgele bir
'aynı kişi' çiftine, rastgele bir 'farklı kişi' çiftinden daha yüksek skor
verme olasılığı). Bu ORTALAMADA anlamlı bir sinyaldir ama TEK BİR örnek
için güvenilir bir karar vermeye yetmez. Resmi bir adli görüş yerine geçmez.

Çalıştırma:
    python app_handwriting_compare.py
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from signature_verifier import extract_features
from domain_router import (
    load_classifier, predict_domain_prob, load_pair_model, score_pair,
    CLASSIFIER_PATH_DEFAULT, SIGNATURE_MODEL_PATH, WRITER_MODEL_PATH,
)

THUMB_SIZE = (260, 150)


def verdict_text(score: float) -> str:
    if score >= 75:
        return "Yüksek benzerlik - aynı kişiden çıkmış olma ihtimali güçlü."
    if score >= 55:
        return "Orta düzey benzerlik - tek başına karar için yeterli değil."
    if score >= 35:
        return "Düşük-orta benzerlik - farklı kişilerden çıkmış olabilir."
    return "Düşük benzerlik - farklı kişilerden çıkmış olma ihtimali güçlü."


class HandwritingCompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("El Yazısı Karşılaştırma - Otomatik Alan Tespitli")
        self.root.configure(padx=14, pady=14)

        self.paths = {"a": None, "b": None}
        self.thumbnails = {"a": None, "b": None}

        self.classifier = None
        self.sig_model = None
        self.writer_model = None
        self.load_error = None
        self._load_models()

        panels = tk.Frame(root)
        panels.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.image_labels = {}
        self.file_labels = {}
        for i, key in enumerate(("a", "b")):
            frame = tk.LabelFrame(panels, text=f"Örnek {key.upper()}", padx=8, pady=8)
            frame.grid(row=0, column=i, padx=8)

            img_label = tk.Label(frame, text="(görüntü seçilmedi)", width=32, height=8,
                                  bg="#f0f0f0", relief="groove")
            img_label.pack()
            self.image_labels[key] = img_label

            file_label = tk.Label(frame, text="", font=("Arial", 9), wraplength=240)
            file_label.pack(pady=(4, 6))
            self.file_labels[key] = file_label

            tk.Button(frame, text="Görsel Seç...", command=lambda k=key: self.choose_image(k)
                      ).pack()

        self.compare_button = tk.Button(root, text="Karşılaştır", command=self.compare,
                                         bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                         state="disabled")
        self.compare_button.grid(row=1, column=0, columnspan=2, pady=10, ipadx=20, ipady=4)

        self.domain_label = tk.Label(root, text="", font=("Arial", 10), fg="#333")
        self.domain_label.grid(row=2, column=0, columnspan=2)

        self.result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
        self.result_label.grid(row=3, column=0, columnspan=2)

        self.verdict_label = tk.Label(root, text="", font=("Arial", 11), justify="left", wraplength=560)
        self.verdict_label.grid(row=4, column=0, columnspan=2, pady=(2, 10))

        warning = (
            "UYARI: İmza alanında test AUC'si ~0.85, genel el yazısında ~0.83\n"
            "(rastgele bir 'aynı kişi' çiftinin, rastgele bir 'farklı kişi' çiftinden\n"
            "daha yüksek skor alma olasılığı). Bu ORTALAMADA anlamlı bir sinyaldir;\n"
            "TEK bir örnek için güvenilir karar vermeye yetmez ve resmi bir adli\n"
            "görüş yerine GEÇMEZ. Akademik/deneysel bir prototiptir."
        )
        tk.Label(root, text=warning, font=("Arial", 9), fg="#8a1f11", justify="left",
                 wraplength=560).grid(row=5, column=0, columnspan=2, pady=(6, 0))

        if self.load_error:
            messagebox.showwarning(
                "Model(ler) bulunamadı",
                f"{self.load_error}\n\n"
                "En kolay çözüm, tek komutla hepsini üretmek:\n"
                "python run_all.py sign_data\\train sign_data\\test "
                "\"eng_data\\...\\English Handwritten Pages Dataset\""
            )

    def _load_models(self):
        missing = [p for p in (CLASSIFIER_PATH_DEFAULT, SIGNATURE_MODEL_PATH, WRITER_MODEL_PATH)
                   if not os.path.exists(p)]
        if missing:
            self.load_error = f"Eksik dosya(lar): {', '.join(missing)}"
            return
        try:
            self.classifier = load_classifier(CLASSIFIER_PATH_DEFAULT)
            self.sig_model = load_pair_model(SIGNATURE_MODEL_PATH)
            self.writer_model = load_pair_model(WRITER_MODEL_PATH)
        except Exception as e:
            self.load_error = str(e)

    def choose_image(self, key: str):
        path = filedialog.askopenfilename(
            title=f"Örnek {key.upper()} - görüntü seç",
            filetypes=[("Görüntü dosyaları", "*.png *.jpg *.jpeg *.PNG *.JPG *.JPEG")]
        )
        if not path:
            return

        self.paths[key] = path
        self.file_labels[key].config(text=os.path.basename(path))

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(THUMB_SIZE)
            photo = ImageTk.PhotoImage(img)
            self.thumbnails[key] = photo
            self.image_labels[key].config(image=photo, text="", width=THUMB_SIZE[0], height=THUMB_SIZE[1])
        except Exception:
            self.image_labels[key].config(text="(önizleme yüklenemedi)")

        self._update_compare_button()
        self.domain_label.config(text="")
        self.result_label.config(text="")
        self.verdict_label.config(text="")

    def _update_compare_button(self):
        ready = self.paths["a"] and self.paths["b"] and self.classifier is not None
        self.compare_button.config(state="normal" if ready else "disabled")

    def compare(self):
        if self.classifier is None:
            messagebox.showerror("Model yok", self.load_error or "Modeller yüklenemedi.")
            return

        self.result_label.config(text="Hesaplanıyor...", fg="black")
        self.domain_label.config(text="")
        self.verdict_label.config(text="")
        self.root.update_idletasks()

        try:
            fa = extract_features(self.paths["a"])
            fb = extract_features(self.paths["b"])
        except Exception as e:
            messagebox.showerror("Hata", f"Özellik çıkarımı başarısız: {e}")
            self.result_label.config(text="")
            return

        if fa is None or fb is None:
            messagebox.showerror("Hata", "Görüntülerden birinde el yazısı/imza tespit edilemedi.")
            self.result_label.config(text="")
            return

        prob_a = predict_domain_prob(self.classifier, fa)
        prob_b = predict_domain_prob(self.classifier, fb)
        label_a = "imza" if prob_a >= 0.5 else "genel yazı"
        label_b = "imza" if prob_b >= 0.5 else "genel yazı"

        if label_a == label_b:
            domain = label_a
            model = self.sig_model if domain == "imza" else self.writer_model
            score = score_pair(model, fa, fb)

            self.domain_label.config(text=f"Tespit edilen alan: {domain}  "
                                           f"(A: %{max(prob_a, 1-prob_a)*100:.0f}, "
                                           f"B: %{max(prob_b, 1-prob_b)*100:.0f} emin)")
            color = "#1a7a1a" if score >= 55 else "#8a1f11"
            self.result_label.config(text=f"Benzerlik Skoru: {score:.1f} / 100", fg=color)
            self.verdict_label.config(text=verdict_text(score))
        else:
            sig_score = score_pair(self.sig_model, fa, fb)
            writer_score = score_pair(self.writer_model, fa, fb)
            self.domain_label.config(
                text=f"UYARI: Alan tespiti uyuşmuyor (A: {label_a}, B: {label_b}) - "
                     f"iki model de gösteriliyor:"
            )
            self.result_label.config(
                text=f"İmza modeli: {sig_score:.1f}   |   Genel yazı modeli: {writer_score:.1f}",
                fg="#8a1f11"
            )
            self.verdict_label.config(
                text="Görüntülerin aynı türde (ikisi de imza ya da ikisi de not) olduğundan "
                     "emin ol; sonuç yalnızca doğru alan seçildiğinde anlamlıdır."
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingCompareApp(root)
    root.mainloop()
