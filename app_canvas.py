"""
Canlı El Yazısı Tanıma Uygulaması (Masaüstü GUI)
Fare ile bir karakter çizip anlık olarak CNN modelinin tahminini görürsün.

NEDEN BU YAKLAŞIM?
Fotoğraf + segmentasyon yönteminde harflerin birbirinden ayrılması (contour
analysis) güvenilmez olabiliyordu. Bu uygulamada kullanıcı zaten her karakteri
kendi temiz kutusuna çiziyor - yani segmentasyon problemi baştan ortadan
kalkıyor. Model doğrudan, güvenilir girdiyle çalışıyor.

Çalıştırma:
    python app_canvas.py
"""

import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

from config import LANGUAGE_CONFIGS, ACTIVE_LANGUAGE
from predict import prepare_character_image, load_trained_model

CHARSET = LANGUAGE_CONFIGS[ACTIVE_LANGUAGE]["charset"]

CANVAS_SIZE = 280   # ekranda görünen tuval boyutu (piksel)
BRUSH_SIZE = 14      # fırça kalınlığı


class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("El Yazısı Tanıma - Canlı Demo")

        print("Model yükleniyor, lütfen bekleyin...")
        self.model = load_trained_model()
        print("Model hazır!")

        # --- Çizim tuvali (ekranda görünen) ---
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                 bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_stroke)

        # --- Arka planda tutulan gerçek görüntü (tahmin için kullanılacak) ---
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        # --- Butonlar ---
        tk.Button(root, text="Tanı", command=self.predict, width=12,
                  bg="#4CAF50", fg="white").grid(row=1, column=0, padx=5, pady=5)
        tk.Button(root, text="Temizle", command=self.clear, width=12,
                  bg="#f44336", fg="white").grid(row=1, column=1, padx=5, pady=5)

        # --- Sonuç etiketleri ---
        self.result_label = tk.Label(root, text="Bir karakter çiz ve 'Tanı'ya bas",
                                      font=("Arial", 14))
        self.result_label.grid(row=2, column=0, columnspan=3, pady=10)

        self.detail_label = tk.Label(root, text="", font=("Consolas", 11), justify="left")
        self.detail_label.grid(row=3, column=0, columnspan=3, pady=5)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                     width=BRUSH_SIZE, fill="black",
                                     capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=BRUSH_SIZE)
        self.last_x, self.last_y = x, y

    def reset_stroke(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Bir karakter çiz ve 'Tanı'ya bas")
        self.detail_label.config(text="")

    def predict(self):
        img_array = np.array(self.image)
        inverted = 255 - img_array  # siyah çizim -> beyaz çizim (model formatı: foreground=beyaz)

        if inverted.max() == 0:
            self.result_label.config(text="Önce bir şey çiz!")
            return

        # Çizilen alanı sıkıca kırp (boş kenar boşluklarını at)
        ys, xs = np.nonzero(inverted)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1
        cropped = inverted[y1:y2, x1:x2]

        prepared = prepare_character_image(cropped)
        predictions = self.model.predict(prepared, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1][:3]

        best_char = CHARSET[top_indices[0]]
        best_conf = predictions[top_indices[0]] * 100
        self.result_label.config(text=f"Tahmin: '{best_char}'  (%{best_conf:.1f})")

        detail_lines = [
            f"{rank}. '{CHARSET[idx]}'   %{predictions[idx] * 100:.1f}"
            for rank, idx in enumerate(top_indices, start=1)
        ]
        self.detail_label.config(text="\n".join(detail_lines))


if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()
