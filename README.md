# Handwriting AI — El Yazısı Tanıma & Adli Yazı Karşılaştırma Sistemi

İki ana bileşenden oluşan bir sistem:

1. **OCR (Optik Karakter Tanıma):** Kağıttaki el yazısını dijital metne çevirir.
2. **Adli Yazı Karşılaştırma:** İki el yazısı örneğinin (imza ya da genel not)
   aynı kişiden çıkıp çıkmadığını istatistiksel olarak değerlendirir.

---

## Kurulum

```bash
git clone https://github.com/afatmanuracar/forensic-handwriting-ai.git
cd forensic-handwriting-ai
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`tkinter` (masaüstü uygulamaları için) Python ile birlikte gelir, ayrıca
kurulum gerekmez.

---

## Proje Yapısı

### A) OCR Modülü

| Dosya | Görev |
|---|---|
| `config.py` | Dil-bağımsız merkezi ayarlar (karakter seti, model boyutu) |
| `preprocessing.py` | Gri tonlama, gürültü temizleme, binarizasyon, deskew, vuruş birleştirme |
| `segmentation.py` | Satır → kelime → karakter bölütleme (kontur analizi) |
| `data_loader.py` | EMNIST veri seti yükleyici |
| `model.py` | CNN mimarisi (TensorFlow/Keras) |
| `train.py` | CNN eğitim script'i |
| `predict.py` | Tam görüntü → metin tahmin pipeline'ı |
| `demo_single_character.py` | Tek karakter tanıma demosu (en güvenilir bileşen) |
| `app_canvas.py` | Fare ile çizip anlık tanıma yapan masaüstü uygulaması |

**Bilinen sınırlama:** Kelime/cümle segmentasyonu, harflerin kendi içinde
kopuk vuruşlarla çizildiği el yazılarında güvenilmez sonuç verebilir (kontur
tabanlı klasik yöntemin doğal zayıflığı). Tek karakter tanıma güçlüdür
(EMNIST test setinde ~%87 doğruluk, kendi el yazısı örneklerinde %81-100
güvenle doğru tahmin).

### B) Adli Yazı Karşılaştırma Modülü

| Dosya | Görev |
|---|---|
| `signature_verifier.py` | İmza verisiyle (gerçek/sahte) model eğitimi ve değerlendirmesi |
| `writer_verifier.py` | Genel el yazısı (imza-dışı) verisiyle yazar doğrulama |
| `unified_verifier.py` | İki veriyi tek modelde birleştirme denemesi (sonuç: performans kaybı, bkz. Sonuçlar) |
| `domain_router.py` | Görüntünün imza mı genel yazı mı olduğunu otomatik tespit edip doğru modeli seçen yönlendirici |
| `batch_validation.py` / `batch_validation_multi.py` | Eski (4 elle-seçilmiş özellikli) yöntemin toplu doğrulaması |
| `graphology.py` | Yazı eğimi, satır düzgünlüğü, harf aralığı, kalem kalınlığı analizi |
| `interpretation.py` | Adli karşılaştırma + **geleneksel (bilimsel geçerliliği olmayan) grafoloji yorumu** |
| `app_signature_compare.py` | *(eski, yerini app_handwriting_compare.py aldı)* |
| `app_handwriting_compare.py` | Otomatik alan tespitli, iki görüntü seçip karşılaştıran masaüstü uygulaması |
| `run_all.py` | Üç modeli (imza, genel yazı, yönlendirici) tek komutla eğiten ana script |

---

## Hızlı Başlangıç — Her Şeyi Tek Komutla Kur

```bash
python run_all.py sign_data\train sign_data\test "eng_data\English Handwritten Pages Dataset\English Handwritten Pages Dataset"
```

Bu komut sırayla:
1. İmza modelini eğitir → `signature_model.json`
2. Genel el yazısı modelini eğitir → `writer_model.json`
3. Alan yönlendiricisini eğitir → `domain_classifier.json`
4. Hepsinin test AUC'sini özetler

Bitince:
```bash
python app_handwriting_compare.py
```
ile GUI üzerinden iki görüntü seçip karşılaştırabilirsin, ya da terminalden:
```bash
python domain_router.py compare ornekA.png ornekB.png
```

---

## Veri Setleri (Kaynak Belirtme)

- **İmza verisi:** Kaggle "Signature Verification Dataset" (Robin Reni).
  55+ kişiden gerçek + sahte imza çiftleri.
- **Genel el yazısı verisi:** Mendeley Data, "English Handwritten Pages
  Dataset" (2026). ~90 yazardan, kişi başına birden fazla sayfa.
- **CNN eğitimi:** EMNIST (byclass split), `tensorflow_datasets` üzerinden
  otomatik indirilir.

> Raporda bu veri setlerine tam atıf vermeyi unutma (Kaggle sayfası ve
> Mendeley DOI'si).

---

## Sonuçlar (Özet)

| Bileşen | Metrik | Sonuç |
|---|---|---|
| CNN - tek karakter tanıma | Test doğruluğu (EMNIST) | ~%87 |
| İmza doğrulama (öğrenilmiş model) | Test AUC | 0.852 (%95 GA: 0.815–0.891) |
| İmza doğrulama (eski, 4 özellik) | Test AUC | 0.584 |
| Genel el yazısı doğrulama | Test AUC | 0.826 (%95 GA: 0.754–0.887) |
| Birleşik model (imza+yazı tek modelde) | Test AUC | 0.794 / 0.750 — **alana özel modellerden düşük** |
| Alan yönlendirici (imza vs genel yazı ayrımı) | Doğruluk / AUC | %100 / 1.000 |

**Önemli bulgu:** İki veriyi naif şekilde birleştirip tek bir model eğitmek,
her iki alanda da performans kaybına yol açtı. Bunun yerine, görüntünün
hangi alana ait olduğunu otomatik tespit edip doğru alana-özel modeli
çağıran bir yönlendirici (`domain_router.py`) kullanıldı — bu, hem yüksek
doğruluğu korudu hem de kullanıcının elle "imza mı yazı mı" seçmesi
gereksinimini ortadan kaldırdı.

**AUC nedir?** Rastgele seçilen bir "aynı kişi" çiftinin, rastgele seçilen
bir "farklı kişi/sahte" çiftinden daha yüksek benzerlik skoru alma
olasılığı. 0.5 = şans düzeyi, 1.0 = mükemmel ayrım. Bu projedeki AUC
değerleri (~0.83-0.85) **ortalamada anlamlı ama tek bir örnek için
güvenilir karar vermeye yetmeyen** bir düzeydedir.

---

## Önemli Bilimsel/Etik Not

`interpretation.py` içindeki `interpret_personality()` fonksiyonu,
yazıdan kişilik çıkarımı yapan geleneksel grafoloji yorumları üretir.
**Bu yöntemin bilimsel geçerliliği akademik psikolojide kanıtlanmamıştır**
(pseudoscience olarak sınıflandırılır). Kod ve çıktı bunu açıkça belirtir;
raporda da bu ayrımı netleştirmek gerekir — adli karşılaştırma (AUC ile
doğrulanmış) ile kişilik yorumu (geleneksel/kültürel referans) birbirinden
farklı güvenilirlik seviyelerindedir.

---

## Gelecek Geliştirmeler

- Kelime/cümle segmentasyonu için uçtan uca (CRNN + CTC tabanlı) model
- Türkçe karakter desteği (ç, ğ, ı, ö, ş, ü)
- Alan-farkındalıklı normalizasyon ile birleşik modelin iyileştirilmesi
- Daha büyük örneklemle (örn. 300+ test görüntüsü) AUC güven aralıklarının daraltılması
