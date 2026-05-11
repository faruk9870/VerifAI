# 🔍 VerifAI

VerifAI, dijital görsellerin orijinalliğini doğrulamak ve yapay zeka (AI) tarafından üretilip üretilmediğini tespit etmek için geliştirilmiş bir görüntü analiz sistemidir.

Şu anda proje bir masaüstü uygulaması (GUI) olarak geliştirilmektedir. (İlerleyen aşamalarda web tabanlı bir arayüze dönüştürülmesi planlanmaktadır.)

## 🚀 Özellikler

- **Görsel Analizi:** Görsellerdeki manipülasyonları, difüzyon kalıntılarını ve yapay zeka izlerini derin öğrenme tabanlı analiz eder.
- **Kullanıcı Arayüzü:** Kullanımı kolay grafik arayüz (GUI) üzerinden görselleri seçip detaylı analiz sonuçlarını görebilirsiniz.

## 🛠️ Kurulum ve Kullanım

Projeyi izole bir ortamda sorunsuz çalıştırmak için Python sanal ortamı (`venv`) kullanmanız gerekmektedir.

**1. Sanal Ortam (venv) Oluşturma ve Aktifleştirme:**
Proje dizininde terminali açın ve sanal ortamı oluşturup aktifleştirin:
```bash
python -m venv venv

# Windows için:
venv\Scripts\activate
```

**2. Gerekli Kütüphaneleri Yükleme:**
Sanal ortam aktifken gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

**3. Uygulamayı Başlatma:**
Kurulum tamamlandıktan sonra ana programı (arayüzü) başlatmak için:
```bash
python src/main.py
```
