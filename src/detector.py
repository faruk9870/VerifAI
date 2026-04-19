"""
VerifAI — İleri Seviye Adli Bilişim Tespit Motoru
8 bağımsız algoritma ile AI üretimi görsel tespiti.

Algoritmalar:
    1. EXIF / Metadata Analizi
    2. Gürültü Residual Analizi (Geliştirilmiş)
    3. Akıllı ELA (Kenar Korumalı)
    4. FFT Frekans Analizi (Geliştirilmiş)
    5. DCT Spektrum Analizi (YENİ)
    6. Wavelet Alt-Bant Analizi (YENİ)
    7. Renk İstatistik Analizi (YENİ)
    8. GLCM Doku Analizi (YENİ)
    9. Kenar Tutarlılık Analizi (YENİ)
"""

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import pywt


def _imread_safe(path, flags=cv2.IMREAD_COLOR):
    """
    Türkçe karakter içeren dosya yollarını da okuyabilen imread fonksiyonu.
    Windows'ta cv2.imread() UTF-8 yolları desteklemez.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    return img


class ManipulationDetector:
    """10 algoritmali ana tespit sinifi."""
    # ─────────────────────────────────────────────────
    # 1. EXIF / METADATA ANALİZİ
    # ─────────────────────────────────────────────────
    @staticmethod
    def analyze_metadata(image_path):
        """
        EXIF metadata analizi.
        Kamera donanım izleri (Make, Model, Software) arar.

        Returns:
            tuple: (has_exif: bool, exif_info: str, ai_score: int)
        """
        try:
            image = Image.open(image_path)
            exifdata = image.getexif()
            if not exifdata:
                return False, "EXIF YOK: Veriler silinmiş veya hiç üretilmemiş.", 80

            metadata = {}
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    try:
                        data = data.decode(errors='ignore')
                    except Exception:
                        pass
                metadata[tag] = data

            hardware_traces = ['Make', 'Model', 'Software']
            found_traces = {t: metadata.get(t) for t in hardware_traces if t in metadata}

            # AI yazılım damgalarını ara
            ai_software_keywords = ['stable diffusion', 'midjourney', 'dall-e', 'comfyui',
                                    'automatic1111', 'novelai', 'nai diffusion']
            software_val = str(metadata.get('Software', '')).lower()
            for kw in ai_software_keywords:
                if kw in software_val:
                    return True, f"AI YAZILIM İZİ: {metadata.get('Software', '')}", 95

            if len(found_traces) > 0:
                make = found_traces.get('Make', '')
                model = found_traces.get('Model', '')
                return True, f"KAMERA İZİ: {make} {model}".strip(), 5
            return True, "EXIF var ama donanım izi yok.", 55
        except Exception:
            return False, "EXIF Okuma Hatası", 60
    # ─────────────────────────────────────────────────
    # 2. GÜRÜLTÜ RESİDUAL ANALİZİ (GELİŞTİRİLMİŞ)
    # ─────────────────────────────────────────────────
    @staticmethod
    def detect_noise_residual(image_path):
        """
        Gürültü residual analizi (median blur tabanlı, Wiener yerine).
        Gerçek fotoğraflarda sensör gürültüsü rastgele ve yüksek varyansa sahip.
        AI görsellerde gürültü çok düşük veya tekdüze.

        Kalibrasyon verileri:
            AI portre: noise_std ≈ 2.2-2.8
            Gerçek portre: noise_std ≈ 4.5
            AI manzara: noise_std ≈ 7.7

        Returns:
            tuple: (noise_map: ndarray, ai_score: int, stats: dict)
        """
        img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, 50, {}

        img_float = img.astype(np.float64)

        # Bilateral filter ile kenarları koruyarak gürültüyü çıkar
        # Bu yöntem doğal gürültüyü yapay pürüzsüzlükten daha iyi ayırır.
        denoised = cv2.bilateralFilter(img, 9, 75, 75).astype(np.float64)
        noise_residual = img_float - denoised

        # İstatistiksel metrikler
        noise_std = np.std(noise_residual)
        noise_mean = np.mean(np.abs(noise_residual))

        # Bölgesel gürültü tutarlılık kontrolü
        h, w = img.shape
        block_size = min(64, max(8, h // 4), max(8, w // 4))

        block_stds = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise_residual[y:y+block_size, x:x+block_size]
                block_stds.append(np.std(block))

        block_std_variation = np.std(block_stds) if block_stds else 0

        # ── Puanlama (Kalibre)
        # Birincil metrik: noise_std
        # AI: noise_std < 2.0 → yüksek puan (çok pürüzsüz)
        # Gerçek (ön kamera): noise_std ≈ 3.0-5.0
        # Gerçek (arka kamera): noise_std > 4 → düşük puan
        if noise_std < 2.0:
            score = 88
        elif noise_std < 2.5:
            score = 72
        elif noise_std < 3.5:
            score = 45
        elif noise_std < 5.0:
            score = 25
        elif noise_std < 7.0:
            score = 15
        else:
            score = 5

        # İkincil metrik: Block tutarlılığı (EN GÜÇLÜ AYRIŞTIRICI)
        # AI: block_std_variation ≈ 1.2-1.9 (her yerde aynı gürültü)
        # Gerçek: block_std_variation ≈ 3.0+ (bölgesel farklılıklar)
        if block_std_variation < 1.5:
            score = min(100, score + 12)
        elif block_std_variation < 2.0:
            score = min(100, score + 5)
        elif block_std_variation > 3.0:
            score = max(0, score - 18)
        elif block_std_variation > 2.5:
            score = max(0, score - 10)

        # Görselleştirme
        noise_visual = np.abs(noise_residual)
        max_val = np.max(noise_visual) if np.max(noise_visual) > 0 else 1
        noise_visual = ((noise_visual / max_val) * 255).astype(np.uint8)
        noise_visual = cv2.equalizeHist(noise_visual)

        stats = {
            "noise_std": round(noise_std, 2),
            "noise_mean": round(noise_mean, 2),
            "block_std_var": round(block_std_variation, 2),
        }

        return noise_visual, score, stats