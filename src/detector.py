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