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
    # ─────────────────────────────────────────────────
    # 3. AKILLI ELA (KENAR KORUMALI)
    # ─────────────────────────────────────────────────
    @staticmethod
    def detect_smart_ela(image_path, quality=90):
        """
        Yeni nesil ELA: Canny Edge ile maskelenmiş hata seviyesi analizi.

        Returns:
            tuple: (ela_map: ndarray, ai_score: int, stats: dict)
        """
        original = _imread_safe(image_path)
        if original is None:
            return None, 50, {}

        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # ELA hesapla
        _, encoded_img = cv2.imencode('.jpg', original, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        compressed = cv2.imdecode(encoded_img, 1)
        diff = cv2.absdiff(original, compressed)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Doğal kenarları maskele
        edges = cv2.Canny(gray_original, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        flat_areas_mask = cv2.bitwise_not(edges_dilated) / 255.0

        # ELA'yı düz alan maskesi ile çarp
        smart_ela = gray_diff * flat_areas_mask

        # İstatistikler
        ela_mean = np.mean(smart_ela)
        ela_max = np.max(smart_ela)
        ela_diff = ela_max - ela_mean
        ela_std = np.std(smart_ela)

        # Puanlama
        if ela_diff > 240:
            score = 85
        elif ela_diff > 200:
            score = 60
        elif ela_diff > 150:
            score = 40
        else:
            score = 15

        # Görselleştirme
        max_val = np.max(smart_ela) if np.max(smart_ela) > 0 else 1
        ela_visual = ((smart_ela / max_val) * 255.0).astype(np.uint8)

        stats = {
            "ela_mean": round(ela_mean, 2),
            "ela_max": round(float(ela_max), 2),
            "ela_diff": round(float(ela_diff), 2),
            "ela_std": round(ela_std, 2),
        }

        return ela_visual, score, stats
    # ─────────────────────────────────────────────────
    # 4. FFT FREKANS ANALİZİ (GELİŞTİRİLMİŞ)
    # ─────────────────────────────────────────────────
    @staticmethod
    def check_smart_fft(image_path, z_threshold=15):
        """
        FFT: Z-skoru + yüksek frekans enerji oranı.

        Returns:
            tuple: (fft_visual: ndarray, ai_score: int, is_recaptured: bool, stats: dict)
        """
        img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, 50, False, {}

        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        fft_visual = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2

        # Merkez ve çapraz maskeleme
        high_freq_region = magnitude_spectrum.copy()
        mask_size = 60
        if rows > mask_size * 2 and cols > mask_size * 2:
            high_freq_region[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0

        cross_thickness = 5
        high_freq_region[crow-cross_thickness:crow+cross_thickness, :] = 0
        high_freq_region[:, ccol-cross_thickness:ccol+cross_thickness] = 0

        valid_pixels = high_freq_region[high_freq_region > 0]
        if len(valid_pixels) == 0:
            return fft_visual, 50, False, {}

        hf_mean = np.mean(valid_pixels)
        hf_std = np.std(valid_pixels)
        if hf_std == 0:
            hf_std = 1

        z_score = (np.max(valid_pixels) - hf_mean) / hf_std
        bright_pixels_count = np.sum(valid_pixels > (hf_mean + 3 * hf_std))

        # Enerji oranı
        total_energy = np.sum(magnitude_spectrum)
        hf_energy = np.sum(valid_pixels)
        hf_energy_ratio = hf_energy / total_energy if total_energy > 0 else 0

        # Moiré tespiti
        is_recaptured = False
        if z_score > z_threshold and bright_pixels_count < 200:
            is_recaptured = True

        # Puanlama — AI'da yüksek frekans enerjisi genellikle düşüktür
        if hf_energy_ratio < 0.25:
            score = 75
        elif hf_energy_ratio < 0.35:
            score = 55
        elif hf_energy_ratio < 0.45:
            score = 35
        else:
            score = 15

        stats = {
            "z_score": round(z_score, 2),
            "bright_pixels": int(bright_pixels_count),
            "hf_energy_ratio": round(hf_energy_ratio, 4),
            "is_recaptured": is_recaptured,
        }

        return fft_visual, score, is_recaptured, stats