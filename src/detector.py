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
    # ─────────────────────────────────────────────────
    # 5. DCT SPEKTRUM ANALİZİ (YENİ — KALİBRE EDİLMİŞ)
    # ─────────────────────────────────────────────────
    @staticmethod
    def analyze_dct_spectrum(image_path):
        """
        DCT Spektrum Analizi.
        8x8 bloklarda DCT uygular, AC katsayı istatistikleri çıkarır.

        Kalibrasyon:
            AI portre: dct_kurtosis ≈ 479-524, dct_std düşük
            Gerçek: dct_kurtosis ≈ 414, dct_std yüksek
            AI manzara: dct_kurtosis ≈ 106

        Ana ayırt edici: DCT yüksek frekans AC enerji yoğunluğu.
        AI portreler düşük yüksek frekans AC enerjisi üretir.

        Returns:
            tuple: (dct_visual: ndarray, ai_score: int, stats: dict)
        """
        img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, 50, {}

        img_float = img.astype(np.float64)
        h, w = img.shape
        block_size = 8

        # Blok bazlı DCT analizi
        hf_energies = []  # Her bloğun yüksek frekans enerjisi
        total_energies = []  # Her bloğun toplam enerjisi
        all_hf_coeffs = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img_float[y:y+block_size, x:x+block_size]
                dct_block = cv2.dct(block)

                # Yüksek frekans: sağ alt üçgen (index toplamı > 4)
                hf_energy = 0
                total_energy = 0
                for i in range(8):
                    for j in range(8):
                        val = abs(dct_block[i, j])
                        total_energy += val
                        if i + j > 4:
                            hf_energy += val
                            all_hf_coeffs.append(dct_block[i, j])

                hf_energies.append(hf_energy)
                total_energies.append(total_energy)

        if not hf_energies:
            return None, 50, {}

        # Yüksek frekans enerji oranları
        hf_ratios = []
        for hf, te in zip(hf_energies, total_energies):
            if te > 0:
                hf_ratios.append(hf / te)

        mean_hf_ratio = np.mean(hf_ratios) if hf_ratios else 0
        std_hf_ratio = np.std(hf_ratios) if hf_ratios else 0

        # HF katsayılarının standart sapması
        hf_std = np.std(all_hf_coeffs) if all_hf_coeffs else 0

        # Puanlama (DÜZELTİLMİŞ YÖN)
        # Kalibrasyon:
        #   Gerçek: mean_hf_ratio ≈ 0.016 (çok düşük — doğal bokeh/blur)
        #   AI portre: mean_hf_ratio ≈ 0.064-0.086 (yapay doku tekrarları)
        #   AI hatali: mean_hf_ratio ≈ 0.079
        #   AI manzara: mean_hf_ratio ≈ 0.124
        # AI HF ratio > Gerçek → Yüksek ratio = AI şüphesi
        if mean_hf_ratio > 0.10:
            score = 80
        elif mean_hf_ratio > 0.07:
            score = 68
        elif mean_hf_ratio > 0.04:
            score = 45
        elif mean_hf_ratio > 0.02:
            score = 25
        else:
            score = 10

        # HF katsayı std: AI portre düşük (1.7-1.9), Gerçek orta (3.3), AI detaylı yüksek (4.0+)
        if hf_std < 2.0:
            score = min(100, score + 12)
        elif hf_std < 3.0:
            score = min(100, score + 5)

        # Block HF oranı varyasyonu (AI'da daha düşük)
        if std_hf_ratio < 0.05:
            score = min(100, score + 5)
        elif std_hf_ratio > 0.12:
            score = max(0, score - 5)

        # Görselleştirme: Block bazlı HF enerji haritası
        dct_energy_map = np.zeros((h, w), dtype=np.float64)
        idx = 0
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                if idx < len(hf_ratios):
                    dct_energy_map[y:y+block_size, x:x+block_size] = hf_ratios[idx]
                idx += 1

        max_val = np.max(dct_energy_map) if np.max(dct_energy_map) > 0 else 1
        dct_visual = ((dct_energy_map / max_val) * 255).astype(np.uint8)

        stats = {
            "mean_hf_ratio": round(mean_hf_ratio, 4),
            "std_hf_ratio": round(std_hf_ratio, 4),
            "hf_std": round(hf_std, 2),
        }

        return dct_visual, score, stats
    # ─────────────────────────────────────────────────
    # 6. WAVELET ALT-BANT ANALİZİ (YENİ — KALİBRE)
    # ─────────────────────────────────────────────────
    @staticmethod
    def analyze_wavelet(image_path):
        """
        Wavelet Alt-Bant Analizi (Haar).

        Kalibrasyon:
            AI portre: hh_std ≈ 1.2-1.3, hh_ratio ≈ 0.000026-0.000039
            Gerçek: hh_std ≈ 1.6, hh_ratio ≈ 0.000027
            AI+ekran: hh_std ≈ 1.4

        hh_std gerçek (1.6) ve AI (1.2-1.3) arasında küçük fark var.

        Returns:
            tuple: (wavelet_visual: ndarray, ai_score: int, stats: dict)
        """
        img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, 50, {}

        img_float = img.astype(np.float64)

        # Daubechies (db2) wavelet decomposition (3 seviye)
        # Doğal görüntü istatistiklerini haar'a göre daha iyi modeller.
        coeffs = pywt.wavedec2(img_float, 'db2', level=3)

        # Seviyelerdeki HH bantlarının enerjisi
        hh_energies = []
        hh_stds = []
        total_detail_energy = 0

        for level_coeffs in coeffs[1:]:
            cH, cV, cD = level_coeffs
            hh_energies.append(np.sum(cD ** 2))
            hh_stds.append(np.std(cD))
            total_detail_energy += np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)

        ll_energy = np.sum(coeffs[0] ** 2)
        total_energy = ll_energy + total_detail_energy

        # En ince seviyenin (seviye 1) HH bandı
        finest_hh = coeffs[-1][2]  # Son seviye = en ince = orijinale en yakın
        finest_hh_std = np.std(finest_hh)
        finest_hh_energy = np.sum(finest_hh ** 2)
        finest_hh_ratio = finest_hh_energy / total_energy if total_energy > 0 else 0

        # Detail/Total enerji oranı
        detail_ratio = total_detail_energy / total_energy if total_energy > 0 else 0

        # Puanlama
        # AI: finest_hh_std < 0.6, detail_ratio düşük
        # Gerçek: finest_hh_std > 1.0, detail_ratio yüksek
        if finest_hh_std < 0.5:
            score = 88
        elif finest_hh_std < 0.8:
            score = 72
        elif finest_hh_std < 1.0:
            score = 48
        elif finest_hh_std < 2.0:
            score = 28
        else:
            score = 12

        # Detail ratio bonusu
        if detail_ratio < 0.0005:
            score = min(100, score + 12)
        elif detail_ratio > 0.003:
            score = max(0, score - 10)

        # Görselleştirme
        hh_visual = np.abs(finest_hh)
        max_val = np.max(hh_visual) if np.max(hh_visual) > 0 else 1
        hh_visual = ((hh_visual / max_val) * 255).astype(np.uint8)
        hh_visual = cv2.resize(hh_visual, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        stats = {
            "finest_hh_std": round(finest_hh_std, 2),
            "finest_hh_ratio": round(finest_hh_ratio, 6),
            "detail_ratio": round(detail_ratio, 6),
        }

        return hh_visual, score, stats
    # ─────────────────────────────────────────────────
    # 7. RENK İSTATİSTİK ANALİZİ (YENİ)
    # ─────────────────────────────────────────────────
    @staticmethod
    def analyze_color_statistics(image_path):
        """
        Renk kanalları arası korelasyon ve histogram entropisi.

        Kalibrasyon:
            AI portre: corr ≈ 0.945-0.948
            Gerçek: corr ≈ 0.946
            AI manzara: corr ≈ 0.801

        Not: Korelasyon tek başına yeterli değil, entropi ve saturation ile birleştirilecek.

        Returns:
            tuple: (color_visual: ndarray, ai_score: int, stats: dict)
        """
        img = _imread_safe(image_path)
        if img is None:
            return None, 50, {}

        b, g, r = cv2.split(img)

        # Kanal korelasyonları
        rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        gb_corr = np.corrcoef(g.flatten(), b.flatten())[0, 1]
        mean_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3

        # Histogram entropisi
        def channel_entropy(channel):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))

        r_entropy = channel_entropy(r)
        g_entropy = channel_entropy(g)
        b_entropy = channel_entropy(b)
        mean_entropy = (r_entropy + g_entropy + b_entropy) / 3

        # Saturation analizi
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        sat_std = np.std(saturation)
        sat_mean = np.mean(saturation)

        # YCbCr Krominans Analizi (Cb, Cr kanalları varyansı)
        # Doğal fotoğraflarda bayer filtresi nedeniyle Cb/Cr kanallarında spesifik varyans bulunur.
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        cr = ycbcr[:, :, 1]
        cb = ycbcr[:, :, 2]
        cr_std = np.std(cr)
        cb_std = np.std(cb)
        chrominance_var = (cr_std + cb_std) / 2

        # Renk histogram pürüzsüzlüğü (AI genelde daha smooth histogram üretir)
        hist_smoothness = 0
        for ch in [r, g, b]:
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
            hist_diff = np.diff(hist)
            hist_smoothness += np.std(hist_diff)
        hist_smoothness /= 3

        # Puanlama (DÜZELTİLMİŞ)
        # Kalibrasyon:
        #   Gerçek: sat_std ≈ 24, hist_smooth ≈ 1200
        #   AI portre: sat_std ≈ 41-47, hist_smooth ≈ 823-1117  
        #   AI hatali: sat_std ≈ 46, hist_smooth ≈ 330
        score = 50

        # Saturation std: AI YÜKSEK (40-47), Gerçek DÜŞÜK (24)
        # Bu en güçlü renk ayrıştırıcısı
        if sat_std > 45:
            score += 20
        elif sat_std > 38:
            score += 12
        elif sat_std > 30:
            score += 5
        elif sat_std < 26:
            score -= 15

        # Krominans varyansı (AI üretimi genellikle krominans düzleminde daha pürüzsüzdür)
        # Doğal görüntülerde renk gürültüsü daha fazladır.
        if chrominance_var < 5.0:
            score += 10
        elif chrominance_var < 8.0:
            score += 5
        elif chrominance_var > 15.0:
            score -= 8

        # Histogram pürüzsüzlüğü: Gerçek yüksek (1200), AI hatali düşük (330)
        # Düşük smoothness = AI olabilir
        if hist_smoothness < 400:
            score += 10
        elif hist_smoothness < 700:
            score += 3
        elif hist_smoothness > 1000:
            score -= 8

        # Düşük entropi → AI
        if mean_entropy < 6.5:
            score += 8
        elif mean_entropy < 7.0:
            score += 3
        elif mean_entropy > 7.5:
            score -= 5

        score = max(0, min(100, score))

        # Görselleştirme
        sat_visual = cv2.equalizeHist(saturation)

        stats = {
            "mean_corr": round(mean_corr, 4),
            "mean_entropy": round(mean_entropy, 2),
            "sat_std": round(sat_std, 2),
            "sat_mean": round(sat_mean, 2),
            "chrominance_var": round(chrominance_var, 2),
            "hist_smoothness": round(hist_smoothness, 2),
        }

        return sat_visual, score, stats