import cv2
import numpy as np
import pywt
from algorithms.utils import _imread_safe

def check_smart_fft(image_path, z_threshold=15):
    """
    FFT: Z-skoru + yuksek frekans enerji orani.

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

    # Merkez ve capraz maskeleme
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

    # Enerji orani
    total_energy = np.sum(magnitude_spectrum)
    hf_energy = np.sum(valid_pixels)
    hf_energy_ratio = hf_energy / total_energy if total_energy > 0 else 0

    # Moiré tespiti
    is_recaptured = False
    if z_score > z_threshold and bright_pixels_count < 200:
        is_recaptured = True

    # Puanlama — AI'da yuksek frekans enerjisi genellikle dusuktur
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

def analyze_dct_spectrum(image_path):
    """
    DCT Spektrum Analizi.
    8x8 bloklarda DCT uygular, AC katsayi istatistikleri cikarir.

    Kalibrasyon:
        AI portre: dct_kurtosis ≈ 479-524, dct_std dusuk
        Gercek: dct_kurtosis ≈ 414, dct_std yuksek
        AI manzara: dct_kurtosis ≈ 106

    Ana ayirt edici: DCT yuksek frekans AC enerji yogunlugu.
    AI portreler dusuk yuksek frekans AC enerjisi uretir.

    Returns:
        tuple: (dct_visual: ndarray, ai_score: int, stats: dict)
    """
    img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 50, {}

    img_float = img.astype(np.float64)
    h, w = img.shape
    block_size = 8

    # Blok bazli DCT analizi
    hf_energies = []  # Her blogun yuksek frekans enerjisi
    total_energies = []  # Her blogun toplam enerjisi
    all_hf_coeffs = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = img_float[y:y+block_size, x:x+block_size]
            dct_block = cv2.dct(block)

            # Yuksek frekans: sag alt ucgen (index toplami > 4)
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

    # Yuksek frekans enerji oranlari
    hf_ratios = []
    for hf, te in zip(hf_energies, total_energies):
        if te > 0:
            hf_ratios.append(hf / te)

    mean_hf_ratio = np.mean(hf_ratios) if hf_ratios else 0
    std_hf_ratio = np.std(hf_ratios) if hf_ratios else 0

    # HF katsayilarinin standart sapmasi
    hf_std = np.std(all_hf_coeffs) if all_hf_coeffs else 0

    # Puanlama (DUZELTILMIS YON)
    # Kalibrasyon:
    #   Gercek: mean_hf_ratio ≈ 0.016 (cok dusuk — dogal bokeh/blur)
    #   AI portre: mean_hf_ratio ≈ 0.064-0.086 (yapay doku tekrarlari)
    #   AI hatali: mean_hf_ratio ≈ 0.079
    #   AI manzara: mean_hf_ratio ≈ 0.124
    # AI HF ratio > Gercek → Yuksek ratio = AI suphesi
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

    # HF katsayi std: AI portre dusuk (1.7-1.9), Gercek orta (3.3), AI detayli yuksek (4.0+)
    if hf_std < 2.0:
        score = min(100, score + 12)
    elif hf_std < 3.0:
        score = min(100, score + 5)

    # Block HF orani varyasyonu (AI'da daha dusuk)
    if std_hf_ratio < 0.05:
        score = min(100, score + 5)
    elif std_hf_ratio > 0.12:
        score = max(0, score - 5)

    # Gorsellestirme: Block bazli HF enerji haritasi
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

def analyze_wavelet(image_path):
    """
    Wavelet Alt-Bant Analizi (Haar).

    Kalibrasyon:
        AI portre: hh_std ≈ 1.2-1.3, hh_ratio ≈ 0.000026-0.000039
        Gercek: hh_std ≈ 1.6, hh_ratio ≈ 0.000027
        AI+ekran: hh_std ≈ 1.4

    hh_std gercek (1.6) ve AI (1.2-1.3) arasinda kucuk fark var.

    Returns:
        tuple: (wavelet_visual: ndarray, ai_score: int, stats: dict)
    """
    img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 50, {}

    img_float = img.astype(np.float64)

    # Daubechies (db2) wavelet decomposition (3 seviye)
    # Dogal goruntu istatistiklerini haar'a gore daha iyi modeller.
    coeffs = pywt.wavedec2(img_float, 'db2', level=3)

    # Seviyelerdeki HH bantlarinin enerjisi
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

    # En ince seviyenin (seviye 1) HH bandi
    finest_hh = coeffs[-1][2]  # Son seviye = en ince = orijinale en yakin
    finest_hh_std = np.std(finest_hh)
    finest_hh_energy = np.sum(finest_hh ** 2)
    finest_hh_ratio = finest_hh_energy / total_energy if total_energy > 0 else 0

    # Detail/Total enerji orani
    detail_ratio = total_detail_energy / total_energy if total_energy > 0 else 0

    # Puanlama
    # Wavelet icin iki uc risklidir:
    # 1) Asiri puruzsuz, detay fakiri uretilmis alanlar.
    # 2) Gemini/manzara tipi gorsellerde gorulen asiri yogun sentetik detay.
    # Orta bant daha dogal kabul edilir.
    if finest_hh_std < 0.5:
        score = 88
    elif finest_hh_std < 0.8:
        score = 72
    elif finest_hh_std < 1.1:
        score = 48
    elif finest_hh_std < 2.0:
        score = 28
    elif finest_hh_std < 2.5:
        score = 55
    else:
        score = 72

    # Detail ratio bonusu
    if detail_ratio < 0.0005:
        score = min(100, score + 12)
    elif detail_ratio > 0.004:
        score = min(100, score + 12)
    elif detail_ratio > 0.003:
        score = min(100, score + 6)

    # Gorsellestirme
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
