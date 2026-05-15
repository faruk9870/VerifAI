import cv2
import numpy as np
from scipy.stats import kurtosis
from algorithms.utils import _imread_safe

def analyze_color_statistics(image_path):
    """
    Renk kanallari arasi korelasyon ve histogram entropisi.

    Kalibrasyon:
        AI portre: corr ≈ 0.945-0.948
        Gercek: corr ≈ 0.946
        AI manzara: corr ≈ 0.801

    Not: Korelasyon tek basina yeterli degil, entropi ve saturation ile birlestirilecek.

    Returns:
        tuple: (color_visual: ndarray, ai_score: int, stats: dict)
    """
    img = _imread_safe(image_path)
    if img is None:
        return None, 50, {}

    b, g, r = cv2.split(img)

    # Kanal korelasyonlari
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

    # YCbCr Krominans Analizi (Cb, Cr kanallari varyansi)
    # Dogal fotograflarda bayer filtresi nedeniyle Cb/Cr kanallarinda spesifik varyans bulunur.
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cr = ycbcr[:, :, 1]
    cb = ycbcr[:, :, 2]
    cr_std = np.std(cr)
    cb_std = np.std(cb)
    chrominance_var = (cr_std + cb_std) / 2

    # Renk histogram puruzsuzlugu (AI genelde daha smooth histogram uretir)
    hist_smoothness = 0
    for ch in [r, g, b]:
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
        hist_diff = np.diff(hist)
        hist_smoothness += np.std(hist_diff)
    hist_smoothness /= 3

    # Puanlama (DUZELTILMIS)
    # Kalibrasyon:
    #   Gercek: sat_std ≈ 24, hist_smooth ≈ 1200
    #   AI portre: sat_std ≈ 41-47, hist_smooth ≈ 823-1117  
    #   AI hatali: sat_std ≈ 46, hist_smooth ≈ 330
    score = 50

    # Saturation std: AI YUKSEK (40-47), Gercek DUSUK (24)
    # Bu en guclu renk ayristiricisi
    if sat_std > 45:
        score += 20
    elif sat_std > 38:
        score += 12
    elif sat_std > 30:
        score += 5
    elif sat_std < 26:
        score -= 15

    # Krominans varyansi (AI uretimi genellikle krominans duzleminde daha puruzsuzdur)
    # Dogal goruntulerde renk gurultusu daha fazladir.
    if chrominance_var < 5.0:
        score += 10
    elif chrominance_var < 8.0:
        score += 5
    elif chrominance_var > 15.0:
        score -= 8

    # Histogram puruzsuzlugu: Gercek yuksek (1200), AI hatali dusuk (330)
    # Dusuk smoothness = AI olabilir
    if hist_smoothness < 400:
        score += 10
    elif hist_smoothness < 700:
        score += 3
    elif hist_smoothness > 1000:
        score -= 8

    # Dusuk entropi → AI
    if mean_entropy < 6.5:
        score += 8
    elif mean_entropy < 7.0:
        score += 3
    elif mean_entropy > 7.5:
        score -= 5

    score = max(0, min(100, score))

    # Gorsellestirme
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

def analyze_edge_consistency(image_path):
    """
    Kenar keskinligi ve tutarlilik analizi.

    Kalibrasyon:
        AI portre: lap_var ≈ 34-61
        Gercek: lap_var ≈ 138
        AI manzara: lap_var ≈ 359 (detayli manzara)

    En guclu ayirt edici: Laplacian varyansi.
    AI portre: < 80, Gercek: > 100

    Returns:
        tuple: (edge_visual: ndarray, ai_score: int, stats: dict)
    """
    img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 50, {}

    # Laplacian varyansi
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    lap_var = np.var(laplacian)

    # Sobel kenarlari
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Gradient kurtosis — EN GUCLU YENI AYRISTIRICI
    # Kalibrasyon:
    #   Gercek: grad_kurtosis ≈ 46 (cok sivri — dogal kenar dagilimi)
    #   AI portre: grad_kurtosis ≈ 35-44 (yuksek ama biraz dusuk)
    #   AI hatali: grad_kurtosis ≈ 17 (cok dusuk — yapay duzgunluk)
    #   AI manzara: grad_kurtosis ≈ 14 (en dusuk)
    grad_kurtosis = kurtosis(sobel_mag.flatten())

    # Kenar yogunlugu
    canny = cv2.Canny(img, 50, 150)
    edge_density = np.sum(canny > 0) / canny.size

    # Bolgesel kenar varyasyonu
    h, w = img.shape
    block_size = min(64, max(8, h // 4), max(8, w // 4))

    block_edge_densities = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = canny[y:y+block_size, x:x+block_size]
            block_edge_densities.append(np.sum(block > 0) / block.size)

    edge_var = np.std(block_edge_densities) if block_edge_densities else 0

    # ── Puanlama (CIFT METRIK: lap_var + grad_kurtosis)
    
    # Birincil: Gradient kurtosis
    # Dusuk kurtosis = yapay gradient dagilimi = AI
    if grad_kurtosis < 10:
        score = 92
    elif grad_kurtosis < 15:
        score = 75
    elif grad_kurtosis < 25:
        score = 55
    elif grad_kurtosis < 40:
        score = 35
    else:
        score = 15

    # Ikincil: Laplacian varyansi (sadece portre/duz sahneler icin)
    if lap_var < 50:
        score = min(100, score + 12)
    elif lap_var < 80:
        score = min(100, score + 5)
    elif lap_var > 200:
        score = max(0, score - 8)

    # Kenar yogunlugu bonusu
    if edge_density < 0.02:
        score = min(100, score + 5)
    elif edge_density > 0.10:
        score = max(0, score - 5)

    score = max(0, min(100, score))

    # Gorsellestirme
    max_val = np.max(sobel_mag) if np.max(sobel_mag) > 0 else 1
    edge_visual = ((sobel_mag / max_val) * 255).astype(np.uint8)

    stats = {
        "lap_var": round(lap_var, 2),
        "grad_kurtosis": round(grad_kurtosis, 2),
        "edge_density": round(edge_density, 4),
        "edge_regional_var": round(edge_var, 4),
    }

    return edge_visual, score, stats