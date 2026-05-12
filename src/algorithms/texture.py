import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from algorithms.utils import _imread_safe

def analyze_glcm_texture(image_path):
    """
    GLCM doku analizi.

    Kalibrasyon:
        AI portre: homo ≈ 0.527-0.566, contrast ≈ 13-17
        Gercek: homo ≈ 0.562, contrast ≈ 31
        AI manzara: homo ≈ 0.324, contrast ≈ 39

    Ana ayirt edici: contrast dusuklugu (AI portre < 20, Gercek > 25)

    Returns:
        tuple: (texture_visual: ndarray, ai_score: int, stats: dict)
    """
    img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 50, {}

    # Goruntuyu kucult
    max_dim = 512
    h, w = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        img_resized = img.copy()

    # GLCM
    levels = 64
    img_quantized = (img_resized / 256 * levels).astype(np.uint8)

    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_quantized, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)

    contrast = np.mean(graycoprops(glcm, 'contrast'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))

    # Puanlama (kalibre edilmis)
    score = 50

    # Dusuk contrast → AI portre
    # SADECE cok dusuk kontrasti AI say. On kameralar ~15-25 arasi kontrastla cekiyor.
    if contrast < 5:
        score += 25
    elif contrast < 8:
        score += 12
    elif contrast > 25:
        score -= 18

    # Yuksek homogeneity → AI (sadece asiri yuksek degerler)
    if homogeneity > 0.65:
        score += 10
    elif homogeneity > 0.55:
        score += 3
    elif homogeneity < 0.30:
        score -= 10

    # Yuksek energy → uniformik → AI (sadece asiri)
    if energy > 0.012:
        score += 8
    elif energy < 0.002:
        score -= 5

    score = max(0, min(100, score))

    # Gorsellestirme: block homogeneity haritasi
    block_sz = 32
    h_r, w_r = img_resized.shape
    homo_map = np.zeros_like(img_resized, dtype=np.float64)

    for y in range(0, h_r - block_sz, block_sz // 2):
        for x in range(0, w_r - block_sz, block_sz // 2):
            block = img_resized[y:y+block_sz, x:x+block_sz]
            block_q = (block / 256 * levels).astype(np.uint8)
            try:
                local_glcm = graycomatrix(block_q, distances=[1], angles=[0],
                                          levels=levels, symmetric=True, normed=True)
                local_homo = graycoprops(local_glcm, 'homogeneity')[0, 0]
                homo_map[y:y+block_sz, x:x+block_sz] = local_homo
            except Exception:
                pass

    max_val = np.max(homo_map) if np.max(homo_map) > 0 else 1
    texture_visual = ((homo_map / max_val) * 255).astype(np.uint8)
    texture_visual = cv2.resize(texture_visual, (w, h), interpolation=cv2.INTER_LINEAR)

    stats = {
        "contrast": round(contrast, 2),
        "correlation": round(correlation, 4),
        "energy": round(energy, 6),
        "homogeneity": round(homogeneity, 4),
    }

    return texture_visual, score, stats

def analyze_lbp_texture(image_path):
    """
    LBP (Local Binary Pattern) mikro-doku analizi.
    AI uretimi gorseller pikseller arasi komsuluklarda (mikro seviyede) 
    dogal kamera sensoru gurultusu uretemez ve cok uniform (tekduze) desenler olusturur.
    Gercek fotograflarda sensor gurultusu nedeniyle LBP histogrami daha daginiktir.
    
    Returns:
        tuple: (lbp_visual: ndarray, ai_score: int, stats: dict)
    """
    img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 50, {}
        
    radius = 1
    n_points = 8 * radius
    
    # 'uniform' metodunu kullaniyoruz, bu metod rotasyondan bagimsizdir ve temel doku yapilarini bulur
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    
    # LBP Histogrami hesapla
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # En baskin desenin yogunlugu (AI'da bu deger genelde daha yuksektir cunku goruntu puruzsuzdur)
    max_pattern_ratio = np.max(hist)
    
    # Histogram varyansi (AI'da dusuktur)
    hist_var = np.var(hist)
    
    # Puanlama Kalibrasyonu
    # max_pattern_ratio dusukse (< 0.25) -> Sikistirma yok, AI olma ihtimali cok yuksek
    # max_pattern_ratio yuksekse (> 0.35) -> Kamera/JPEG sikistirma izleri, gercek olma ihtimali yuksek
    
    score = 50
    
    if max_pattern_ratio < 0.20:
        score += 35
    elif max_pattern_ratio < 0.25:
        score += 20
    elif max_pattern_ratio > 0.45:
        score -= 30
    elif max_pattern_ratio > 0.35:
        score -= 15
        
    if hist_var < 0.002:
        score += 15
    elif hist_var > 0.010:
        score -= 10
        
    score = max(0, min(100, score))
    
    # Gorsellestirme
    lbp_visual = (lbp * (255.0 / (n_bins - 1))).astype(np.uint8)
    
    stats = {
        "max_pattern_ratio": round(max_pattern_ratio, 4),
        "lbp_hist_var": round(hist_var, 6),
    }
    
    return lbp_visual, score, stats