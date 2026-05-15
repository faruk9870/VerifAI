import cv2
import numpy as np
from algorithms.utils import _imread_safe

def detect_noise_residual(image_path):
    """
    Gurultu residual analizi (median blur tabanli, Wiener yerine).
    Gercek fotograflarda sensor gurultusu rastgele ve yuksek varyansa sahip.
    AI gorsellerde gurultu cok dusuk veya tekduze.

    Kalibrasyon verileri:
        AI portre: noise_std ≈ 2.2-2.8
        Gercek portre: noise_std ≈ 4.5
        AI manzara: noise_std ≈ 7.7

    Returns:
        tuple: (noise_map: ndarray, ai_score: int, stats: dict)
    """
    img = _imread_safe(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 50, {}

    img_float = img.astype(np.float64)

    # Bilateral filter ile kenarlari koruyarak gurultuyu cikar
    # Bu yontem dogal gurultuyu yapay puruzsuzlukten daha iyi ayirir.
    denoised = cv2.bilateralFilter(img, 9, 75, 75).astype(np.float64)
    noise_residual = img_float - denoised

    # Istatistiksel metrikler
    noise_std = np.std(noise_residual)
    noise_mean = np.mean(np.abs(noise_residual))

    # Sahne dokusu (cim, tuy, yaprak vb.) residual'i yapay sekilde yukseltir.
    # Asil gurultu karari kenarsiz/duz alanlardan alinir.
    edges = cv2.Canny(img, 50, 150)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    flat_mask = (gradient_mag < 12) & (edges == 0)
    flat_area_ratio = np.mean(flat_mask)

    if np.count_nonzero(flat_mask) > 100:
        flat_values = noise_residual[flat_mask]
    else:
        flat_values = noise_residual.ravel()
        flat_area_ratio = 0.0

    flat_noise_std = np.std(flat_values)
    flat_noise_mean = np.mean(np.abs(flat_values))

    # Bolgesel gurultu tutarlilik kontrolu
    h, w = img.shape
    block_size = min(64, max(8, h // 4), max(8, w // 4))

    block_stds = []
    flat_block_stds = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise_residual[y:y+block_size, x:x+block_size]
            block_stds.append(np.std(block))

            block_mask = flat_mask[y:y+block_size, x:x+block_size]
            if np.mean(block_mask) > 0.25:
                flat_block = block[block_mask]
                if flat_block.size > 32:
                    flat_block_stds.append(np.std(flat_block))

    block_std_variation = np.std(block_stds) if block_stds else 0
    flat_block_std_variation = np.std(flat_block_stds) if flat_block_stds else block_std_variation
    texture_residual_gap = noise_std - flat_noise_std

    # ── Puanlama (Kalibre)
    # Birincil metrik: flat_noise_std. Orta degerler tek basina AI kaniti
    # degildir; gercek fotograflarda JPEG/noise reduction benzer degerler uretebilir.
    # AI: noise_std < 2.0 → yuksek puan (cok puruzsuz)
    # Gercek (on kamera): noise_std ≈ 3.0-5.0
    # Gercek (arka kamera): noise_std > 4 → dusuk puan
    if flat_noise_std < 1.5:
        score = 88
    elif flat_noise_std < 2.0:
        score = 72
    elif flat_noise_std < 2.3:
        score = 55
    elif flat_noise_std < 3.2:
        score = 38
    elif flat_noise_std < 4.5:
        score = 30
    elif flat_noise_std < 6.0:
        score = 25
    else:
        score = 10

    # Ikincil metrik: duz alan block tutarliligi. Sadece flat residual zaten
    # dusukse uniformluk bonusu verilir.
    # AI: block_std_variation ≈ 1.2-1.9 (her yerde ayni gurultu)
    # Gercek: block_std_variation ≈ 3.0+ (bolgesel farkliliklar)
    if flat_block_std_variation < 0.8 and flat_noise_std < 2.3:
        score = min(100, score + 12)
    elif flat_block_std_variation < 1.2 and flat_noise_std < 2.4:
        score = min(100, score + 6)
    elif flat_block_std_variation > 2.5:
        score = max(0, score - 14)
    elif flat_block_std_variation > 1.8:
        score = max(0, score - 7)

    if flat_area_ratio >= 0.08 and flat_noise_std < 2.4 and texture_residual_gap > 3.0:
        score = min(100, score + 8)

    # Gorsellestirme
    noise_visual = np.abs(noise_residual)
    max_val = np.max(noise_visual) if np.max(noise_visual) > 0 else 1
    noise_visual = ((noise_visual / max_val) * 255).astype(np.uint8)
    noise_visual = cv2.equalizeHist(noise_visual)

    stats = {
        "noise_std": round(noise_std, 2),
        "noise_mean": round(noise_mean, 2),
        "block_std_var": round(block_std_variation, 2),
        "flat_noise_std": round(flat_noise_std, 2),
        "flat_noise_mean": round(flat_noise_mean, 2),
        "flat_area_ratio": round(float(flat_area_ratio), 3),
        "flat_block_std_var": round(flat_block_std_variation, 2),
        "texture_residual_gap": round(texture_residual_gap, 2),
    }

    return noise_visual, score, stats

def detect_smart_ela(image_path, quality=90):
    """
    Yeni nesil ELA: Canny Edge ile maskelenmis hata seviyesi analizi.

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

    # Dogal kenarlari maskele
    edges = cv2.Canny(gray_original, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    flat_areas_mask = cv2.bitwise_not(edges_dilated) / 255.0

    # ELA'yi duz alan maskesi ile carp
    smart_ela = gray_diff * flat_areas_mask

    # Istatistikler
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

    # Gorsellestirme
    max_val = np.max(smart_ela) if np.max(smart_ela) > 0 else 1
    ela_visual = ((smart_ela / max_val) * 255.0).astype(np.uint8)

    stats = {
        "ela_mean": round(ela_mean, 2),
        "ela_max": round(float(ela_max), 2),
        "ela_diff": round(float(ela_diff), 2),
        "ela_std": round(ela_std, 2),
    }

    return ela_visual, score, stats
