"""
VerifAI — Ileri Seviye Adli Bilisim Tespit Motoru
8 bagimsiz algoritma ile AI uretimi gorsel tespiti.

Algoritmalar:
    1. EXIF / Metadata Analizi
    2. Gurultu Residual Analizi (Gelistirilmis)
    3. Akilli ELA (Kenar Korumali)
    4. FFT Frekans Analizi (Gelistirilmis)
    5. DCT Spektrum Analizi (YENI)
    6. Wavelet Alt-Bant Analizi (YENI)
    7. Renk Istatistik Analizi (YENI)
    8. GLCM Doku Analizi (YENI)
    9. Kenar Tutarlilik Analizi (YENI)
"""

from algorithms.metadata import analyze_metadata, check_social_media_wash
from algorithms.noise_ela import detect_noise_residual, detect_smart_ela
from algorithms.frequency import check_smart_fft, analyze_dct_spectrum, analyze_wavelet
from algorithms.texture import analyze_glcm_texture, analyze_lbp_texture
from algorithms.color_edge import analyze_color_statistics, analyze_edge_consistency

class ManipulationDetector:
    """8 algoritmayi iceren ana tespit sinifi."""
    
    # ─────────────────────────────────────────────────
    # ANA ANALIZ ORKESTRATORU
    # ─────────────────────────────────────────────────
    @classmethod
    def run_full_suite(cls, image_path, threshold=15):
        """
        Tum algoritmalari calistirir.

        Returns:
            dict: Tam analiz sonuclari
        """
        # 1. Metadata
        has_exif, exif_info, metadata_score = analyze_metadata(image_path)
        is_washed = check_social_media_wash(image_path, has_exif)

        # 2. Gurultu Residual
        noise_map, noise_score, noise_stats = detect_noise_residual(image_path)

        # 3. Akilli ELA
        ela_map, ela_score, ela_stats = detect_smart_ela(image_path)

        # 4. FFT
        fft_map, fft_score, is_recaptured, fft_stats = check_smart_fft(image_path, threshold)

        # 5. DCT
        dct_map, dct_score, dct_stats = analyze_dct_spectrum(image_path)

        # 6. Wavelet
        wavelet_map, wavelet_score, wavelet_stats = analyze_wavelet(image_path)

        # 7. Renk Istatistik
        color_map, color_score, color_stats = analyze_color_statistics(image_path)

        # 8. GLCM Doku
        glcm_map, glcm_score, glcm_stats = analyze_glcm_texture(image_path)

        # 9. Kenar Tutarlilik
        edge_map, edge_score, edge_stats = analyze_edge_consistency(image_path)

        # 10. LBP Mikro-Doku
        lbp_map, lbp_score, lbp_stats = analyze_lbp_texture(image_path)

        algorithm_scores = {
            "metadata": metadata_score,
            "noise": noise_score,
            "ela": ela_score,
            "fft": fft_score,
            "dct": dct_score,
            "wavelet": wavelet_score,
            "color_stats": color_score,
            "glcm_texture": glcm_score,
            "edge_consistency": edge_score,
            "lbp_texture": lbp_score,
        }

        algorithm_stats = {
            "metadata": {"exif_info": exif_info, "has_exif": has_exif},
            "noise": noise_stats,
            "ela": ela_stats,
            "fft": fft_stats,
            "dct": dct_stats,
            "wavelet": wavelet_stats,
            "color_stats": color_stats,
            "glcm_texture": glcm_stats,
            "edge_consistency": edge_stats,
            "lbp_texture": lbp_stats,
        }

        maps = {
            "Gurultu Residual": noise_map,
            "Akilli ELA": ela_map,
            "FFT Frekans": fft_map,
            "DCT Spektrum": dct_map,
            "Wavelet HH Bandi": wavelet_map,
            "Renk Doygunluk": color_map,
            "GLCM Doku": glcm_map,
            "Kenar Haritasi": edge_map,
            "LBP Mikro-Doku": lbp_map,
        }

        # None map'leri temizle
        maps = {k: v for k, v in maps.items() if v is not None}

        modifiers = {
            "has_exif": has_exif,
            "metadata_score": metadata_score,
            "is_social_washed": is_washed,
            "is_recaptured": is_recaptured,
        }

        return {
            "algorithm_scores": algorithm_scores,
            "algorithm_stats": algorithm_stats,
            "maps": maps,
            "modifiers": modifiers,
            "exif_info": exif_info,
        }
