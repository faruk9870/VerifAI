"""
VerifAI Test Script — data/ klasöründeki tüm fotoğrafları analiz eder.
Her fotoğraf için 8 algoritma puanı ve final güven skorunu tablo halinde gösterir.
"""
import sys
import os
import glob
import io

# Windows konsol encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# src klasörünü path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detector import ManipulationDetector
from confidence import ConfidenceEngine


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Tüm görsel dosyalarını bul
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))

    if not image_files:
        print("❌ data/ klasöründe görsel bulunamadı!")
        return

    print("=" * 120)
    print(f"{'VerifAI — Toplu Test Raporu':^120}")
    print("=" * 120)
    print()

    display_names = ConfidenceEngine.get_algo_display_names()

    # Başlık satırı
    header = f"{'Dosya':<30}"
    for name in display_names.values():
        short = name[:8]
        header += f" {short:>8}"
    header += f" {'FİNAL':>8} {'KARAR':>20}"
    print(header)
    print("-" * 120)

    for img_path in sorted(image_files):
        filename = os.path.basename(img_path)

        try:
            result = ManipulationDetector.run_full_suite(img_path, threshold=15)
            confidence = ConfidenceEngine.compute(
                result["algorithm_scores"],
                result["modifiers"]
            )

            row = f"{filename:<30}"
            for algo_name in display_names.keys():
                score = result["algorithm_scores"].get(algo_name, 0)
                row += f" {score:>8}"

            row += f" {confidence['final_score']:>8}"
            row += f" {confidence['emoji']} {confidence['verdict']:>18}"

            print(row)

            # Detaylı istatistikler
            stats = result["algorithm_stats"]
            details = []
            if "noise" in stats and stats["noise"]:
                details.append(f"noise_std={stats['noise'].get('noise_std', '?')}")
            if "dct" in stats and stats["dct"]:
                details.append(f"dct_kurt={stats['dct'].get('dct_kurtosis', '?')}")
            if "wavelet" in stats and stats["wavelet"]:
                details.append(f"hh_ratio={stats['wavelet'].get('hh_ratio', '?')}")
            if "color_stats" in stats and stats["color_stats"]:
                details.append(f"corr={stats['color_stats'].get('mean_corr', '?')}")
            if "glcm_texture" in stats and stats["glcm_texture"]:
                details.append(f"homo={stats['glcm_texture'].get('homogeneity', '?')}")

            print(f"{'':>30} └─ {', '.join(details)}")

        except Exception as e:
            print(f"{filename:<30} ❌ HATA: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 120)
    print("Skor Aralıkları:  0-25 🟢 GERÇEK  |  25-45 🔵 MUH. GERÇEK  |  45-65 🟡 BELİRSİZ  |  65-85 🟠 ŞÜPHELİ  |  85-100 🔴 AI ÜRETİMİ")
    print("=" * 120)


if __name__ == "__main__":
    main()
