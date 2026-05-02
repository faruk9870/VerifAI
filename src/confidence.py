"""
VerifAI — Ağırlıklı Ensemble Güven Skoru Motoru
Her algoritmadan gelen 0-100 puanı ağırlıkları ile birleştirerek tek bir "Güven Skoru" üretir.
"""


class ConfidenceEngine:
    """
    Ağırlıklı ensemble güven skoru hesaplayıcı.
    Her algoritma 0-100 arası bir AI olasılık puanı üretir.
    Bu puanlar ağırlıklarla çarpılıp toplam güven skoru elde edilir.

    Skor Aralıkları:
        0-25   → 🟢 GERÇEK
        25-45  → 🔵 MUHTEMELEN GERÇEK
        45-65  → 🟡 BELİRSİZ
        65-85  → 🟠 ŞÜPHELİ
        85-100 → 🔴 AI ÜRETİMİ
    """

    # Kalibrasyon sonuçlarına göre yeni algoritmalarla güncellenmiş ağırlıklar:
    #   - noise (Bilateral Filter ile daha güvenilir)
    #   - color_stats (YCbCr krominans varyansı eklendi)
    #   - wavelet (db2 ile doğal görüntü modellemesi)
    WEIGHTS = {
        "metadata":         0.03,
        "noise":            0.18,
        "ela":              0.06,
        "fft":              0.07,
        "dct":              0.13,
        "wavelet":          0.10,
        "color_stats":      0.09,
        "glcm_texture":     0.10,
        "edge_consistency": 0.14,
        "lbp_texture":      0.10,
    }

    # Skor → Karar eşleme tablosu
    VERDICT_TABLE = [
        (25,  "GERÇEK",             "Yüksek güvenle gerçek fotoğraf.",                       "#2ecc71", "🟢"),
        (45,  "MUHTEMELEN GERÇEK",  "Algoritmalar düşük risk tespit etti.",                   "#3498db", "🔵"),
        (65,  "BELİRSİZ",          "Karışık sinyaller — bazı algoritmalar şüpheli buldu.",    "#f1c40f", "🟡"),
        (85,  "ŞÜPHELİ",          "Birden fazla algoritma AI izleri tespit etti.",           "#e67e22", "🟠"),
        (101, "AI ÜRETİMİ",        "Çok güçlü yapay üretim sinyalleri.",                     "#e74c3c", "🔴"),
    ]

    @classmethod
    def compute(cls, algorithm_scores: dict, modifiers: dict = None) -> dict:
        """
        Ana güven skoru hesaplama.

        Args:
            algorithm_scores: dict — Her algoritmanın adı ve 0-100 puanı
                Örnek: {"noise": 85, "ela": 40, "dct": 92, ...}
            modifiers: dict — Ek düzeltme faktörleri
                - has_exif (bool): EXIF varsa skor düşürülür
                - is_social_washed (bool): Sosyal medya sıkıştırması varsa
                    gürültü bazlı algoritmalar dikkate alınmaz
                - is_recaptured (bool): Ekrandan çekim tespiti

        Returns:
            dict: Detaylı sonuç
        """
        if modifiers is None:
            modifiers = {}

        has_exif = modifiers.get("has_exif", False)
        metadata_score = modifiers.get("metadata_score", 50)
        is_social_washed = modifiers.get("is_social_washed", False)
        is_recaptured = modifiers.get("is_recaptured", False)

        # Filtrelenecek algoritmalar (sosyal medya sıkıştırmasında)
        suppressed = set()
        if is_social_washed:
            suppressed = {"noise", "ela"}

        # Ağırlıklı toplam
        weighted_sum = 0.0
        total_weight = 0.0
        per_algo_details = []

        for algo_name, weight in cls.WEIGHTS.items():
            score = algorithm_scores.get(algo_name, 50)  # varsayılan: belirsiz

            if algo_name in suppressed:
                per_algo_details.append({
                    "name": algo_name,
                    "score": score,
                    "weight": weight,
                    "active": False,
                    "reason": "Sosyal medya sıkıştırması nedeniyle devre dışı"
                })
                continue

            weighted_sum += score * weight
            total_weight += weight
            per_algo_details.append({
                "name": algo_name,
                "score": score,
                "weight": weight,
                "active": True,
                "reason": ""
            })

        # ── AKILLI GERÇEKLİK KALKANI ──
        # Problem: Ön kameralar beauty mode, JPEG sıkıştırma ve noise reduction
        # uyguladığından pürüzsüzlük algoritmaları (wavelet, glcm, edge) yanlış alarm verir.
        # Çözüm: Gerçekliği kesinleştiren algoritmalar (noise, ela, fft, dct) düşük çıkıyorsa,
        # bu fotoğrafın gerçek olduğunu kabul et ve pürüzsüzlük algoritmalarını baskıla.
        
        # Gerçekliği kesinleştiren algoritmaları say
        # Bu algoritmalar pürüzsüzlükten BAĞIMSIZDIR ve AI'dan kesin olarak ayrışır
        reality_signals = ["noise", "ela", "fft", "dct"]
        reality_count = sum(1 for name in reality_signals 
                          if algorithm_scores.get(name, 50) <= 30)
        
        is_real_camera = has_exif and metadata_score < 20
        lbp_score = algorithm_scores.get("lbp_texture", 50)
        
        # AI VETO: Eğer LBP kesin AI gösteriyorsa veya güçlü bir AI sinyali varsa
        # kalkan ASLA devreye girmemeli. Bu, AI fotoğraflarının yanlışlıkla korunmasını engeller.
        ai_veto_algos = ["dct", "color_stats", "lbp_texture", "noise"]
        strong_ai_signals = sum(1 for name in ai_veto_algos 
                               if algorithm_scores.get(name, 50) >= 65)
        ai_veto = lbp_score >= 60 or strong_ai_signals >= 2
        
        # Kalkan aktifleşme koşulları:
        # 1. EXIF gerçek donanım izi gösteriyor (AI veto yok ise)
        # 2. 3+ bağımsız algoritma "gerçek" diyor VE AI veto yok
        # 3. LBP düşük VE en az 2 gerçeklik sinyali var
        beauty_shield_active = not ai_veto and (
            is_real_camera or 
            reality_count >= 3 or
            (lbp_score <= 40 and reality_count >= 2)
        )
        
        if beauty_shield_active:
            # Pürüzsüzlükten etkilenen algoritmaları baskıla
            shield_targets = {"wavelet", "glcm_texture", "edge_consistency", "color_stats"}
            for d in per_algo_details:
                if d["name"] in shield_targets and d["score"] > 30:
                    d["score"] = d["score"] * 0.3  # %70 baskıla
                    d["reason"] = "Gerçeklik Kalkanı aktif (" + str(reality_count) + " sinyal)"
            
            # Weighted sum'ı yeni skorlarla tekrar hesapla
            weighted_sum = sum(d["score"] * d["weight"] for d in per_algo_details if d["active"])

        # Normalize (devre dışı algoritmalar çıkarıldıysa)
        if total_weight > 0:
            raw_score = weighted_sum / total_weight
        else:
            raw_score = 50.0

        # ── KONSENSÜS BONUSU ──
        # Bireysel puanlar orta olsa bile, çoğunluk "AI" diyorsa skoru yükselt
        # Çoğunluk "gerçek" diyorsa skoru düşür
        # ÖNEMLİ: Gerçek kamera doğrulanmışsa veya kalkan aktifse konsensüs bonusları UYGULANMAZ
        active_scores = [a["score"] for a in per_algo_details if a["active"]]
        if active_scores and not beauty_shield_active:
            high_count = sum(1 for s in active_scores if s >= 60)
            low_count = sum(1 for s in active_scores if s <= 25)
            total_active = len(active_scores)

            # 5+ algoritma "AI" diyorsa → güçlü konsensüs bonusu
            if high_count >= 5:
                raw_score += 20
            elif high_count >= 4:
                raw_score += 12
            elif high_count >= 3:
                raw_score += 6

            # Kesin AI sinyalleri yakalandıysa ekstra bonus
            strong_ai_count = sum(1 for s in active_scores if s >= 80)
            if strong_ai_count >= 3:
                raw_score += 18
            elif strong_ai_count >= 2:
                raw_score += 10
            elif strong_ai_count >= 1:
                raw_score += 5

            # Güçlü "Gerçek" konsensüsü
            if high_count < 2 and strong_ai_count == 0:
                if low_count >= 5:
                    raw_score -= 18
                elif low_count >= 4:
                    raw_score -= 10
                elif low_count >= 3:
                    raw_score -= 5

        # EXIF bonusu: Gerçek kamera verisi varsa skoru ciddi düşür
        exif_adjustment = 0
        if is_real_camera and raw_score >= 20:
            exif_adjustment = -35
            raw_score = max(0, raw_score + exif_adjustment)

        # Ekran çekimi ayrı sınıf
        if is_recaptured:
            return {
                "final_score": round(raw_score, 1),
                "verdict": "EKRANDAN ÇEKİM",
                "description": "FFT analizi ekran (Moiré) ızgarası tespit etti. "
                               "Bu görsel bir ekrandan fotoğraflanmış olabilir.",
                "color": "#8e44ad",
                "emoji": "🟣",
                "exif_adjustment": exif_adjustment,
                "per_algorithm": per_algo_details,
            }

        # Final skoru sınırla
        final_score = max(0, min(100, round(raw_score, 1)))

        # Karar tablosundan uygun kararı bul
        verdict = "BELİRSİZ"
        description = ""
        color = "#f1c40f"
        emoji = "🟡"
        for threshold, v, d, c, e in cls.VERDICT_TABLE:
            if final_score < threshold:
                verdict = v
                description = d
                color = c
                emoji = e
                break

        return {
            "final_score": final_score,
            "verdict": verdict,
            "description": description,
            "color": color,
            "emoji": emoji,
            "exif_adjustment": exif_adjustment,
            "per_algorithm": per_algo_details,
        }

    @classmethod
    def get_algo_display_names(cls) -> dict:
        """UI'da gösterilecek anlaşılır algoritma isimleri."""
        return {
            "metadata":         "EXIF / Metadata",
            "noise":            "Gürültü Residual",
            "ela":              "Akıllı ELA",
            "fft":              "FFT Frekans",
            "dct":              "DCT Spektrum",
            "wavelet":          "Wavelet Alt-Bant",
            "color_stats":      "Renk İstatistik",
            "glcm_texture":     "GLCM Doku",
            "edge_consistency": "Kenar Tutarlılık",
            "lbp_texture":      "LBP Mikro-Doku",
        }

    @classmethod
    def get_algo_descriptions(cls) -> dict:
        """Her algoritmanın kısa açıklaması."""
        return {
            "metadata":         "EXIF verilerinde kamera/donanım izi arar. Yoksa şüphe puanı artar.",
            "noise":            "Sensör gürültü kalıntılarını analiz eder. AI görsellerde gürültü çok pürüzsüz olur.",
            "ela":              "JPEG sıkıştırma sonrası hata seviyesini analiz eder. Montaj ve yapay alanları yakalar.",
            "fft":              "Frekans spektrumunda periyodik paternler ve Moiré izleri arar.",
            "dct":              "DCT katsayı dağılımını analiz eder. AI'ın doğal olmayan frekans profili tespit edilir.",
            "wavelet":          "Çoklu çözünürlükte frekans analizi. HH bandındaki enerji eksikliği AI işaretidir.",
            "color_stats":      "Renk kanalları arası korelasyon ve histogram entropisi analiz eder.",
            "glcm_texture":     "Mikro-doku tutarsızlıklarını GLCM matrisi ile tespit eder.",
            "edge_consistency": "Kenar keskinliği ve tutarlılığını analiz eder. AI kenarları genellikle çok düzgündür.",
            "lbp_texture":      "Local Binary Pattern (LBP) histogramı ile piksel seviyesindeki doğal olmayan tekdüzeliği tespit eder.",
        }
