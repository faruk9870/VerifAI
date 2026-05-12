"""
VerifAI — Agirlikli Ensemble Guven Skoru Motoru
Her algoritmadan gelen 0-100 puani agirliklari ile birlestirerek tek bir "Guven Skoru" uretir.
"""


class ConfidenceEngine:
    """
    Agirlikli ensemble guven skoru hesaplayici.
    Her algoritma 0-100 arasi bir AI olasilik puani uretir.
    Bu puanlar agirliklarla carpilip toplam guven skoru elde edilir.

    Skor Araliklari:
        0-25   → 🟢 GERCEK
        25-45  → 🔵 MUHTEMELEN GERCEK
        45-65  → 🟡 BELIRSIZ
        65-85  → 🟠 SUPHELI
        85-100 → 🔴 AI URETIMI
    """

    # Kalibrasyon sonuclarina gore yeni algoritmalarla guncellenmis agirliklar:
    #   - noise (Bilateral Filter ile daha guvenilir)
    #   - color_stats (YCbCr krominans varyansi eklendi)
    #   - wavelet (db2 ile dogal goruntu modellemesi)
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

    # Skor → Karar esleme tablosu
    VERDICT_TABLE = [
        (25,  "GERCEK",             "Yuksek guvenle gercek fotograf.",                       "#2ecc71", "🟢"),
        (45,  "MUHTEMELEN GERCEK",  "Algoritmalar dusuk risk tespit etti.",                   "#3498db", "🔵"),
        (65,  "BELIRSIZ",          "Karisik sinyaller — bazi algoritmalar supheli buldu.",    "#f1c40f", "🟡"),
        (85,  "SUPHELI",          "Birden fazla algoritma AI izleri tespit etti.",           "#e67e22", "🟠"),
        (101, "AI URETIMI",        "Cok guclu yapay uretim sinyalleri.",                     "#e74c3c", "🔴"),
    ]

    @classmethod
    def compute(cls, algorithm_scores: dict, modifiers: dict = None) -> dict:
        """
        Ana guven skoru hesaplama.

        Args:
            algorithm_scores: dict — Her algoritmanin adi ve 0-100 puani
                Ornek: {"noise": 85, "ela": 40, "dct": 92, ...}
            modifiers: dict — Ek duzeltme faktorleri
                - has_exif (bool): EXIF varsa skor dusurulur
                - is_social_washed (bool): Sosyal medya sikistirmasi varsa
                    gurultu bazli algoritmalar dikkate alinmaz
                - is_recaptured (bool): Ekrandan cekim tespiti

        Returns:
            dict: Detayli sonuc
        """
        if modifiers is None:
            modifiers = {}

        has_exif = modifiers.get("has_exif", False)
        metadata_score = modifiers.get("metadata_score", 50)
        is_social_washed = modifiers.get("is_social_washed", False)
        is_recaptured = modifiers.get("is_recaptured", False)
        ai_watermark_detected = modifiers.get("ai_watermark_detected", False)

        # Yeniden sikistirma sinyali karar asamasinda guard olarak kullanilir.
        # Algoritmalari bastan kapatmak yerine baglam icinde agirliklarini dusuruyoruz.
        suppressed = set()

        # Agirlikli toplam
        weighted_sum = 0.0
        total_weight = 0.0
        per_algo_details = []

        for algo_name, weight in cls.WEIGHTS.items():
            score = algorithm_scores.get(algo_name, 50)  # varsayilan: belirsiz

            if algo_name in suppressed:
                per_algo_details.append({
                    "name": algo_name,
                    "score": score,
                    "weight": weight,
                    "active": False,
                    "reason": "Sosyal medya sikistirmasi nedeniyle devre disi"
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

        # ── AKILLI GERCEKLIK KALKANI ──
        # Problem: On kameralar beauty mode, JPEG sikistirma ve noise reduction
        # uyguladigindan puruzsuzluk algoritmalari (wavelet, glcm, edge) yanlis alarm verir.
        # Cozum: Gercekligi kesinlestiren algoritmalar (noise, ela, fft, dct) dusuk cikiyorsa,
        # bu fotografin gercek oldugunu kabul et ve puruzsuzluk algoritmalarini baskila.
        
        # Gercekligi kesinlestiren algoritmalari say
        # Bu algoritmalar puruzsuzlukten BAGIMSIZDIR ve AI'dan kesin olarak ayrisir
        reality_signals = ["noise", "ela", "fft", "dct"]
        reality_count = sum(1 for name in reality_signals 
                          if algorithm_scores.get(name, 50) <= 30)
        
        is_real_camera = has_exif and metadata_score < 20
        lbp_score = algorithm_scores.get("lbp_texture", 50)
        dct_score = algorithm_scores.get("dct", 50)
        noise_score = algorithm_scores.get("noise", 50)
        ela_score = algorithm_scores.get("ela", 50)
        fft_score = algorithm_scores.get("fft", 50)
        forensic_real_anchor = ela_score <= 20 and fft_score <= 20
        compression_artifact_guard = (
            is_social_washed and
            not ai_watermark_detected and
            forensic_real_anchor and
            dct_score <= 85
        )

        # Texture algoritmalari (noise/wavelet/glcm/edge/lbp) tekrarlayan gercek
        # dokularda yukselebilir: kumas, karbon fiber, sac/tuy, yaprak, devre karti,
        # baskili yazi vb. Bu guard sadece ELA/FFT ve DCT gibi adli ankrajlar
        # guclu AI demiyorsa calisir.
        texture_false_positive_guard = (
            forensic_real_anchor and
            (
                compression_artifact_guard or
                (
                    dct_score <= 55 and
                    (noise_score <= 45 or lbp_score <= 75 or dct_score <= 35)
                )
            )
        )

        # Genis, pürüzsüz ve dusuk detayli gercek fotograflarda noise/wavelet/color
        # yanlis yukselebilir. LBP ve DCT ayni anda gercek diyorsa noise da baskilanabilir.
        smooth_capture_guard = (
            forensic_real_anchor and
            dct_score <= 35 and
            lbp_score <= 35 and
            noise_score >= 65
        )
        
        # AI VETO: Eger LBP/DCT birlikte guclu AI gosteriyorsa veya birden fazla
        # bagimsiz guclu AI sinyali varsa kalkan ASLA devreye girmemeli.
        ai_veto_algos = ["dct", "color_stats", "lbp_texture", "noise"]
        strong_ai_signals = sum(1 for name in ai_veto_algos 
                               if algorithm_scores.get(name, 50) >= 65)
        ai_veto = (
            ai_watermark_detected or
            (lbp_score >= 90 and dct_score >= 60 and not compression_artifact_guard) or
            (strong_ai_signals >= 2 and not texture_false_positive_guard and not smooth_capture_guard)
        )
        
        # Kalkan aktiflesme kosullari:
        # 1. EXIF gercek donanim izi gosteriyor (AI veto yok ise)
        # 2. 3+ bagimsiz algoritma "gercek" diyor VE AI veto yok
        # 3. LBP dusuk VE en az 2 gerceklik sinyali var
        beauty_shield_active = not ai_veto and (
            is_real_camera or 
            reality_count >= 3 or
            texture_false_positive_guard or
            compression_artifact_guard or
            smooth_capture_guard or
            (lbp_score <= 40 and reality_count >= 2)
        )
        
        if beauty_shield_active:
            # Puruzsuzlukten etkilenen algoritmalari baskila
            shield_targets = {"wavelet", "glcm_texture", "edge_consistency", "color_stats"}
            if texture_false_positive_guard:
                shield_targets.add("lbp_texture")
            if compression_artifact_guard:
                shield_targets.add("dct")
            if smooth_capture_guard:
                shield_targets.add("noise")
            for d in per_algo_details:
                if d["name"] in shield_targets and d["score"] > 30:
                    d["score"] = d["score"] * 0.3  # %70 baskila
                    shield_count = max(
                        reality_count,
                        2 if texture_false_positive_guard or compression_artifact_guard or smooth_capture_guard else 0
                    )
                    d["reason"] = "Gerceklik Kalkani aktif (" + str(shield_count) + " sinyal)"
            
            # Weighted sum'i yeni skorlarla tekrar hesapla
            weighted_sum = sum(d["score"] * d["weight"] for d in per_algo_details if d["active"])

        # Normalize (devre disi algoritmalar cikarildiysa)
        if total_weight > 0:
            raw_score = weighted_sum / total_weight
        else:
            raw_score = 50.0

        # ── KONSENSUS BONUSU ──
        # Bireysel puanlar orta olsa bile, cogunluk "AI" diyorsa skoru yukselt
        # Cogunluk "gercek" diyorsa skoru dusur
        # ONEMLI: Gercek kamera dogrulanmissa veya kalkan aktifse konsensus bonuslari UYGULANMAZ
        consensus_scores = [
            a["score"] for a in per_algo_details
            if a["active"] and a["name"] != "metadata"
        ]
        forensic_combo_bonus = 0
        if consensus_scores and not beauty_shield_active:
            high_count = sum(1 for s in consensus_scores if s >= 60)
            low_count = sum(1 for s in consensus_scores if s <= 25)

            # 5+ algoritma "AI" diyorsa → guclu konsensus bonusu
            if high_count >= 5:
                raw_score += 20
            elif high_count >= 4:
                raw_score += 12
            elif high_count >= 3:
                raw_score += 6

            # Kesin AI sinyalleri yakalandiysa ekstra bonus
            strong_ai_count = sum(1 for s in consensus_scores if s >= 80)
            if strong_ai_count >= 3:
                raw_score += 18
            elif strong_ai_count >= 2:
                raw_score += 10
            elif strong_ai_count >= 1:
                raw_score += 5

            # Guclu "Gercek" konsensusu
            if high_count < 2 and strong_ai_count == 0:
                if low_count >= 5:
                    raw_score -= 18
                elif low_count >= 4:
                    raw_score -= 10
                elif low_count >= 3:
                    raw_score -= 5

            missing_camera_trace = (not has_exif) or metadata_score >= 55
            if lbp_score >= 90 and dct_score >= 65:
                forensic_combo_bonus = 14 if missing_camera_trace else 9
            elif lbp_score >= 80 and dct_score >= 60:
                forensic_combo_bonus = 8 if missing_camera_trace else 5
            if ai_watermark_detected:
                forensic_combo_bonus = max(forensic_combo_bonus, 35)
            raw_score += forensic_combo_bonus

        # EXIF bonusu: Gercek kamera verisi varsa skoru ciddi dusur
        exif_adjustment = 0
        if is_real_camera and raw_score >= 20:
            exif_adjustment = -35
            raw_score = max(0, raw_score + exif_adjustment)

        # Ekran cekimi ayri sinif
        if is_recaptured:
            return {
                "final_score": round(raw_score, 1),
                "verdict": "EKRANDAN CEKIM",
                "description": "FFT analizi ekran (Moiré) izgarasi tespit etti. "
                               "Bu gorsel bir ekrandan fotograflanmis olabilir.",
                "color": "#8e44ad",
                "emoji": "🟣",
                "exif_adjustment": exif_adjustment,
                "forensic_combo_bonus": forensic_combo_bonus,
                "ai_watermark_detected": ai_watermark_detected,
                "texture_false_positive_guard": texture_false_positive_guard,
                "compression_artifact_guard": compression_artifact_guard,
                "smooth_capture_guard": smooth_capture_guard,
                "per_algorithm": per_algo_details,
            }

        # ── SKOR ESNETME (S-CURVE / POLARIZATION) ──
        # 50'nin ustunu 100'e, 50'nin altini 0'a yaklastirir. (Ornek: 70 -> ~78.4)
        raw_score = max(0.0, min(100.0, raw_score))
        x = raw_score / 100.0
        stretched_score = (3 * (x ** 2) - 2 * (x ** 3)) * 100.0
        
        # Final skoru sinirla
        final_score = max(0, min(100, round(stretched_score, 1)))

        # Karar tablosundan uygun karari bul
        verdict = "BELIRSIZ"
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
            "forensic_combo_bonus": forensic_combo_bonus,
            "ai_watermark_detected": ai_watermark_detected,
            "texture_false_positive_guard": texture_false_positive_guard,
            "compression_artifact_guard": compression_artifact_guard,
            "smooth_capture_guard": smooth_capture_guard,
            "per_algorithm": per_algo_details,
        }

    @classmethod
    def get_algo_display_names(cls) -> dict:
        """UI'da gosterilecek anlasilir algoritma isimleri."""
        return {
            "metadata":         "EXIF / Metadata",
            "noise":            "Gurultu Residual",
            "ela":              "Akilli ELA",
            "fft":              "FFT Frekans",
            "dct":              "DCT Spektrum",
            "wavelet":          "Wavelet Alt-Bant",
            "color_stats":      "Renk Istatistik",
            "glcm_texture":     "GLCM Doku",
            "edge_consistency": "Kenar Tutarlilik",
            "lbp_texture":      "LBP Mikro-Doku",
        }

    @classmethod
    def get_algo_descriptions(cls) -> dict:
        """Her algoritmanin kisa aciklamasi."""
        return {
            "metadata":         "EXIF verilerinde kamera/donanim izi arar. Yoksa suphe puani artar.",
            "noise":            "Duz alan sensor gurultusunu analiz eder; sahne dokusunu gurultuden ayirmaya calisir.",
            "ela":              "JPEG sikistirma sonrasi hata seviyesini analiz eder. Montaj ve yapay alanlari yakalar.",
            "fft":              "Frekans spektrumunda periyodik paternler ve Moiré izleri arar.",
            "dct":              "DCT katsayi dagilimini analiz eder. AI'in dogal olmayan frekans profili tespit edilir.",
            "wavelet":          "Coklu cozunurlukte frekans analizi; asiri puruzsuzluk ve sentetik yogun detayi yakalar.",
            "color_stats":      "Renk kanallari arasi korelasyon ve histogram entropisi analiz eder.",
            "glcm_texture":     "Mikro-doku tutarsizliklarini GLCM matrisi ile tespit eder.",
            "edge_consistency": "Kenar keskinligi ve tutarliligini analiz eder. AI kenarlari genellikle cok duzgundur.",
            "lbp_texture":      "Local Binary Pattern (LBP) histogrami ile piksel seviyesindeki dogal olmayan tekduzeligi tespit eder.",
        }
