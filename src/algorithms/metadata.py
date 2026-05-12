import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


def detect_ai_visual_watermark(image_path):
    """
    Sag-alt bolgede dusuk doygunluklu, kompakt yapay uretim filigrani arar.
    Bu, WhatsApp/JPEG sikistirmasi EXIF'i sildiginde kalan guclu provenance sinyalidir.
    """
    try:
        buf = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return False

        h, w = img.shape[:2]
        crop = img[int(h * 0.72):h, int(w * 0.82):w]
        if crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        for sat_max in (30, 45):
            for val_min in (145, 160):
                mask = (
                    (hsv[:, :, 1] < sat_max) &
                    (hsv[:, :, 2] > val_min) &
                    (hsv[:, :, 2] < 245)
                ).astype(np.uint8) * 255
                _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

                for i in range(1, len(stats)):
                    x, y, cw, ch, area = stats[i]
                    # Uretim araclarinin ekledigi filigran kucuk, kompakt ve
                    # sag-alt kosede belirli bir ofsete sahiptir. Genis olcekli
                    # araliklar gercek fotograflardaki vida, tas, tuy ve parlak
                    # nesneleri filigran sanabildigi icin burada bilincli olarak
                    # dar geometri kullaniyoruz.
                    if not (34 <= cw <= 62 and 34 <= ch <= 62):
                        continue

                    aspect = cw / max(ch, 1)
                    fill = area / max(cw * ch, 1)
                    cx, cy = centroids[i]
                    rel_x = cx / max(crop.shape[1], 1)
                    rel_y = cy / max(crop.shape[0], 1)
                    expected_position = 0.62 <= rel_x <= 0.88 and 0.52 <= rel_y <= 0.84

                    if expected_position and 0.92 <= aspect <= 1.12 and 0.25 <= fill <= 0.43:
                        return True
    except Exception:
        return False

    return False


def analyze_metadata(image_path):
    """
    EXIF metadata analizi.
    Kamera donanim izleri (Make, Model, Software) arar.

    Returns:
        tuple: (has_exif: bool, exif_info: str, ai_score: int)
    """
    try:
        if detect_ai_visual_watermark(image_path):
            return False, "AI FILIGRAN IZI: Sag-alt gorsel uretim filigrani tespit edildi.", 98

        image = Image.open(image_path)
        exifdata = image.getexif()
        if not exifdata:
            return False, "EXIF YOK: Veriler silinmis veya hic uretilmemis.", 80

        metadata = {}
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            if isinstance(data, bytes):
                try:
                    data = data.decode(errors="ignore")
                except Exception:
                    pass
            metadata[tag] = data

        hardware_traces = ["Make", "Model", "Software"]
        found_traces = {t: metadata.get(t) for t in hardware_traces if t in metadata}

        # AI yazilim damgalarini ara
        ai_software_keywords = [
            "stable diffusion", "midjourney", "dall-e", "comfyui",
            "automatic1111", "novelai", "nai diffusion",
        ]
        software_val = str(metadata.get("Software", "")).lower()
        for kw in ai_software_keywords:
            if kw in software_val:
                return True, f"AI YAZILIM IZI: {metadata.get('Software', '')}", 95

        if len(found_traces) > 0:
            make = found_traces.get("Make", "")
            model = found_traces.get("Model", "")
            return True, f"KAMERA IZI: {make} {model}".strip(), 5
        return True, "EXIF var ama donanim izi yok.", 55
    except Exception:
        return False, "EXIF Okuma Hatasi", 60


def check_social_media_wash(image_path, has_exif):
    """
    EXIF'i silinmis ve yeniden sikistirilmis JPEG tespiti.
    Mesajlasma/sosyal medya ciktilari DCT, wavelet, edge ve LBP gibi
    frekans/doku sinyallerini sistematik olarak yukseltebilir.
    """
    if has_exif:
        return False

    try:
        image = Image.open(image_path)
        if image.format != "JPEG":
            return False

        quantization = getattr(image, "quantization", None) or {}
        if not quantization:
            return False

        table_sums = [sum(table) for table in quantization.values()]
        luma_sum = table_sums[0] if len(table_sums) > 0 else 0
        chroma_sum = table_sums[1] if len(table_sums) > 1 else 0
        is_progressive = bool(image.info.get("progressive") or image.info.get("progression"))

        return is_progressive and (luma_sum >= 1600 or chroma_sum >= 2400)
    except Exception:
        return False
