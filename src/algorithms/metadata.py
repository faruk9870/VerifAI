from PIL import Image
from PIL.ExifTags import TAGS

def analyze_metadata(image_path):
    """
    EXIF metadata analizi.
    Kamera donanim izleri (Make, Model, Software) arar.

    Returns:
        tuple: (has_exif: bool, exif_info: str, ai_score: int)
    """
    try:
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
                    data = data.decode(errors='ignore')
                except Exception:
                    pass
            metadata[tag] = data

        hardware_traces = ['Make', 'Model', 'Software']
        found_traces = {t: metadata.get(t) for t in hardware_traces if t in metadata}

        # AI yazilim damgalarini ara
        ai_software_keywords = ['stable diffusion', 'midjourney', 'dall-e', 'comfyui',
                                'automatic1111', 'novelai', 'nai diffusion']
        software_val = str(metadata.get('Software', '')).lower()
        for kw in ai_software_keywords:
            if kw in software_val:
                return True, f"AI YAZILIM IZI: {metadata.get('Software', '')}", 95

        if len(found_traces) > 0:
            make = found_traces.get('Make', '')
            model = found_traces.get('Model', '')
            return True, f"KAMERA IZI: {make} {model}".strip(), 5
        return True, "EXIF var ama donanim izi yok.", 55
    except Exception:
        return False, "EXIF Okuma Hatasi", 60

def check_social_media_wash(image_path, has_exif):
    """
    Sosyal medya sikistirmasi tespiti — DEVRE DISI.
    Gurultu ve ELA algoritmalarini susturdugu icin devre disi birakildi.
    Bu algoritmalar en guclu ayristiricilar oldugundan, her zaman aktif olmalilar.
    """
    return False