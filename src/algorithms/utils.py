import cv2
import numpy as np

def _imread_safe(path, flags=cv2.IMREAD_COLOR):
    """
    Turkce karakter iceren dosya yollarini da okuyabilen imread fonksiyonu.
    Windows'ta cv2.imread() UTF-8 yollari desteklemez.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    return img
