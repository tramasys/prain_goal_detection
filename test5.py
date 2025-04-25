#!/usr/bin/env python3
# Dependencies
#   sudo apt install tesseract-ocr
#   pip install pytesseract opencv-python
import cv2, numpy as np, pytesseract, sys
from pathlib import Path

WHITELIST   = "ABC"
ANGLE_STEP  = 1                                                                 # degrees between successive trials
MIN_AREA    = 500                                                               # px² discard smaller blobs
KERNEL      = np.ones((3,3), np.uint8)                                          # morphological kernel
TESS_CFG    = (f"-c tessedit_char_whitelist={WHITELIST} --oem 3 --psm 10")      # tesseract config / single char

def prep_for_ocr(bin_img: np.ndarray, target_h: int = 120) -> np.ndarray:
    """
    bin_img   : uint8, glyph=0, bg=255   (output of filter_letter / rotate)
    returns   : resized uint8 image ready for pytesseract
    """
    # 1. tight crop around the glyph
    ys, xs = np.where(bin_img < 128)
    if ys.size == 0:                          # nothing left (shouldn't happen)
        return bin_img.copy()
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    roi = bin_img[y0:y1, x0:x1]

    # 2. constant white border (so glyph is not glued to the frame)
    roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10,
                             cv2.BORDER_CONSTANT, value=255)

    # 3. scale so height ≈ target_h (keep aspect ratio)
    h, w = roi.shape
    if h != target_h:
        new_w = int(w * (target_h / h))
        roi   = cv2.resize(roi, (new_w, target_h),
                           interpolation=cv2.INTER_NEAREST)
    return roi


def filter_letter(src_bgr: np.ndarray) -> np.ndarray:
    """
    Returns 8-bit image with *letter black (0)*, background white (255).
    """
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

    # binary mask of dark regions
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, 1)

    # keep only the biggest interior blob (drop border speckles)
    h, w = mask.shape
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x+bw == w or y+bh == h
        if touches or area < MIN_AREA:
            mask[lbl == i] = 0

    return 255 - mask # black glyph, white background

def ocr_letter(img_gray: np.ndarray) -> str | None:
    ready = prep_for_ocr(img_gray)         # <-- NEW
    txt   = pytesseract.image_to_string(ready, config=TESS_CFG)
    txt   = txt.strip().upper()
    return txt[0] if txt and txt[0] in WHITELIST else None

def rotate(img: np.ndarray, angle: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def main(file: Path) -> None:
    src = cv2.imread(str(file))
    if src is None:
        sys.exit(f"cannot read {file}")

    filtered = filter_letter(src)
    letter   = ocr_letter(filtered)
    angle    = 0

    # try rotated versions until we get A/B/C or run full circle
    if letter is None:
        for angle in range(ANGLE_STEP, 360, ANGLE_STEP):
            trial = rotate(filtered, angle)
            letter = ocr_letter(trial)
            print(f"[DBG] {angle:3}° → {letter or '-'}")
            if letter:
                filtered = trial
                break

    print("Detected:", letter or "None", f"(orientation {angle}°)" if letter else "")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: detect_letter_tesseract_masked.py <image>")

    main(Path(sys.argv[1]))
