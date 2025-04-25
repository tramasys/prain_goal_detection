import cv2, numpy as np, sys, math
from collections import deque

# ----------------------------------------------------------------------
VIDEO          = sys.argv[1] if len(sys.argv) > 1 else "test1.mp4"

FRAME_SIDE     = 320          # camera output is 320×320 px
MIN_AREA_PCT   = 0.02         # glyph must cover ≥ 2 % of frame
MIN_AREA       = int(FRAME_SIDE * FRAME_SIDE * MIN_AREA_PCT)
MIN_SOLIDITY   = 0.50         # blob must be fairly solid
MAX_AR         = 2.0          # reject blobs with extreme aspect ratio
ROI_SIZE       = 128          # square patch fed to hole counter
KERNEL         = np.ones((3, 3), np.uint8)

STABLE_FRAMES  = 4            # debounce length
# ----------------------------------------------------------------------

def classify_letter(bgr):
    """
    Return (tag, poly):
        tag  ∈ {'A','B','C'} or None
        poly = 4×2 float32 array of glyph bounding box (or None)
    """
    # 1. dark-pixel mask
    g   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g   = cv2.GaussianBlur(g, (5,5), 0)                      # de-speckle
    _, bw = cv2.threshold(g, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, KERNEL, 1)    # kill pinholes

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    # 2. candidate screen
    FRAME_PIX = bgr.shape[0] * bgr.shape[1]
    MIN_AREA  = int(FRAME_PIX * 0.04)        # >= 4 % of frame
    MAX_AREA  = int(FRAME_PIX * 0.35)        # <= 35 % of frame

    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        hull = cv2.convexHull(c)
        if area / (cv2.contourArea(hull)+1e-8) < MIN_SOLIDITY:
            continue
        (w,h) = cv2.minAreaRect(c)[1]
        if min(w,h)==0 or max(w,h)/min(w,h) > MAX_AR:
            continue
        cands.append((area, c))
    if not cands:
        return None, None

    c   = max(cands, key=lambda t: t[0])[1]                  # biggest glyph

    # 3. warp glyph to square ROI
    rect = cv2.minAreaRect(c)
    box  = cv2.boxPoints(rect).astype(np.float32)
    dst  = np.float32([[0,0],[ROI_SIZE-1,0],[ROI_SIZE-1,ROI_SIZE-1],[0,ROI_SIZE-1]])
    M    = cv2.getPerspectiveTransform(box, dst)
    roi  = cv2.warpPerspective(bgr, M, (ROI_SIZE,ROI_SIZE))

    # 4. re-threshold, close gaps, count *significant* holes
    _, lmask = cv2.threshold(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                             0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    lmask = cv2.morphologyEx(lmask, cv2.MORPH_CLOSE, KERNEL, 1)

    cnts, hier = cv2.findContours(lmask, cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None, box

    glyph_area = lmask.sum() / 255.0
    HOLE_MIN   = glyph_area * 0.02          # ignore holes < 2 % of glyph

    holes = 0
    for i,h in enumerate(hier[0]):
        if h[3] != -1:                      # child contour = hole
            if cv2.contourArea(cnts[i]) >= HOLE_MIN:
                holes += 1

    tag = 'A' if holes == 1 else 'B' if holes >= 2 else 'C'
    return tag, box.astype(int)


# -------------------- main loop ---------------------------------------
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    sys.exit(f"Cannot open {VIDEO}")

history = deque(maxlen=STABLE_FRAMES)
font = cv2.FONT_HERSHEY_SIMPLEX

fps          = cap.get(cv2.CAP_PROP_FPS) or 30      # fallback if header missing
SLOW_FACTOR  = 1.0                                 # 1 × slower than real time
delay_ms     = int(1000 / fps * SLOW_FACTOR)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    tag, poly = classify_letter(frame)
    history.append(tag)

    stable = tag if history.count(tag) == history.maxlen else None

    if poly is not None:
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
    if stable:
        cv2.putText(frame, stable, (20, 40), font,
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("A/B/C detection (press q to quit)", frame)
    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
