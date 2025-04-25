#!/usr/bin/env python3
# detect_abc.py
#
# Usage:
#   python3 detect_abc.py  test1.mp4
#
# Detects letters A / B / C (black on light background) in a 320×320 video
# without using any ML model.  Works by finding the largest dark blob,
# normalising it, and counting interior holes.

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
    # 1. keep dark pixels
    g   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, KERNEL, iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    # 2. screen candidates
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-8
        if area / hull_area < MIN_SOLIDITY:
            continue

        (w, h) = cv2.minAreaRect(c)[1]
        if min(w, h) == 0 or max(w, h) / min(w, h) > MAX_AR:
            continue

        candidates.append((area, c))

    if not candidates:
        return None, None

    c = max(candidates, key=lambda t: t[0])[1]   # largest surviving blob

    # 3. perspective-correct to fixed ROI
    rect = cv2.minAreaRect(c)
    box  = cv2.boxPoints(rect).astype(np.float32)
    dst  = np.float32([[0, 0],
                       [ROI_SIZE - 1, 0],
                       [ROI_SIZE - 1, ROI_SIZE - 1],
                       [0, ROI_SIZE - 1]])
    M    = cv2.getPerspectiveTransform(box, dst)
    roi  = cv2.warpPerspective(bgr, M, (ROI_SIZE, ROI_SIZE))

    # 4. re-threshold & count holes
    _, letter = cv2.threshold(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                              0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, hier = cv2.findContours(letter, cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None, box

    holes = sum(1 for h in hier[0] if h[3] != -1)
    tag   = 'A' if holes == 1 else 'B' if holes >= 2 else 'C'
    return tag, box.astype(int)


# -------------------- main loop ---------------------------------------
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    sys.exit(f"Cannot open {VIDEO}")

history = deque(maxlen=STABLE_FRAMES)
font = cv2.FONT_HERSHEY_SIMPLEX

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
