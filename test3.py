#!/usr/bin/env python3
# live_black_filter.py
#
# Show an MP4 so that only the black letters remain; everything else is white.
#    q  : quit
#    p  : pause / resume
#    s  : start / stop recording the filtered stream to filtered.mp4
#
# -------------------------------------------------------------------------
import cv2, sys, numpy as np
from pathlib import Path

SRC = sys.argv[1] if len(sys.argv) > 1 else "test4.mp4"

def classify_abc(filtered_bin: np.ndarray) -> str | None:
    """
    Detects ‘A’, ‘B’ or ‘C’ (or None) in a binary frame produced by live_black_filter.py.
    Assumes exactly one big glyph is present.
    """
    inv = cv2.bitwise_not(filtered_bin)                       # glyph white, bg black

    # single-pass contour + hierarchy extraction
    contours, hier = cv2.findContours(inv, cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None
    hier = hier[0]                                           # shape: (N, 4)

    # largest *outer* contour = the glyph
    outer_ids = [i for i, h in enumerate(hier) if h[3] == -1]
    if not outer_ids:
        return None
    glyph_id = max(outer_ids, key=lambda i: cv2.contourArea(contours[i]))

    # walk its children to count sufficiently-big holes
    MIN_HOLE_AREA = 200                                       # px; tune if needed
    hole_cnt = 0
    child = hier[glyph_id][2]                                 # 1st child
    while child != -1:
        if cv2.contourArea(contours[child]) > MIN_HOLE_AREA:
            hole_cnt += 1
        child = hier[child][0]                                # next sibling

    return {0: 'C', 1: 'A', 2: 'B'}.get(hole_cnt)



cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    sys.exit(f"Cannot open {SRC!r}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
delay = int(1000 / fps)

KERNEL = np.ones((3, 3), np.uint8)       # small speckle cleaner
record = False
writer = None
paused = False

print("LIVE FILTER  –  q quit, p pause, s save filtered.mp4")

while True:
    if not paused:
        ok, frame = cap.read()
        if not ok:
            break

        # --- keep-only interior, sufficiently-big black blobs --------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ❶ binary mask of the darkest pixels (letters + shadows + borders)
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, 1)

        # ❷ connected-component analysis
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        H, W = mask.shape
        MIN_AREA = 500          # px² – adjust to drop tiny blobs

        for i in range(1, num):                      # 0 is background
            x, y, w, h, area = stats[i]
            touches_border = (x == 0 or y == 0 or
                            x + w == W or y + h == H)
            if touches_border or area < MIN_AREA:
                mask[labels == i] = 0               # discard blob

        filtered = 255 - mask                       # glyphs black, bg white

        letter = classify_abc(filtered)
        if letter is not None:
            print("DETECTED:", letter)

        if record:
            writer.write(filtered)

        cv2.imshow("black-only view", filtered)

    k = cv2.waitKey(delay if not paused else 50) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('p'):
        paused = not paused
    elif k == ord('s'):
        record = not record
        if record:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = Path("filtered.mp4")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h),
                                     isColor=False)
            print(f" recording → {out_path}")
        else:
            writer.release()
            writer = None
            print(" recording stopped")

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
