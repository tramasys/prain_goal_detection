import cv2, numpy as np, sys, math

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "test1.mp4"

MIN_AREA       = 1500        # = ~25×25 px at QVGA – ignore noise
MIN_SOLIDITY   = 0.30        # reject thin lines, floor cracks, etc.
ROI_SIZE       = 128         # square normalised letter patch

def classify_letter(bgr):
    # 1. keep only dark blobs (letter ≈ black)
    g     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    # 2. choose most plausible glyph contour
    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:                # too small
            continue
        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-8)
        if solidity < MIN_SOLIDITY:        # too hollow / noise
            continue
        cands.append((area, c))
    if not cands:
        return None, None

    c = max(cands, key=lambda t: t[0])[1]  # biggest candidate

    # 3. perspective-correct to fixed ROI
    rect = cv2.minAreaRect(c)
    box  = cv2.boxPoints(rect).astype(np.float32)
    dst  = np.float32([[0,0],[ROI_SIZE-1,0],[ROI_SIZE-1,ROI_SIZE-1],[0,ROI_SIZE-1]])
    M    = cv2.getPerspectiveTransform(box, dst)
    roi  = cv2.warpPerspective(bgr, M, (ROI_SIZE, ROI_SIZE))

    # 4. binarise ROI & count holes
    _, letter = cv2.threshold(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                              0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts, hier = cv2.findContours(letter, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None, box

    holes = sum(1 for h in hier[0] if h[3] != -1)
    tag   = 'A' if holes == 1 else 'B' if holes >= 2 else 'C'
    return tag, box

# -------------------- main loop ---------------------------------------
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    sys.exit(f"Cannot open {VIDEO}")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    tag, circ = classify_letter(frame)
    print(tag)
    if tag:
        cv2.putText(frame, tag, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Node detection (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
