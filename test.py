import cv2, numpy as np, sys, math

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "test1.mp4"

WHITE_LO        = (0, 0, 200)     # HSV lower/upper for the white disk
WHITE_HI        = (180, 30, 255)
CIRC_MIN_AREA   = 1000            # px²
CIRC_MIN_CIRC   = 0.80            # circularity threshold

def classify_node(bgr):
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, WHITE_LO, WHITE_HI)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    c     = max(cnts, key=cv2.contourArea)
    area  = cv2.contourArea(c)
    peri  = cv2.arcLength(c, True)
    circ  = 4 * math.pi * area / (peri * peri + 1e-8)

    if area < CIRC_MIN_AREA or circ < CIRC_MIN_CIRC:
        return None, None

    (cx, cy), r = cv2.minEnclosingCircle(c)
    r_i  = int(r * 0.9)
    x0   = max(int(cx - r_i), 0)
    y0   = max(int(cy - r_i), 0)
    x1   = min(x0 + 2 * r_i, bgr.shape[1])
    y1   = min(y0 + 2 * r_i, bgr.shape[0])

    roi  = bgr[y0:y1, x0:x1]
    g    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    glyph = 255 - bw                             # letter is white, background black

    cnts, hier = cv2.findContours(glyph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None, None
    holes = sum(1 for h in hier[0] if h[3] != -1)

    letter = 'A' if holes == 1 else 'B' if holes >= 2 else 'C' if holes == 0 else None
    return letter, (int(cx), int(cy), int(r))

# -------------------- main loop ---------------------------------------
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    sys.exit(f"Cannot open {VIDEO}")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    tag, circ = classify_node(frame)
    if circ:
        cx, cy, r = circ
        cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2, cv2.LINE_AA)
    if tag:
        cv2.putText(frame, tag, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Node detection (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
