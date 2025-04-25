import cv2, numpy as np

# ------ constants ------
WHITE_LO = (0, 0, 200)      # HSV mask for the white disk
WHITE_HI = (180, 30, 255)
CIRC_MIN_AREA = 1000        # px², ignore noise
CIRC_MIN_CIRC = 0.80        # circularity filter

def classify_node(frame_bgr):
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_LO, WHITE_HI)

    # 1. locate the white circle
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:                       return None
    c      = max(cnts, key=cv2.contourArea)
    area   = cv2.contourArea(c); per = cv2.arcLength(c, True)
    if area < CIRC_MIN_AREA or 4*np.pi*area/(per**2+1e-8) < CIRC_MIN_CIRC:
        return None                    # not circular enough

    (cx,cy),r = cv2.minEnclosingCircle(c)
    r = int(r*0.9)                     # drop rim
    x0,y0 = int(cx-r), int(cy-r)
    roi   = frame_bgr[y0:y0+2*r, x0:x0+2*r]

    # 2. binarise letter only
    g      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _,bw   = cv2.threshold(g, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    letter = 255 - bw                 # white glyph on black

    # 3. count holes → A/B/C
    cnts,hier = cv2.findContours(letter, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:                  return None
    holes = sum(1 for h in hier[0] if h[3] != -1)   # children in hierarchy :contentReference[oaicite:0]{index=0}

    if   holes == 0: return 'C'
    elif holes == 1: return 'A'
    elif holes >= 2: return 'B'
    else:            return None      # shouldn’t occur
