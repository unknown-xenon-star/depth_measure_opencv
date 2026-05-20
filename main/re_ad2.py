def find_object(frame: np.ndarray, mask: np.ndarray):
    """
    Locate the largest masked contour and draw it on *frame*.
    Returns
    -------
    center : (cx, cy) or None
    bbox   : [[x1, y1], [x2, y2]] or None
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None, None
    x, y, w, h = cv2.boundingRect(c)
    center = (x + w // 2, y + h // 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.circle(frame, center, 5, (0, 0, 0), -1)
    return center, [[x, y], [x + w, y + h]]