import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Replace with your camera index or video file

# Parameters
INSPECTION_LINE_X = 120
MIN_AREA = 800

# Yellow color range (tighter range for less background interference)
LOWER_YELLOW = np.array([22, 150, 120])
UPPER_YELLOW = np.array([35, 255, 255])

# Capture background (take first few frames)
print("Calibrating background... please keep conveyor empty.")
for i in range(30):
    ret, bg_frame = cap.read()
background = cv2.GaussianBlur(bg_frame, (5, 5), 0)
print("✅ Background captured!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Define ROI (adjust as per your conveyor area)
    roi = frame[200:400, 100:540]  # y1:y2, x1:x2
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Background subtraction
    diff = cv2.absdiff(cv2.GaussianBlur(roi, (5, 5), 0), background[200:400, 100:540])
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, motion_mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)

    # Color mask for yellow parts
    color_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # Combine both masks (object must be moving + yellow)
    combined_mask = cv2.bitwise_and(color_mask, motion_mask)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours in ROI
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw inspection line in ROI
    cv2.line(roi, (INSPECTION_LINE_X, 0), (INSPECTION_LINE_X, roi.shape[0]), (0, 0, 255), 2)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area
        extent = area / (w * h if w * h > 0 else 1)
        aspect_ratio = min(w, h) / max(w, h)

        is_square_like = 0.9 < aspect_ratio < 1.1
        is_solid = solidity > 0.9
        is_filled = extent > 0.7

        if is_square_like and is_solid and is_filled:
            color = (0, 255, 0)
            label = "GOOD"
        else:
            color = (0, 0, 255)
            label = "DEFECTIVE"
            if cx <= INSPECTION_LINE_X:
                print("⚠️ Signal: Defective part crossed inspection line!")

        cv2.drawContours(roi, [box], 0, color, 2)
        cv2.putText(roi, label, (int(cx - 30), int(cy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    frame[200:400, 100:540] = roi

    # Display
    cv2.imshow("Inspection", frame)
    cv2.imshow("Mask", combined_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

