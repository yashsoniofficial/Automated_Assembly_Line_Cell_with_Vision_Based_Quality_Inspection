import cv2
import numpy as np

# --- Load template image (the good part) ---
template = cv2.imread("D:/admin/Pictures/Camera Roll/WIN_20251025_18_30_50_Pro.jpg", cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(0)  # Replace with your camera index or video file

# Resize to a reasonable size (e.g., 1/3 of frame width)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
scale_factor = 0.2
template = cv2.resize(template, (int(frame_width * scale_factor), int(frame_width * scale_factor)))
template = cv2.GaussianBlur(template, (3, 3), 0)
w, h = template.shape[::-1]

# --- Initialize camera ---
cap = cv2.VideoCapture(0)

# --- Matching threshold (tune this experimentally) ---
THRESHOLD = 0.6

# --- Define detection line (left side of frame) ---
DETECTION_X = 150  # Adjust based on your setup

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # --- Perform template matching ---
    result = cv2.matchTemplate(gray_blur, template, cv2.TM_CCOEFF_NORMED)

    # --- Find best match location ---
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw rectangle around detected region
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Draw detection line
    cv2.line(frame, (DETECTION_X, 0), (DETECTION_X, frame.shape[0]), (255, 0, 0), 2)

    # --- Check if object crossed detection line ---
    obj_center_x = top_left[0] + w // 2

    print("max_val",max_val)

    if obj_center_x <= DETECTION_X:
        if max_val < THRESHOLD:
            print("🚨 Defective part detected!")
            cv2.putText(frame, "DEFECTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Good part", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display similarity score
    cv2.putText(frame, f"Match: {max_val:.2f}", (frame.shape[1]-200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Conveyor Inspection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
