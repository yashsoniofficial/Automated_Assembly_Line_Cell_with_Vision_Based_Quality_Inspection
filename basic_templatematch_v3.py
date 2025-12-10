import cv2
import numpy as np
import os

# -----------------------------
# Load templates (GOOD and BAD)
# -----------------------------
root = os.getcwd()

goodImg_path = os.path.join(root, 'good_box.jpg')
badImg_path  = os.path.join(root, 'bad_box.jpg')

goodImg = cv2.imread(goodImg_path)
badImg  = cv2.imread(badImg_path)

goodImg = cv2.cvtColor(goodImg, cv2.COLOR_BGR2RGB)
badImg  = cv2.cvtColor(badImg, cv2.COLOR_BGR2RGB)

# Extract the templates exactly as you did before
good_template = goodImg[238:901, 528:1447]
bad_template  = badImg[116:894,510:1413]   # adjust if needed

hg, wg, _ = good_template.shape
hb, wb, _ = bad_template.shape

# -----------------------------
# Initialize camera
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("System running... Comparing both templates continuously")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # Template match both templates
    # -----------------------------
    result_good = cv2.matchTemplate(frame, good_template, cv2.TM_CCOEFF_NORMED)
    _, score_good, _, loc_good = cv2.minMaxLoc(result_good)

    result_bad = cv2.matchTemplate(frame, bad_template, cv2.TM_CCOEFF_NORMED)
    _, score_bad, _, loc_bad = cv2.minMaxLoc(result_bad)

    # -----------------------------
    # Decide classification
    # -----------------------------
    if score_good > score_bad:
        status = "GOOD"
        color = (0, 255, 0)
        top_left = loc_good
        w, h = wg, hg
        best_score = score_good
    else:
        status = "BAD"
        color = (0, 0, 255)
        top_left = loc_bad
        w, h = wb, hb
        best_score = score_bad

    # -----------------------------
    # Draw bounding box
    # -----------------------------
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, color, 2)

    # -----------------------------
    # Display info
    # -----------------------------
    cv2.putText(frame, status, (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"Good score: {score_good:.3f}", (50, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"Bad score:  {score_bad:.3f}", (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Inspection Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
