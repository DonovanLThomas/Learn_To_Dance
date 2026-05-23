import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    flipped_frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)

    if cv2.waitkey(1) & '0xFF' == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()