import cv2

cap = cv2.VideoCapture(0)

while True:
    suscess,frame = cap.read()
    if not suscess:
        print("Failed to grab frame")
        break
    
    cv2.line(frame, (0, 400), (640, 400), (0, 255, 0), 4)
    cv2.putText("Deadlift threshould ")

    cv2.imshow("Vinafit live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows