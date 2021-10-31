import cv2
import uuid # universally unique identifiers
import os

cap = cv2.VideoCapture(0)

while cap.isOpened():
    return_, frame = cap.read()

    frame = frame[100:100+250, 250:250+250, :]

    cv2.imshow("Image", frame)

    #anchors
    if cv2.waitKey(1) & 0xFF == ord("a"):
        imgname = os.path.join("data/anchor", f"{uuid.uuid1()}.jpg")
        cv2.imwrite(imgname, frame)

    #positives
    if cv2.waitKey(1) & 0xFF == ord("p"):
        imgname = os.path.join("data/positive", f"{uuid.uuid1()}.jpg")
        cv2.imwrite(imgname, frame)

    #negative
    if cv2.waitKey(1) & 0xFF == ord("n"):
        imgname = os.path.join("data/negative", f"{uuid.uuid1()}.jpg")
        cv2.imwrite(imgname, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
