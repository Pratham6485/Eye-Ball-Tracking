import cv2
import numpy as np
import module as m

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img,face=m.faceDetector(img,gray,False)
    if face is not None:
        img,pointlist=m.faceLandmakDetector(img,gray,face,False)
        RightEyePoint = pointlist[36:42]
        LeftEyePoint = pointlist[42:48]
        for i in RightEyePoint:
            cv2.circle(img, i, 3, m.ORANGE, 1)
        for i in LeftEyePoint:
            cv2.circle(img, i, 3, m.ORANGE, 1)

    cv2.imshow("wimdow",img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
