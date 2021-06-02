import cv2, time, math
import numpy as np
import mediapipe as mp
import HandModule as htm


###############################
wCam, hCam = 640, 480
###############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
ptime = 0

detector = htm.HandDetector(detectionCon=0.7)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    img = detector.FindHands(img)
    lmlist = detector.FindPos(img, draw=False)
    
    if len(lmlist) != 0:
        fingers = []
        if lmlist[tipIds[0]][1] < lmlist[tipIds[0]-1][1]:
            fingers.append(1)
        else: fingers.append(0)
        
        for id in range(1, 5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else: fingers.append(0)

        cv2.putText(img, str(sum(fingers)), (100, 150), cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 5)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break