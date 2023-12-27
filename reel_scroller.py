import cv2
import pyautogui as pg
import mediapipe as mp
from trackers.hand_tracker import HandDetector

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

cap = cv2.VideoCapture(0)
temp  = ""

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        hands, img = detector.findHands(image, draw=True, flipType=True)
        
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]

            p1 = lmList1[1][0:2]
            p2 = lmList1[8][0:2]
            
            length, info, img = detector.findDistance(p1, p2, img, scale=10)

            print(f"Distance: {length}")
            pg.press("down") if (length > 180) else None 
            # print("down") if (length > 140) else None 
            
            cv2.imshow('Hand Gestures', image)
            if cv2.waitKey(5) & 0xFF == 27: 
                break

cap.release()
cv2.destroyAllWindows()