import cv2
import mediapipe as mp
from trackers.hand_tracker import HandDetector

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

detector = HandDetector(maxHands=1)

cap = cv2.VideoCapture(0)

temp  = ""

def put_text(img, text, position):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

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

        hands, img = detector.findHands(image)
        
        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            
            switch = {0:"rock", 1:"", 2:"scissors", 4:"", 5:"paper"}
            file_name = switch.get(sum(fingers), fingers)
            if fingers[-1]==1 and sum(fingers)==1:
                put_text(img, str("pee pee break"), (img.shape[1]-150, 50))
            else:
                put_text(img, str(file_name), (img.shape[1]-150, 50))
                print(fingers)

            try:
                cv2.imshow("BG", f'assets/{file_name}.jpg')
            except Exception:
                pass
            
        cv2.imshow('Hand Gestures', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()