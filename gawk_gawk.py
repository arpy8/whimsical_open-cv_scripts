import cv2
import pygame
from trackers.pose_tracker import PoseDetector

count = 0
hand_near_face = False

def put_text(img, text, position):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

def increase():
    global count
    count += 1
    play_sound()

def play_sound():
    pygame.mixer.init()
    pygame.mixer.music.load('assets/audio.mp3')
    pygame.mixer.music.play()

def main():
    global count, hand_near_face
    
    cap = cv2.VideoCapture(0)

    pose_detector = PoseDetector(staticMode=False,
                                modelComplexity=1,
                                smoothLandmarks=True,
                                enableSegmentation=False,
                                smoothSegmentation=True,
                                detectionCon=0.5,
                                trackCon=0.5)

    while True:
        success, raw_img = cap.read()

        img = pose_detector.findPose(raw_img)
        lmList, bboxInfo = pose_detector.findPosition(img, draw=True, bboxWithHands=True)

        try:
            if len(lmList) >= 16:
                
                right_lips_coords = lmList[10][:2]
                right_wrist_coords = lmList[16][:2]

                distance_chest, img, _ = pose_detector.findDistance(right_lips_coords, right_wrist_coords, img=img, color=(0, 255, 0), scale=5)
                # print(distance_chest)
                if round(distance_chest, 2) < 200:
                    if not hand_near_face:
                        increase()
                        hand_near_face = True
                else:
                    hand_near_face = False
                
                put_text(img, str(count), (img.shape[1]-150, 50))
                
        except Exception as e:
            print(e)
                
        cv2.imshow("Image", raw_img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
 
if __name__ == "__main__":
    main()