# %%


import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import keyboard


actions = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'hello', 'my', 'call', 'name'
])



sequences = 30
frames = 10


PATH = os.path.join('data')


for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()


with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    for action, sequence, frame in product(actions, range(sequences), range(frames)):
        if frame == 0: 
            while True:
                success, image = cap.read()
                if not success:
                    print("Failed to capture image")
                    continue
                if keyboard.is_pressed(' '):
                    break

                results = image_process(image, holistic)
                image.flags.writeable = True
                draw_landmarks(image, results)

                image = image.copy()
                cv2.putText(image, 'Recroding data for the "{}". Sequence number {}.'.format(action, sequence),
                            (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(image, '', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Press "Space" when you are ready.', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                cv2.waitKey(1)
                
                if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
                    break
        else:
            success, image = cap.read()
            if not success:
                print("Failed to capture image")
                continue
            results = image_process(image, holistic)
            image.flags.writeable = True
            draw_landmarks(image, results)

            image = image.copy()
            cv2.putText(image, 'Recroding data for the "{}". Sequence number {}.'.format(action, sequence),
                        (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.imshow('Camera', image)
            cv2.waitKey(1)

        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

        keypoints = keypoint_extraction(results)
        frame_path = os.path.join(PATH, action, str(sequence), str(frame))
        np.save(frame_path, keypoints)

    cap.release()
    cv2.destroyAllWindows()
