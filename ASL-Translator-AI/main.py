# %%


import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python


PATH = os.path.join('data')


actions = np.array(os.listdir(PATH))


model = load_model('my_model.keras')


tool = language_tool_python.LanguageToolPublicAPI('en-UK')

sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()


with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
   
    while cap.isOpened():
       
        success, image = cap.read()
        if not success:
            print("Failed to capture image")
            continue
       
        results = image_process(image, holistic)
        
        image = image.copy()
       
        draw_landmarks(image, results)
        
        keypoints.append(keypoint_extraction(results))

       
        if len(keypoints) == 10:
            
            keypoints = np.array(keypoints)
            
            prediction = model.predict(keypoints[np.newaxis, :, :])
           
            keypoints = []

           
            if np.amax(prediction) > 0.9:
               
                if last_prediction != actions[np.argmax(prediction)]:
                    
                    sentence.append(actions[np.argmax(prediction)])
                    
                    last_prediction = actions[np.argmax(prediction)]

        
        if len(sentence) > 7:
            sentence = sentence[-7:]

       
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

       
        if sentence:
           
            sentence[0] = sentence[0].capitalize()

       
        if len(sentence) >= 2:
           
            if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
              
                if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

      
        if keyboard.is_pressed('enter'):
            text = ' '.join(sentence)
        else:
            text = ' '.join(sentence) 

        try:
            grammar_result = tool.correct(text)  
        except Exception as e:
            print(f"Grammar correction failed: {e}") 
            grammar_result = text


        if grammar_result:

            image = image.copy() 
           
            textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2


            if image.dtype != 'uint8':
                image = image.astype('uint8')
            if len(image.shape) == 2:  
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            
            image = image.copy() 
           
            cv2.putText(image, grammar_result, (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:

            image = np.ascontiguousarray(image)
           
            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2
            

            image = image.copy() 
            
            cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        

        cv2.imshow('Camera', image)

        cv2.waitKey(1)

       
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

   
    cap.release()
    cv2.destroyAllWindows()

   
    tool.close()
