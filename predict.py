import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import pyttsx3
from mediapipe import solutions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard
mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])

def hand_detected(results):
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh,rh])

no_sequences = 50
sequence_length = 30
# actions = np.array(['A', 'B', 'C'])
# actions = np.array(['G', 'H', 'I'])
actions = np.array(['J', 'K', 'L'])
# actions = np.array(['Hello', 'Thanks', 'ILoveYou'])
# actions = np.array(['Where', 'Doctor', 'Help'])
# actions = np.array(['Mother', 'Give', 'Me', 'Chapati'])
label_map = {label:num for num,label in enumerate(actions)}

model = Sequential()
model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,1662)))
model.add(LSTM(128,return_sequences=True,activation='relu'))
model.add(LSTM(64,return_sequences=False,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(actions.shape[0],activation='softmax'))
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
# model.load_weights('Trained Models/A_B_C.h5')
# model.load_weights('Trained Models/G_H_I.h5')
model.load_weights('Trained Models/J_K_L.h5')
# model.load_weights('Trained Models/Hello_Thanks_ILoveYou.h5')
# model.load_weights('Trained Models/Where_Doctor_Help.h5')
# model.load_weights('Trained Models/Mother_Give_ME_Chapthi.h5')

colors = [(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(16,117,245),(255,117,16),(117,245,16),(255,117,16),(117,245,16),(16,117,245)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100),90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0,85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

sequence = []
sentence = []
threshold = 0.6
cap = cv2.VideoCapture(0)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
def open_camera():
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            # draw_styled_landmarks(image, results)
            
            if hand_detected(results).sum() == 0:
                cv2.rectangle(image, (0,0), (1280,40), (245,117,16), -1)
                cv2.putText(image, 'Start Action', (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('Train Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue
            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:30]
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence,axis=0))[0]
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]: 
                            text = actions[np.argmax(res)]
                            sentence.append(actions[np.argmax(res)])
                            engine.say(text)
                            engine.runAndWait()
                    else:
                        text = actions[np.argmax(res)]
                        sentence.append(actions[np.argmax(res)])
                        engine.say(text)
                        engine.runAndWait()
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                    
                image = prob_viz(res, actions, image, colors)
                cv2.rectangle(image, (0,0), (1280,40), (245,117,16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.imshow('Train Feed', image)
            
            if time.time() - start_time > 40:
                print(sentence)
                break
   
            if cv2.waitKey(10) & 0xFF == ord('p'):
                print(sentence)
                break
    cap.release()
    cv2.destroyAllWindows()