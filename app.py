from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import mediapipe as mp
import time
import math
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from collections import OrderedDict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
global fpredict_result
predict_result=[]
@app.route('/video_feed')
def video_feed(stop=False):
    stop=request.args.get('stop') 
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    def call():
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        classifier = Classifier("Model/keras_model.h5" , "Model/labels.txt")
        offset = 20
        imgSize = 300
        counter = 0
        labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]
        while True:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
                imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize-wCal)/2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                    prediction , index = classifier.getPrediction(imgWhite, draw= False)
                    print(prediction, index)

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize
                    prediction , index = classifier.getPrediction(imgWhite, draw= False)

            
                cv2.rectangle(imgOutput,(15,10),(15+400, 35+60-15),(0,255,0),cv2.FILLED)  

                cv2.putText(imgOutput,labels[index],(13,45),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
                print(labels[index])
                if(len(predict_result)==0):
                    predict_result.append(labels[index])
                elif(labels[index] not in predict_result):
                    predict_result.append(labels[index])
                else:
                    pass
            # cv2.imshow('Image', imgOutput)
            ret, buffer = cv2.imencode('.jpg', imgOutput)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    if(stop):
        
        res=' '.join(predict_result)
        return render_template('index.html',ans=res)
    return Response(call(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
