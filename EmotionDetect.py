import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces_detected:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        yhat= model.predict(cropped_img)
        cv2.putText(img, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: "+labels[int(np.argmax(yhat))])
    
    cv2.imshow("Result",img)
    key=cv2.waitKey(1)
    if key==27:
         break


cap.release()
cv2.destroyAllWindows