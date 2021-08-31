import cv2
#from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np


#root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#from keras import load_model
from keras.models import load_model

vgg = load_model('live_own.h5')
class_names = ['Real','Spoof']
video = cv2.VideoCapture(0)
while True:
    try:
        ret,frame = video.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        print(len(faces))
        for (x,y,w,h) in faces:  
            face = frame[y-5:y+h+5,x-5:x+w+5]
            resized_face = cv2.resize(face,(160,160))
            resized_face = resized_face.astype("float") / 255.0
            # resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = vgg.predict_classes(resized_face)
            out = class_names[preds[0]]
            print(out)
            
            cv2.putText(frame, out, (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.rectangle(frame, (x, y), (x+w,y+h),(0, 0, 255), 2)
            
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        pass
video.release()        
cv2.destroyAllWindows()
