from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import model_from_json
from flask import Flask,render_template,Response


app=Flask(__name__)

json_file=open(r"C:\Users\saini\projects\emotion_detection\emlotion_detection1.json")
loaded_model_json=json_file.read()
json_file.close()
emotion_model=model_from_json(loaded_model_json)


face_classifier = cv2.CascadeClassifier(r"C:\Users\saini\projects\emotion_detection\haarcascade_frontalface_default.xml")
emotion_model.load_weights(r"C:\Users\saini\projects\emotion_detection\emotion_detect1 (1).h5")

emotion_labels = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad', 6:'Surprise'}

cap = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = cap.read()
        labels = []
        if not success:
            break
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
            
                prediction = emotion_model.predict(roi)
                label=int(np.argmax(prediction))
                label_position = (x,y-10)
                cv2.putText(frame,emotion_labels[label],label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/')
def index():
    return render_template('app.html')
@app.route('/video_feed',methods=['Post','Get'])
def video_feed():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)

