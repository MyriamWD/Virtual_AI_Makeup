# face_detection.py

import numpy as np
import cv2
from AIMakeup import Makeup,Face,Organ,NoFace

# Load the pre-trained classifiers. These can be found in opencv/data/haarcascades
# but are also inside the lab directory for your convenience.


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # Open the webcam device.
while True:
    ret, img = cap.read()  # Read an frame from the webcam.


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for face detection.

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
       cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # Draw a blue box around each face.
       mu = Makeup()
       im, temp_bgr, facesAiMakeup = mu.read_and_mark_no_path(img)
       for face in facesAiMakeup['test']:
         face.organs['mouth'].lipstipN(171.845, 119.85)

    cv2.imshow('new',img)

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
