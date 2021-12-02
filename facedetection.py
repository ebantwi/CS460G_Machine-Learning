import numpy as np, cv2, os

cv2_base_directory = os.path.dirname(os.path.abspath(cv2.__file__))
classifier_path = os.path.join(cv2_base_directory, 'data/haarcascade_frontalface_default.xml')

face_classifier = cv2.CascadeClassifier(classifier_path)

def detect_face(img, size=0.5):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(grayscale, 1.3, 5)  #scalefactor, min_neighbours
    if face is ():
        return img
    
    for (x,y,w,h) in face:
        #can crop here if desired to show only face
        #cv2.rectangle(img, begin_point, end_point, color, thickness)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 1)
         
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Detected_Face:', detect_face(frame))
    if cv2.waitKey(1) == 13: #13 for return (enter) key
        break
        
cap.release()
cv2.destroyAllWindows()
