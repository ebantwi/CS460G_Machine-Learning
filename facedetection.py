import numpy as np, cv2, os

if not os.path.isdir('Screenshot'):
    os.makedirs('Screenshot')

#cv2_base_directory = os.path.dirname(os.path.abspath(cv2.__file__))
#classifier_path = cv2_base_directory+'\\data\\haarcascade_frontalface_default.xml'
classifier_path = 'haarcascades/haarcascade_frontalface_default.xml'

face_classifier = cv2.CascadeClassifier(classifier_path)

# To capture video from webcam.
cap = cv2.VideoCapture(0)
while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_classifier.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)
    # Display
    cv2.imshow('img', img)
    cv2.imwrite('Screenshot/face_screenshot.jpg',img)
    # Stop if escape|enter key is pressed
    k = cv2.waitKey(30) & 0xff

    if (cv2.waitKey(1) == 13) | (k == 27): #13 for return (enter) key
        break
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
