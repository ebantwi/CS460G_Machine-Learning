from image_display import display_image
import numpy as np, cv2, os
from torchvision.transforms import Grayscale 
from torchvision.transforms import Compose
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import PILToTensor
from PIL import Image


model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, padding = 1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(in_channels=30, out_channels=30, kernel_size=7, padding = 2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(in_channels=30, out_channels=30, kernel_size=11, padding = 3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Dropout(.5),
    nn.Flatten(),
    nn.Linear(in_features=270, out_features=256),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(in_features=128, out_features=7)
    )

model.load_state_dict(torch.load("./Finalmodel40"))

transform = Compose([
    ToTensor(),
    Grayscale()
    ])

def detect_face():
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

def main():
    detect_face()
    path = "./Screenshot/face_screenshot.jpg"
    img = cv2.imread(path)
    #Format for the Mul:0 Tensor
    img= cv2.resize(img,dsize=(48,48), interpolation = cv2.INTER_CUBIC)
    #Numpy array
    np_image_data = np.asarray(img)

    img = transform(np_image_data)
    img = img.unsqueeze(1)
    output = model(img)
    _, prediction = torch.max(output.data, 1)
    prediction = prediction[0].int()
    emotion = ""
    if prediction == 0:
        emotion = "angry"
    if prediction == 1:
        emotion = "disgust"
    if prediction == 2:
        emotion = "fear"
    if prediction == 3:
        emotion = "happy"
    if prediction == 4:
        emotion = "neutral"
    if prediction == 5:
        emotion = "sad"
    if prediction == 6:
        emotion = "suprise"
    
    display_image('./Screenshot/face_screenshot.jpg', emotion, 1)
    
if __name__ == "__main__":
    main()