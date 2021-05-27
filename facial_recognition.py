import cv2
import os
import torch
import convolutional_neural_network
import image_processing
imagecount = 500
cap = cv2.VideoCapture(0)
model = torch.load('facial_cnn.pt')
classifier = cv2.CascadeClassifier("D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")
#path of classfier should be changed to the corresponding saving path of haarcascade_frontalface_default.xml, this path only exists in Hana's laptop

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects) > 0:
        for faceRect in faceRects:
            global img
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
            face_gray = gray[(y - 10):(y + h + 10), (x - 10): (x + w + 10)]
            img = frame[(y - 10):(y + h + 10), (x - 10): (x + w + 10)]
            probs, index = convolutional_neural_network.cnn_output(img)
            print(probs)
            if index == 1:
                cv2.putText(frame, 'Hana', (x-15, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == '__main__':
    cap.release()
    cv2.destroyAllWindows()
