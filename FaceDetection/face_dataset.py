import cv2
import os

body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_id = input('\n Enter user id : ')
print('\n [INFO] Capturing your face. Look at the camera and wait ...')

count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayframe = cv2.equalizeHist(grayframe)
    faces = face_cascade.detectMultiScale(grayframe, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        
        cv2.imwrite('dataset/User' + str(face_id) + '.' + str(count) + '.jpg', grayframe[y:y+h,x:x+w])
        cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    elif count>= 200:
        break
    
print('\n [INFO] Exiting Program and cleanup stuff')
cap.release()
cv2.destroyAllWindows()

