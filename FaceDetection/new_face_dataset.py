import cv2
import os
import time
import RPi.GPIO as GPIO

BUTTON_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def capture_face_dataset(face_id):
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print('\n [INFO] Press button to start capturing ...')
    
    while GPIO.input(BUTTON_PIN) == GPIO.LOW:
        time.sleep(0.1)
        
    print('\n [INFO] Capturing your face. Look at the camera and wait ...')

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayframe = cv2.equalizeHist(grayframe)
        faces = face_cascade.detectMultiScale(grayframe, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            timestamp = int(time.time() *1000)            
            cv2.imwrite('dataset/User.'+str(face_id)+'.'+str(timestamp)+'.jpg', grayframe[y:y + h, x: x+w])
            cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            print('\n [INFO] Stop capturing your face.')
            break

    while GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        time.sleep(0.1)
    #print('\n [INFO] Exiting program')
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup([BUTTON_PIN])

if __name__ == "__main__":
    face_id = 3
    capture_face_dataset(face_id)



