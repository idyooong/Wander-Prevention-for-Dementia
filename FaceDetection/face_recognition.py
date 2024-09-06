# 사람 검출_Haar Cascade(얼굴 감지), opencv로 처리
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
import pygame
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

GPIO.setmode(GPIO.BCM)
PIR_PIN = 6
PIR_LED_PIN = 17
CAM_LED_PIN = 19
SPEAKER_LED_PIN = 5
TRIGER = 24
ECHO = 23
BUTTON_PIN = 26
GPIO.setwarnings(False)
GPIO.setup(PIR_PIN, GPIO.IN)
GPIO.setup(PIR_LED_PIN, GPIO.OUT)
GPIO.setup(CAM_LED_PIN, GPIO.OUT)
GPIO.setup(SPEAKER_LED_PIN, GPIO.OUT)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(TRIGER, GPIO.OUT)

def play_sound(volume):
    beep_sound.set_volume(volume)
    beep_sound.play()
    time.sleep(beep_sound.get_length())

def measure_distance():
    start_time = time.time()
    end_time = time.time()
   
    GPIO.output(TRIGER, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(TRIGER, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIGER, GPIO.LOW)
   
    while GPIO.input(ECHO) == GPIO.LOW:
        start_time = time.time()

    while GPIO.input(ECHO) == GPIO.HIGH:
        end_time = time.time()

    period = end_time - start_time
    distance = period * 17150
    distance = round(distance, 2)

    return distance

def add_new_family():
    print("\n [INFO] Button pressed! Start adding a new family.")
    time.sleep(0.2)
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.2)
    subprocess.run(["python3", "new_face_dataset.py"])
    subprocess.run(["python3", "new_face_training.py"])
    print("\n [INFO] Successfully added face!\n")
    time.sleep(0.2)
    
    while GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        time.sleep(0.1)
    GPIO.cleanup([BUTTON_PIN])
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
def debounce_button(pin):
    initial_state = GPIO.input(pin)
    time.sleep(0.05)
    return GPIO.input(pin) == initial_state

pygame.mixer.init()
sound_file_path = "beep.wav"
beep_sound = pygame.mixer.Sound(sound_file_path)

id = 0
names = ['None', 'patient', 'family', 'family']

try:
    print("\n [INFO] System powered on (CTRL+C to exit)")
    time.sleep(2)
    print("\n [INFO] System working...")
    
    while GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        time.sleep(0.1)
        
    while True:
        GPIO.output(SPEAKER_LED_PIN, GPIO.LOW)
        if GPIO.input(PIR_PIN) == GPIO.HIGH:
            GPIO.output(PIR_LED_PIN, GPIO.HIGH)
            
            t = time.localtime()
            print("%d:%d:%d Motion detected!" % (t.tm_hour, t.tm_min, t.tm_sec))

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            minW = 0.1 * cap.get(3)
            minH = 0.1 * cap.get(4)
           
            patient_detected = False
            motion_detected = True
            
            while motion_detected:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grayframe = cv2.equalizeHist(grayframe)
                faces = face_cascade.detectMultiScale(grayframe, scaleFactor=1.1, minNeighbors=5, minSize=(int(minW), int(minH)))

                if not len(faces):
                    GPIO.output(CAM_LED_PIN, GPIO.LOW)
                    patient_detected = False

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    id, confidence = recognizer.predict(grayframe[y:y + h, x:x + w])
                    user_id = ' user' + str(id)

                    if confidence < 70:
                        if id == 3:
                            confidence = "  {0}%".format(round(100 - confidence))
                        else:
                            confidence = "  {0}%".format(round(100 - confidence)*2)
                        id = names[id]
                        GPIO.output(CAM_LED_PIN, GPIO.HIGH)
                        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
                   
                    else:
                        id = "unknown"
                        #confidence = "  {0}%".format(round(100 - confidence))
                        user_id = ''
                        GPIO.output(CAM_LED_PIN, GPIO.LOW)
                        patient_detected = False

                    cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                    cv2.putText(frame, user_id, (x, y - 35), font, 1, (0, 255, 255), 2)
                   
                    if id == 'patient':
                        patient_detected = True
                       
                frame_resized = cv2.resize(frame, (800, 480))
                cv2.imshow('camera', frame_resized)

                if patient_detected:
                    distance = measure_distance()
                    GPIO.output(SPEAKER_LED_PIN, GPIO.HIGH)
                    smtp = smtplib.SMTP('smtp.gmail.com', 587)
                    smtp.starttls()
                    smtp.login('SENDER_EMAIL@gmail.com','PASSWORD')
                    msg = MIMEText('Patient trying to get out!!    distance : {} cm'.format(distance))
                    msg['From'] = 'SENDER_EMAIL@gmail.com'
                    msg['Subject'] = 'WARNING : PATIENT DETECTED! ({}:{}:{})'.format(t.tm_hour, t.tm_min, t.tm_sec)
                    msg['To'] = 'FAMILY_EMAIL@gmail.com'
                    smtp.sendmail('SENDER_EMAIL@gmail.com','FAMILY_EMAIL@gmail.com', msg.as_string())
                    smtp.quit()
                    
                    print("Distance", distance, "cm")

                    if distance < 51:
                        print("danger")
                        play_sound(1.0)
                       
                    elif 51 <= distance < 101:
                        print("caution")
                        play_sound(0.5)
                    else:
                        print("safe")
                        play_sound(0.1)
                    GPIO.output(SPEAKER_LED_PIN, GPIO.LOW)


                if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
                    if debounce_button(BUTTON_PIN):
                        add_new_family()
                        motion_detected = False
                        break
                   
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
        else:
            GPIO.output(PIR_LED_PIN, GPIO.LOW)
            GPIO.output(CAM_LED_PIN, GPIO.LOW)
            print("No motion detected")
            cv2.destroyAllWindows()
            time.sleep(0.5)

except KeyboardInterrupt:
    print("\n [INFO] Exiting Program and cleanup stuff")
    GPIO.cleanup()
