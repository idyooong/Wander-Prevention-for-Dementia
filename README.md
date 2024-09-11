#  치매환자 무단외출 방지 시스템
[2024.03 ~ 2024.06] 캡스톤디자인 프로젝트 졸업작품

## 1️⃣ 프로젝트 소개  

 - 이 시스템은 치매환자가 보호자 없이 혼자 외출하는 상황에서 발생할 수 있는 다양한 위험을 예방·방지하기 위해 설계됨.
 - 이를 통해 보호자는 치매환자의 안전을 효과적으로 모니터링할 수 있고 수시로 확인해야 하는 불편을 줄일 수 있으며 치매환자가 안전하게 생활할 수 있도록 도울 수 있음.

</br>

## 2️⃣ 프로젝트 배경 및 필요성  

 - 현재 치매환자의 수가 증가하면서 그로 인한 사회적 문제가 대두되고 있음.
 - 많은 보호자들이 치매환자의 무단 외출이나 배회로 인한 불안과 스트레스를 겪고 있음.
 - 치매환자가 보호자의 동의없이 외출할 경우 예상치 못한 사고나 위험이 발생할 수 있음.
 - 치매환자의 외출 사실을 즉각적으로 알 수 있다면 치매환자의 사고 예방과 보호자의 스트레스 감소에 큰 도움이 될 것.

</br>

## 3️⃣개발환경 및 도구  

 * Python 3.9.2
 * OpenCV 4.5.5
 * 라즈베리파이4 (32bit OS) 
 * 저음강화 2채널 스마트폰 3D 멀티 스피커
 * 앱코 APC850 FHD 웹캠
 * 라즈베리파이 공식 7인치 터치스크린
 * 소형 인체 적외선 감지센서 모듈 (NS-PIRSM)
 * 초음파 거리센서 모듈 HC-SR04 [SZH-EK004]
 * RGB LED

</br>

## 4️⃣ 순서도

![치매흐름도 drawio](https://github.com/user-attachments/assets/15c59437-8525-4ce8-94c8-7a0bb59775ac)

</br>

## 5️⃣ 구현 사항(주요 기능)

* ### PIR 센서를 이용한 움직임 감지와 카메라 전원 조절

  
    ![image](https://github.com/user-attachments/assets/43542686-4e79-4ba2-b4e3-a76b3b8f6df7)

  [그림 1] 사람의 움직임이 감지되면 파란 LED가 켜짐과 동시에 카메라의 전원을 켬.

  </br>

- ### 치매환자 인식 카메라

  
    ![결과물 모자이크](https://github.com/user-attachments/assets/81fcd24b-edde-4591-8919-a4913dc6e9e9)
  
   [그림 2] OpenCV를 통해 치매환자와 보호자 구분하여 얼굴 인식 

</br>

  * ### 치매환자로 인식될 경우
 
    ![환자 작동방식](https://github.com/user-attachments/assets/b5b0cc0b-07c8-479d-90c5-da987938fc64)
    
    [그림 3] 치매환자일 경우 초록 LED가 켜지고 보호자에게 알림을 줌.


     - __초음파 거리센서__ : 현관문과 치매환자 사이의 거리 측정 (거리는 각각 50cm, 100cm, 150cm로 설정)
      
     - __무단외출 알림 스피커__ : 측정된 거리에 따라 스피커에서 경고음 울림
      
       -현관문과의 거리가 가까워질수록 스피커 소리 크게 하여 보호자에게 긴급함 알림.
    
       -보호자가 집 안에 있을 때 적절한 조치를 취할 수 있도록 하기 위함.
      
     - __이메일 전송__ : SMTP 방식을 사용하여 보호자 휴대폰으로 이메일 전송
   
       -이메일에 실시간 시간 정보와 거리 센서 값을 같이 전송해줌으로써 치매환자가 현관문에 얼만큼 왔는지 알 수 있음.
    
       -보호자가 외출했거나 스피커 소리를 못 들었을 때 상황 판단을 하는 데 도움을 주기 위함.

</br>

   * ### 보호자로 인식될 경우
        - 초록 LED는 작동하지만 그 외 다른 센서들은 작동하지 않음. (__치매환자일 때만 작동__)

</br>

- ### 보호자 추가 등록 버튼
    ![보호자 추가등록방식](https://github.com/user-attachments/assets/3b33872b-0116-4717-826f-322080c69ec2)
  
  [그림 4] 등록 버튼 누르면 추가 보호자를 새로 학습하여 보호자 추가 등록이 가능해짐.

  
  버튼을 통해 자신이 원하는 만큼 데이터를 수집할 수 있으며 수집이 끝나게 되면 모델을 재학습시켜 다시 카메라가 켜졌을 때 unknown이 아닌 family로 인식되도록 함.



   ![image](https://github.com/user-attachments/assets/66ad225d-d1ad-4e72-abc1-df75ed978d20)
   
   [그림 5] dataset 파일에 신규 보호자의 얼굴 데이터가 수집되었음을 알 수 있음. 

</br>

- ### RGB LED : 시스템 상태에 따라 3가지의 LED 불 켜짐.
   - BLUE : 움직임 감지 시 켜짐
   - GREEN : 치매환자나 보호자 인식될 시 켜짐 (미등록자들은 해당 안 됨.)
   - RED : 스피커 작동 시 깜박깜박 거림

</br>

## 6️⃣ 결과물

![image](https://github.com/user-attachments/assets/4839ca75-1743-4cf6-8e3e-1442b6e2c5f4)

</br>

## 7️⃣ 기대효과 및 활용 분야

 - 사고 예방 및 보호자의 심리적 부담감 감소
 - CCTV를 활용한 모니터링 시스템으로 집뿐만 아니라 병원, 요양원 등 다양한 시설에서 사용 가능
 - 다중 환자 관리 : 보호자 추가 기능과 유사하게 치매환자 추가 기능 구현 시 여러 환자를 동시에 관리할 수 있음
 - 다양한 대상 적용 : 치매환자 외에도 영유아나 관리가 필요한 다른 대상자들도 관리할 수 있도록 확장하여 활용 가능.
