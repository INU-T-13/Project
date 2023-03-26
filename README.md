# INUCapstone-Project-Thermal-Doorlock-Prediction

인천대학교 캡스톤 디자인, T17 팀의 프로젝트입니다.
## 사용 기술
* `Python` 언어로 작성되었으며, 구동은 `Jupyter Notebook` 환경에서 주로 이루어졌습니다.
* 객체 탐지 모델은 `Pytorch` 를 기반으로 한 `YOLOv5`를 사용하였습니다.
* 딥러닝에는 `Tensorflow` 라이브러리를 활용하였습니다.
* GUI 구현은 `ADD_THIS_BLOCK` 라이브러리를 사용하였습니다.
## 팀원 구성 및 역할
* 이용학 :
* 정보규 :
* 배동제 :
## 기능 소개

#### 데이터 처리
* 기본적으로, 열 자국이 남은 도어락을 열화상 카메라로 촬영합니다.

 <img width="30%" src="https://user-images.githubusercontent.com/114134093/227794951-8290502e-07b6-48d3-9f1f-2bf732a2d717.jpg"/>
 
* 열화상 카메라로 촬영한 이미지의 품질은 가변적입니다. 따라서 그 차이를 줄이기 위해 이미지에 필터를 적용합니다.
 <img width="30%" src="https://user-images.githubusercontent.com/114134093/227795340-de020422-ed98-419c-8bc5-1398b28754c7.jpg"/>



#### 데이터 학습
* 먼저, Doorlock과 Password로 이미지의 레이블을 생성하고, 객체 탐지 모델 (YOLOv5를 사용합니다)을 통해 학습합니다.
> <img width="30%" src="https://user-images.githubusercontent.com/114134093/227794518-67723dd8-77bf-47d6-bb75-4506d865ceb4.jpg"/>

* 그리고, 데이터 처리 부분에서 처리된 이미지를 바탕으로, Password에 해당하는 레이블을 자릅니다.

![cropped_password](https://user-images.githubusercontent.com/114134093/227795389-ba3cfdc1-8aaa-49eb-96d1-1e9a81f693b2.jpg)

* 0부터 9까지, 10개의 버튼에 대해 각각의 버튼 이미지들을 잘라내고, 하나의 데이터로 만듭니다. 아래는 해당 부분을 처리하는 코드의 일부입니다.

        keypad_imgs = []
        keypad_imgs.append(resized_pix[4*40:(4+1)*40, 1*50:(1+1)*50])  # 0번 키패드 추가
        for y in range(1, 4):
            for x in range(3):
                keypad_imgs.append(resized_pix[y*40:(y+1)*40, x*50:(x+1)*50]) # 1번부터, 9번까지

        keypad_imgs = np.array(keypad_imgs)
        data_x.append(keypad_imgs)
        

이런식으로 데이터를 만들고, Pointer Network Model을 통해 학습을 진행합니다.
  
#### 비밀번호 예측하기

충분한 데이터가 학습되었으면, 비밀번호를 예측해볼만 합니다.
* 2023-03-27 기준, 약 1300개의 데이터 셋으로 학습을 진행하였습니다.
* 데이터 셋은 지속적으로 업데이트를 하고 있는 중이며, 대략 4~5천개 정도를 미니멈으로 잡고 있습니다.
```predict_password.py```에 Custom Image에 대한 전처리 및 예측하는 과정이 담겨있습니다.

## 사용 예시 - GUI 환경

## 성능 평가

* 현재 랜덤한 이미지를 받으면, 해당 배열이 구성하고 있는 숫자들은 어느정도 인식하는 편입니다.
* 예를 들어, 비밀번호가 `2443` 이라는 이미지를 받으면, `2443`을 정확히 집진 못 하더라도 `2446`, `2434`와 같이 유사한 비밀번호를 예측하고 있습니다.

  
