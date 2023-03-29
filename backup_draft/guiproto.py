#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 테스트용
# 임포트, 파라미터 

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import tkinter.messagebox as msgbox
from IPython.display import Image as im
import predict_password as pp #predict_password 모듈 임포트
import numpy as np
import tensorflow as tf
import os
import io
import time

#Define Token
START_ID = 0 #시퀀스 시작 토큰 ID
PAD_ID = 1   #시퀀스 공백 토큰 ID
END_ID = 2   #시퀀스 종결 토큰 ID

#Parameter
#입력 최대 길이
ENC_MAX_STEP = 10
#Label 최대 길이
DATA_MAX_OUTPUT = 4

DATA_PATH = "./data/convex_hull_5_test.txt"
PREFIX = "thermo"
#Label + End Token 최대 길이 
DEC_MAX_STEP = DATA_MAX_OUTPUT + 1 #max output lengths

BATCH_SIZE = 32
EPOCHS = 100
UNITS = 256
LEARNING_RATE = 0.001
BEAM_WIDTH = 4
OUTPUT_STD = None
DROPOUT_RATE = 0.0
#graident Clipping 파라미터
CLIPPING_VALUE = None
TRAIN_DATA_RATIO = 0.9


# In[2]:


# PN 모델 선언

import batcher
import attention_models
import pointer_network

PN = pointer_network.PointerNetwork(UNITS, BATCH_SIZE,
                    learning_rate=LEARNING_RATE,
                    output_std=0.1,
                    dropout_rate=DROPOUT_RATE)


checkpoint_dir = "./training_checkpoints/pn" + PREFIX
checkpoint_prefix = os.path.join(checkpoint_dir, "pn")
checkpoint = tf.train.Checkpoint(**PN.get_model())

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# In[3]:


# 작동부
    
def btncmd_test():
#label의 위치 추출 (type : dataframe), YOLOv5 모델 불러오기 
#YOLOv5 모델은 predict_password.py에 경로가 적혀있음
    password_results_df = pp.extract_password_label(input_image) 

# label에 해당하는 부분을 자르고, 150x200 크기로 리사이징
    cropped_image = pp.cropping_image(input_image, password_results_df) 

#test_data에 0부터 9까지의 키패드 이미지를 넣음
    test_data = pp.preprocess_evaluating_image(cropped_image) 
    test_data = np.expand_dims(test_data, axis=0) #encoder의 입력 layer에 맞춰줌


    inp, targ = test_data, np.zeros((test_data.shape[0],DEC_MAX_STEP), dtype=np.int32) #targ에는 dummy값을 넣어줌 (정답이 없는 이미지이기 때문)
#포인터 네트워크 모델로 예측 수행
    result = PN.predict(inp, targ) 
    predictions = result[0] 
    password_prediction = (predictions.numpy().squeeze()[:-1] - 1).tolist()  #종결토큰무시
    password_string = ''.join(map(str, password_prediction))
    msgbox.showinfo("비밀번호",f"비밀번호는{password_prediction}입니다.")
    
def btncmd_test2():
    print(root.filename)
    test_image=Image.open(root.filename)
    predict_image(test_image)

def start():   
    root=Tk()
    root.title('Doorlockstudy')
    root.geometry("360x640+600+100")

# 사용 예시
#image_path = ("경로입력")
#predict_image(image_path)
    def btncmd_open():
        global my_image
        root.filename=filedialog.askopenfilename(initialdir='', title="파일선택", 
        filetype=(('jpg files', '*.jpg'),('png files', '*.png'),('all files', '*.*')))

        Label(root, text=root.filename).pack()
        m_image=Image.open(root.filename)
        m=m_image.resize((360,600))
        my_image=ImageTk.PhotoImage(m)
        Label(image=my_image).pack(padx=0,pady=0)
        global input_image
        input_image=root.filename

    frame_btn=Frame(root,relief="solid",bd=1)
    frame_btn.pack(side="bottom",fill="both")
    open_btn=Button(frame_btn, width=20, text="파일열기", command=btncmd_open).grid(row=0,column=0,padx=15,pady=1)
    run_btn=Button(frame_btn, width=20, text="추측하기", command=btncmd_test).grid(row=0,column=1,padx=15,pady=1)  
    
    
    
    root.mainloop()


# In[ ]:





# In[ ]:




