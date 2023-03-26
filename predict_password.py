import torch
import cv2
from pathlib import Path
import filter_utils
import PIL.Image as pilimg
import numpy as np


def extract_password_label(image_input):
	#YOLOv5 모델과 이미지를 받아서 label을 추출하는 함수
	# Set YOLOv5 path
	yolov5_path = Path('/yolov5')
	model_path = 'best.pt'

	# Load YOLOv5 model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
	model.to(device)

	results = model(image_input)

	# label명이 Password인 레이블 추출
	password_results_df = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'Password']

	return password_results_df


def cropping_image(image_input, dataframe):
	image = cv2.imread(image_input)
	filtered_img = filter_utils.apply_new_filters(image) #전체 이미지에 대해 필터 적용
	xmin, ymin, xmax, ymax = dataframe.iloc[0][['xmin', 'ymin', 'xmax', 'ymax']]
	xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
	cropped_image = filtered_img[ymin:ymax, xmin:xmax] #필터 적용 후 자름
	cropped_image = cv2.resize(cropped_image, (150,200)) #이미지 리사이징
	#cv2.imwrite("cropped_password.jpg", cropped_image)
	#type(cropped_image)

	return cropped_image


def preprocess_evaluating_image(img_array):
    img = pilimg.fromarray(np.uint8(img_array))
    pix = np.array(img)
    
    # 이미지를 잘라내기 위해 미리 지정한 좌표
    rec_points = np.array([[0, 0], [150, 200]])
    
    cropped_pix = pix[rec_points[0, 1]:rec_points[1, 1], rec_points[0, 0]:rec_points[1, 0]]

    if cropped_pix is None or cropped_pix.size == 0:
        print(f"Error: Failed to crop image {img_path}")
        return None
    
    resized_pix = cv2.resize(cropped_pix, (150, 200))

    keypad_imgs = []
    keypad_imgs.append(resized_pix[4*40:(4+1)*40, 1*50:(1+1)*50])  # 0번 키패드
    for y in range(1, 4):
        for x in range(3):
            keypad_imgs.append(resized_pix[y*40:(y+1)*40, x*50:(x+1)*50])

    keypad_imgs = np.array(keypad_imgs)
    data_x = np.expand_dims(keypad_imgs, axis=-1)
    data_x = (data_x / 255.0).astype(np.float32)
    
    return data_x

def final_image(image):
	preprocessed_img = preprocess_evaluating_image(cropped_image)
	preprocessed_img = np.expand_dims(preprocessed_img, axis=0) 
	return preprocessed_img




