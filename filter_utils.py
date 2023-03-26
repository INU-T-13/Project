#!/usr/bin/env python
# coding: utf-8
#filter_utils.py
# In[1]:


import os
import cv2

def apply_filters(img):
    # 이미지를 grayscale로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 히스토그램 평활화 적용
    equalized = cv2.equalizeHist(blurred)

    # Adaptive Thresholding 적용
    #thresholded = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return equalized

def apply_new_filters(img):
    # 이미지를 grayscale로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 히스토그램 평활화 적용
    equalized = cv2.equalizeHist(blurred)

    # 라플라시안 필터 적용
    laplacian = cv2.Laplacian(equalized, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # 소벨 필터 적용
    sobelx = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=5)
    sobelx_abs = cv2.convertScaleAbs(sobelx)
    sobely_abs = cv2.convertScaleAbs(sobely)
    sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

    # 라플라시안과 소벨 필터를 합침
    combined_filters = cv2.addWeighted(laplacian_abs, 0.5, sobel_combined, 0.5, 0)

    return combined_filters


def process_images(input_folder, output_folder):
    img_files = os.listdir(input_folder)

    for i, file_name in enumerate(img_files):
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path)

        #filtered_img = apply_filters(img)
        filtered_img = apply_new_filters(img)
        

        save_path = os.path.join(output_folder, file_name)

        cv2.imwrite(save_path, filtered_img)
    print("이미지 처리 완료")

def get_filtered_image_names(dir_path):
    file_list = [file_name for file_name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file_name))]
    
    data_names = []
    for file_name in file_list:
        data_name, extension = os.path.splitext(file_name)
        data_names.append(data_name)
    
    return data_names

