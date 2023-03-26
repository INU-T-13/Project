#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import scipy.ndimage as ndimage

def preprocess_images(dataname, img_folder, json_folder):
    data_x = []
    original_imgs = []

    for name in dataname:
        img = pilimg.open(f"{img_folder}/{name}.jpg")
        pix = np.array(img)

        with open(f"{json_folder}/{name}.json", "r") as json_file:
            img_json = json.load(json_file)

        rec_points = None
        for shape in img_json["shapes"]:
            if shape["shape_type"] == "rectangle":
                rec_points = np.array(shape["points"]).astype(int)
                break

        if rec_points is None:
            print(f"Error: Rectangle not found in data {name}")
            continue

        original_imgs.append(pix)
        cropped_pix = pix[rec_points[0, 1]:rec_points[1, 1], rec_points[0, 0]:rec_points[1, 0]]

        if cropped_pix is None or cropped_pix.size == 0:
            print(f"Error: Failed to crop image {name}")
            continue

        resized_pix = cv2.resize(cropped_pix, (150, 200))

        keypad_imgs = []
        keypad_imgs.append(resized_pix[4*40:(4+1)*40, 1*50:(1+1)*50])  # 0번 키패드
        for y in range(1, 4):
            for x in range(3):
                keypad_imgs.append(resized_pix[y*40:(y+1)*40, x*50:(x+1)*50])

        keypad_imgs = np.array(keypad_imgs)
        data_x.append(keypad_imgs)

    data_x = np.array(data_x)
    data_x = (data_x / 255.0).astype(np.float32)
    original_imgs = np.array(original_imgs)
    data_x = np.expand_dims(data_x, axis=-1)
    
    return data_x, original_imgs

def generate_labels(dataname):
    data_y = []

    for name in dataname:
        tgt = [int(c) + 1 for c in name] + [0]
        data_y.append(np.array(tgt))

    data_y = np.array(data_y)
    
    return data_y




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