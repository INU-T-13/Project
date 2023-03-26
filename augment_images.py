#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import numpy as np
import json
import cv2

def augment_images(dataname, file_list, cut, shifts, rotates, img_folder="./filtered_images", json_folder="./221204"):
    data_x = []
    data_y = []

    for name in dataname:
        # Test 데이터 제외
        if name in [dataname[i] for i in np.where(cut)[0]]:
            continue
        try:
            a = Image.open(f"{img_folder}/{name}.jpg")
        except:
            a = Image.open(f"{img_folder}/{name}.png")
        pix = np.array(a)
        with open(f"{json_folder}/{name}.json", "r") as json_file:
            a_json = json.load(json_file)

        where_is_rec = None
        for i_sh in range(len(a_json["shapes"])):
            if a_json["shapes"][i_sh]["shape_type"] == "rectangle":
                where_is_rec = i_sh
                break;
        rec = np.array(a_json["shapes"][where_is_rec]["points"])
        rec = rec.astype(int)

        for shift in shifts:
            dy, dx = shift
            for rotate in rotates:
                nums = []
                new_pix = pix[rec[0, 1]+dy:rec[1, 1]+dy, rec[0, 0]+dx:rec[1, 0]+dx]

                resized_pix = cv2.resize(new_pix,(150, 200))

                cx, cy = 25, 20
                M = cv2.getRotationMatrix2D((cx,cy), rotate, 1)
                y = 4
                x = 1
                npad = resized_pix[y*40:(y+1)*40, x*50:(x+1)*50]
                rotated = cv2.warpAffine(npad, M, (50, 40))
                nums.append(rotated)
                for y in range(1, 4):
                    for x in range(3):
                        npad = resized_pix[y*40:(y+1)*40, x*50:(x+1)*50]
                        rotated = cv2.warpAffine(npad, M, (50, 40))
                        nums.append(rotated)
                nums = np.array(nums)

                tgt = [int(c)+1 for c in name] + [0]
                tgt = np.array(tgt)

                data_x.append(nums)
                data_y.append(tgt)

                # 적용 후 nums 변수 삭제
                del nums

    data_x = np.array(data_x)
    data_x = (data_x / 255.0).astype(np.float32)
    data_y = np.array(data_y)

    return data_x, data_y
