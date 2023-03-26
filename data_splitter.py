#!/usr/bin/env python
# coding: utf-8
import numpy as np

def split_train_val(data_x, data_y, original_imgs, val_size): # val_size = validation data의 크기
    n_data = len(data_x)  # 데이터 개수 계산
    cut = np.zeros([n_data], dtype=bool)
    cut[np.random.choice(n_data, val_size, replace=False)] = True  # validation data를 random하게 선택

    train_inputs = data_x[~cut]
    train_labels = data_y[~cut]
    train_orig = original_imgs[~cut]

    val_inputs = data_x[cut]
    val_labels = data_y[cut]
    val_orig = original_imgs[cut]

    return train_inputs, train_labels, train_orig, val_inputs, val_labels, val_orig


