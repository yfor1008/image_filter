#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : benchMark.py
# @Author : xxxx
# @Mail   : xxxx@mail.com
# @Date   : 2021/12/15
# @Docs   : 均值滤波算法耗时测试
'''

from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
from meanFilter import meanFilterBase, meanFilterSat

if __name__ == '__main__':

    img_path = '../src/lena.jpg'
    im = Image.open(img_path)
    im = np.array(im, dtype=float)

    filter_size = [2,4,8,16,32,64]
    time_base = np.zeros((len(filter_size,)))
    time_sat = np.zeros((len(filter_size,)))
    mean_base = np.zeros(im.shape, im.dtype)
    mean_sat = np.zeros(im.shape, im.dtype)
    for idx, fs in enumerate(filter_size):
        time_start = time.perf_counter()
        mean_base[:,:,0] = meanFilterBase(im[:,:,0], fs)
        mean_base[:,:,1] = meanFilterBase(im[:,:,1], fs)
        mean_base[:,:,2] = meanFilterBase(im[:,:,2], fs)
        time_end = time.perf_counter()
        time_base[idx] = time_end - time_start
        
        time_start = time.perf_counter()
        mean_sat[:,:,0] = meanFilterSat(im[:,:,0], fs)
        mean_sat[:,:,1] = meanFilterSat(im[:,:,1], fs)
        mean_sat[:,:,2] = meanFilterSat(im[:,:,2], fs)
        time_end = time.perf_counter()
        time_sat[idx] = time_end - time_start

    plt.plot(filter_size, time_base)
    plt.plot(filter_size, time_sat)
    plt.title('Time Consuming', fontsize=20)
    plt.xlabel('filter window radius', fontsize=14)
    plt.ylabel('time(s)', fontsize=14)
    plt.legend(['base', 'integral'])
    plt.savefig('../src/mean_filter_time_cmp.png')
    plt.show()
