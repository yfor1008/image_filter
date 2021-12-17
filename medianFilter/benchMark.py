#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : benchMark.py
# @Author : xxxx
# @Mail   : xxxx@mail.com
# @Date   : 2021/12/17
# @Docs   : 中值滤波算法耗时测试
'''

from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
from medianFilter import medianFilterBase, medianFilterFast, medianFilterConstant

if __name__ == '__main__':

    img_path = '../src/lena.jpg'
    im = Image.open(img_path)
    im = np.array(im, dtype=float)

    filter_size = [2,4,8,16,32,64]
    time_base = np.zeros((len(filter_size,)))
    time_fast = np.zeros((len(filter_size,)))
    time_constant = np.zeros((len(filter_size,)))
    median_base = np.zeros(im.shape, im.dtype)
    median_fast = np.zeros(im.shape, im.dtype)
    median_constant = np.zeros(im.shape, im.dtype)
    for idx, fs in enumerate(filter_size):
        time_start = time.perf_counter()
        median_base[:,:,0] = medianFilterBase(im[:,:,0], fs)
        median_base[:,:,1] = medianFilterBase(im[:,:,1], fs)
        median_base[:,:,2] = medianFilterBase(im[:,:,2], fs)
        time_end = time.perf_counter()
        time_base[idx] = time_end - time_start
        
        time_start = time.perf_counter()
        median_fast[:,:,0] = medianFilterFast(im[:,:,0], fs)
        median_fast[:,:,1] = medianFilterFast(im[:,:,1], fs)
        median_fast[:,:,2] = medianFilterFast(im[:,:,2], fs)
        time_end = time.perf_counter()
        time_fast[idx] = time_end - time_start

        time_start = time.perf_counter()
        median_constant[:,:,0] = medianFilterConstant(im[:,:,0], fs)
        median_constant[:,:,1] = medianFilterConstant(im[:,:,1], fs)
        median_constant[:,:,2] = medianFilterConstant(im[:,:,2], fs)
        time_end = time.perf_counter()
        time_constant[idx] = time_end - time_start

    plt.plot(filter_size, time_base)
    plt.plot(filter_size, time_fast)
    plt.plot(filter_size, time_constant)
    plt.title('Time Consuming', fontsize=20)
    plt.xlabel('filter window radius', fontsize=14)
    plt.ylabel('time(s)', fontsize=14)
    plt.legend(['base', 'fast', 'constant'])
    plt.savefig('../src/median_filter_time_cmp.png')
    plt.show()
