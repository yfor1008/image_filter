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
from meanFilter import meanFilterBase, meanFilterSat, meanFilterSWF, meanFilterSNN, meanFilterKuwahara

def time_test():
    '''耗时测试'''
    img_path = '../src/lena.jpg'
    im = Image.open(img_path)
    im = np.array(im, dtype=float)

    filter_size = [1,2,4,8,16]

    time_base = np.zeros((len(filter_size,)))
    mean_base = np.zeros(im.shape, im.dtype)
    for idx, fs in enumerate(filter_size):
        time_start = time.perf_counter()
        mean_base[:,:,0] = meanFilterBase(im[:,:,0], fs)
        mean_base[:,:,1] = meanFilterBase(im[:,:,1], fs)
        mean_base[:,:,2] = meanFilterBase(im[:,:,2], fs)
        time_end = time.perf_counter()
        time_base[idx] = time_end - time_start

    time_sat = np.zeros((len(filter_size,)))
    mean_sat = np.zeros(im.shape, im.dtype)
    for idx, fs in enumerate(filter_size):
        time_start = time.perf_counter()
        mean_sat[:,:,0] = meanFilterSat(im[:,:,0], fs)
        mean_sat[:,:,1] = meanFilterSat(im[:,:,1], fs)
        mean_sat[:,:,2] = meanFilterSat(im[:,:,2], fs)
        time_end = time.perf_counter()
        time_sat[idx] = time_end - time_start

    time_swf = np.zeros((len(filter_size,)))
    mean_swf = np.zeros(im.shape, im.dtype)
    for idx, fs in enumerate(filter_size):
        time_start = time.perf_counter()
        mean_swf[:,:,0] = meanFilterSWF(im[:,:,0], fs)
        mean_swf[:,:,1] = meanFilterSWF(im[:,:,1], fs)
        mean_swf[:,:,2] = meanFilterSWF(im[:,:,2], fs)
        time_end = time.perf_counter()
        time_swf[idx] = time_end - time_start

    # time_snn = np.zeros((len(filter_size,)))
    # mean_snn = np.zeros(im.shape, im.dtype)
    # for idx, fs in enumerate(filter_size):
    #     time_start = time.perf_counter()
    #     mean_snn[:,:,0] = meanFilterSNN(im[:,:,0], fs)
    #     mean_snn[:,:,1] = meanFilterSNN(im[:,:,1], fs)
    #     mean_snn[:,:,2] = meanFilterSNN(im[:,:,2], fs)
    #     time_end = time.perf_counter()
    #     time_snn[idx] = time_end - time_start

    time_kuwahara = np.zeros((len(filter_size,)))
    mean_kuwahara = np.zeros(im.shape, im.dtype)
    for idx, fs in enumerate(filter_size):
        time_start = time.perf_counter()
        mean_kuwahara[:,:,0] = meanFilterKuwahara(im[:,:,0], fs)
        mean_kuwahara[:,:,1] = meanFilterKuwahara(im[:,:,1], fs)
        mean_kuwahara[:,:,2] = meanFilterKuwahara(im[:,:,2], fs)
        time_end = time.perf_counter()
        time_kuwahara[idx] = time_end - time_start

    plt.plot(filter_size, time_base)
    plt.plot(filter_size, time_sat)
    plt.plot(filter_size, time_swf)
    # plt.plot(filter_size, time_snn)
    plt.plot(filter_size, time_kuwahara)
    plt.title('Time Consuming', fontsize=20)
    plt.xlabel('filter window radius', fontsize=14)
    plt.ylabel('time(s)', fontsize=14)
    # plt.legend(['base', 'sat', 'swf', 'snn', 'kuwahara'])
    plt.legend(['base', 'sat', 'swf', 'kuwahara'])
    plt.savefig('../src/mean_filter_time_cmp.png')
    plt.show()

def effect_test():
    '''效果测试'''

    img_path = '../src/lena.jpg'
    im = Image.open(img_path).resize((256,256))
    im = np.array(im, dtype=float)

    radius = 3
    run_times = 3
    
    mean_base = Image.open(img_path).resize((256,256))
    mean_base = np.array(mean_base, dtype=float)
    for i in range(run_times):
        mean_base[:,:,0] = meanFilterBase(mean_base[:,:,0], radius)
        mean_base[:,:,1] = meanFilterBase(mean_base[:,:,1], radius)
        mean_base[:,:,2] = meanFilterBase(mean_base[:,:,2], radius)

    mean_sat = Image.open(img_path).resize((256,256))
    mean_sat = np.array(mean_sat, dtype=float)
    for i in range(run_times):
        mean_sat[:,:,0] = meanFilterSat(mean_sat[:,:,0], radius)
        mean_sat[:,:,1] = meanFilterSat(mean_sat[:,:,1], radius)
        mean_sat[:,:,2] = meanFilterSat(mean_sat[:,:,2], radius)

    mean_swf = Image.open(img_path).resize((256,256))
    mean_swf = np.array(mean_swf, dtype=float)
    for i in range(run_times):
        mean_swf[:,:,0] = meanFilterSWF(mean_swf[:,:,0], radius)
        mean_swf[:,:,1] = meanFilterSWF(mean_swf[:,:,1], radius)
        mean_swf[:,:,2] = meanFilterSWF(mean_swf[:,:,2], radius)

    mean_snn = Image.open(img_path).resize((256,256))
    mean_snn = np.array(mean_snn, dtype=float)
    for i in range(run_times):
        mean_snn[:,:,0] = meanFilterSNN(mean_snn[:,:,0], radius)
        mean_snn[:,:,1] = meanFilterSNN(mean_snn[:,:,1], radius)
        mean_snn[:,:,2] = meanFilterSNN(mean_snn[:,:,2], radius)

    mean_kuwahara = Image.open(img_path).resize((256,256))
    mean_kuwahara = np.array(mean_kuwahara, dtype=float)
    for i in range(run_times):
        mean_kuwahara[:,:,0] = meanFilterKuwahara(mean_kuwahara[:,:,0], radius)
        mean_kuwahara[:,:,1] = meanFilterKuwahara(mean_kuwahara[:,:,1], radius)
        mean_kuwahara[:,:,2] = meanFilterKuwahara(mean_kuwahara[:,:,2], radius)

    im_pair1 = np.concatenate((im, mean_base, mean_sat), axis=1)
    im_pair2 = np.concatenate((mean_swf, mean_snn, mean_kuwahara), axis=1)
    im_pair = np.concatenate((im_pair1, im_pair2), axis=0)
    im_pair = Image.fromarray(im_pair.astype('uint8'))
    im_pair.save('../src/original-base-sat-swf-snn-kuwahara.png')

    plt.imshow(im_pair)
    plt.title('effect of numerical filter:\noriginal, base, sat, swf, snn, kuwahara', fontsize=14)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':

    time_test()
    # effect_test()
