#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : meanFilter.py
# @Author : xxxx
# @Mail   : xxxx@mail.com
# @Date   : 2021/12/11
# @Docs   : 中值滤波及其改进
'''

import numpy as np

def meanFilterBase(im, win_half):
    '''
    ### Docs: 最基本的均值滤波
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - win_half: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    filtered = np.zeros((im_h, im_w), im.dtype)
    for r in range(im_h):
        blk_top = max(0, r-win_half)
        blk_bottom = min(im_h, r+win_half+1)
        for c in range(im_w):
            blk_left = max(0, c-win_half)
            blk_right = min(im_w, c+win_half+1)
            blk = im[blk_top:blk_bottom, blk_left:blk_right]
            filtered[r, c] = np.mean(blk)
    return filtered

def meanFilterSat(im, win_half):
    '''
    ### Docs: 使用积分图加速均值滤波
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - win_half: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    # 向外扩充, 前面左边多增加1行1列, 方便计算
    im_pad = np.zeros((im_h+2*win_half+1, im_w+2*win_half+1), im.dtype)
    im_pad[win_half+1:im_h+win_half+1, win_half+1:im_w+win_half+1] = im
    im_pad0 = np.zeros((im_h+2*win_half+1, im_w+2*win_half+1), im.dtype)
    im_pad0[win_half+1:im_h+win_half+1, win_half+1:im_w+win_half+1] = np.ones((im_h, im_w))

    # 计算积分图, nums用于记录每个block像素个数
    sat = np.cumsum(im_pad, axis=0)
    sat = np.cumsum(sat, axis=1)
    nums = np.cumsum(im_pad0, axis=0)
    nums = np.cumsum(nums, axis=1)

    # 滤波
    filtered = np.zeros((im_h, im_w), im.dtype)
    blk_nums = np.zeros((im_h, im_w), im.dtype)
    for r in range(win_half+1, im_h+win_half+1):
        bottom_r = r+win_half
        top_r = r-win_half-1
        for c in range(win_half+1, im_w+win_half+1):
            right_c = c+win_half
            left_c = c-win_half-1
            blk = sat[bottom_r, right_c] - sat[bottom_r, left_c] - sat[top_r, right_c] + sat[top_r, left_c]
            blk_num = nums[bottom_r, right_c] - nums[bottom_r, left_c] - nums[top_r, right_c] + nums[top_r, left_c]
            filtered[r-win_half-1, c-win_half-1] = blk
            blk_nums[r-win_half-1, c-win_half-1] = blk_num
    filtered = filtered / blk_nums
    return filtered

if __name__ == '__main__':

    from PIL import Image
    import matplotlib.pyplot as plt
    import time

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

    # im_filtered = Image.fromarray(meanSat.astype('uint8'))
    # plt.imshow(im_filtered)
    # plt.axis('off')
    # plt.show()

    plt.plot(filter_size, time_base)
    plt.plot(filter_size, time_sat)
    plt.title('Time Consuming', fontsize=20)
    plt.xlabel('filter window radius', fontsize=14)
    plt.ylabel('time(s)', fontsize=14)
    plt.legend(['base', 'integral'])
    plt.savefig('../src/mean_filter_time_cmp.png')
    plt.show()
