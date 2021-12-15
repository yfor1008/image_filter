#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : meanFilter.py
# @Author : xxxx
# @Mail   : xxxx@mail.com
# @Date   : 2021/12/11
# @Docs   : 均值滤波及其改进
'''

import numpy as np
import sys
if '../imPadding/' not in sys.path:
    sys.path.append('../imPadding')
from imPadding import padding

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

    im_pad = padding(im, win_half)

    filtered = np.zeros((im_h, im_w), im.dtype)
    for r in range(win_half, im_h+win_half):
        blk_top = r-win_half
        blk_bottom = r+win_half+1
        for c in range(win_half, im_w+win_half):
            blk_left = c-win_half
            blk_right = c+win_half+1
            blk = im_pad[blk_top:blk_bottom, blk_left:blk_right]
            filtered[r-win_half, c-win_half] = np.mean(blk)
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
    im_pad[1:, 1:] = padding(im, win_half)
    im_pad0 = np.zeros((im_h+2*win_half+1, im_w+2*win_half+1), im.dtype)
    im_pad0[1:, 1:] = padding(np.ones((im_h, im_w)), win_half)

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

    img_path = '../src/lena.jpg'
    im = Image.open(img_path)
    im = np.array(im, dtype=float)

    mean_base = np.zeros(im.shape, im.dtype)
    mean_sat = np.zeros(im.shape, im.dtype)

    win_half = 3

    mean_base[:,:,0] = meanFilterBase(im[:,:,0], win_half)
    mean_base[:,:,1] = meanFilterBase(im[:,:,1], win_half)
    mean_base[:,:,2] = meanFilterBase(im[:,:,2], win_half)

    mean_sat[:,:,0] = meanFilterSat(im[:,:,0], win_half)
    mean_sat[:,:,1] = meanFilterSat(im[:,:,1], win_half)
    mean_sat[:,:,2] = meanFilterSat(im[:,:,2], win_half)

    assert (mean_base == mean_sat).all()

    plt.figure()
    im_base = Image.fromarray(mean_base.astype('uint8'))
    plt.imshow(im_base)
    plt.axis('off')
    plt.show()

    plt.figure()
    im_sat = Image.fromarray(mean_sat.astype('uint8'))
    plt.imshow(im_sat)
    plt.axis('off')
    plt.show()
