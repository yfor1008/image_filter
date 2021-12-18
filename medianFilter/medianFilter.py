#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : medianFilter.py
# @Author : xxxx
# @Mail   : xxxx@mail.com
# @Date   : 2021/12/15
# @Docs   : 中值滤波及其改进
'''

import numpy as np
import sys

if '../imPadding/' not in sys.path:
    sys.path.append('../imPadding')
from imPadding import padding

def medianFilterBase(im, radius):
    '''
    ### Docs: 最基本的中值滤波
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - radius: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    im_pad = padding(im, radius)

    mid = ((radius*2+1) * (radius*2+1)) // 2

    filtered = np.zeros((im_h, im_w), im.dtype)
    for r in range(radius, im_h+radius):
        blk_top = r-radius
        blk_bottom = r+radius+1
        for c in range(radius, im_w+radius):
            blk_left = c-radius
            blk_right = c+radius+1
            blk = im_pad[blk_top:blk_bottom, blk_left:blk_right].flatten()
            blk = np.sort(blk)
            filtered[r-radius, c-radius] = blk[mid]
    return filtered

def getHist(blk):
    '''获取直方图'''
    hist = np.zeros((256, ), dtype='int')
    for val in blk:
        hist[val] += 1
    return hist

def getMedian(hist, thres):
    '''获取中值'''
    nsum = 0
    for idx, val in enumerate(hist):
        nsum += val
        if nsum >= thres:
            break
    return idx

def medianFilterFast(im, radius):
    '''
    ### Docs: 快速中值滤波, "A Fast Two-Dimensional Median Filtering Algorithm", 1979
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - radius: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    im_pad = padding(im, radius)

    win_size = radius*2+1
    mid = (win_size * win_size) // 2

    filtered = np.zeros((im_h, im_w), im.dtype)
    for r in range(radius, im_h+radius):
        blk_top = r-radius
        blk_bottom = r+radius+1

        # 原始图像第0列, 生成直方图
        c = radius
        blk_left = c-radius
        blk_right = c+radius+1
        blk = im_pad[blk_top:blk_bottom, blk_left:blk_right].flatten()
        hist = getHist(blk.astype('uint8'))
        med = getMedian(hist, mid)
        filtered[r-radius, c-radius] = med

        # 第0列以后
        for c in range(radius+1, im_w+radius):
            cl = c-radius-1 # 左边列
            cr = c+radius # 右边列
            for k in range(blk_top, blk_bottom):
                pl = im_pad[k, cl]
                pr = im_pad[k, cr]
                hist[int(pl)] -= 1
                hist[int(pr)] += 1
            med = getMedian(hist, mid)
            filtered[r-radius, c-radius] = med
    return filtered

def medianFilterConstant(im, radius):
    '''
    ### Docs: 时间恒定中值滤波, "Median Filter in Constant Time", 2007
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - radius: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    im_pad = padding(im, radius)

    win_size = radius*2+1
    mid = (win_size * win_size) // 2

    filtered = np.zeros((im_h, im_w), im.dtype)

    # 初始化原始图像第0行所有列的直方图
    hists = np.zeros((256, im_w+radius*2), dtype='int')
    for c in range(im_w+radius*2):
        im_col = im_pad[:win_size, c]
        hist = getHist(im_col.astype('uint8'))
        hists[:,c] = hist

    # 原始图像第0行的中值
    hist = np.sum(hists[:, :win_size], axis=1)
    med = getMedian(hist, mid)
    filtered[0,0] = med
    for c in range(radius+1, im_w+radius):
        pre_c = c - radius - 1
        cur_c = c + radius
        hist -= hists[:, pre_c]
        hist += hists[:, cur_c]
        med = getMedian(hist, mid)
        filtered[0,c-radius] = med

    # 更新直方图
    for r in range(radius+1, im_h+radius):
        # 减去前一行, 加上最后行
        pre_r = r-radius-1
        cur_r = r+radius
        for c in range(im_w+radius*2):
            pre_val = int(im_pad[pre_r, c])
            cur_val = int(im_pad[cur_r, c])
            hists[pre_val, c] -= 1
            hists[cur_val, c] += 1
        
        # 原始图像第0列
        hist = np.sum(hists[:, :win_size], axis=1)
        med = getMedian(hist, mid)
        filtered[r-radius,0] = med
        for c in range(radius+1, im_w+radius):
            pre_c = c - radius - 1
            cur_c = c + radius
            hist -= hists[:, pre_c]
            hist += hists[:, cur_c]
            med = getMedian(hist, mid)
            filtered[r-radius,c-radius] = med
    return filtered

if __name__ == '__main__':

    from PIL import Image
    import matplotlib.pyplot as plt

    img_path = '../src/lena.jpg'
    im = Image.open(img_path)
    im = np.array(im, dtype=float)

    median_base = np.zeros(im.shape, im.dtype)
    median_fast = np.zeros(im.shape, im.dtype)
    median_constant = np.zeros(im.shape, im.dtype)

    radius = 1

    median_base[:,:,0] = medianFilterBase(im[:,:,0], radius)
    median_base[:,:,1] = medianFilterBase(im[:,:,1], radius)
    median_base[:,:,2] = medianFilterBase(im[:,:,2], radius)

    median_fast[:,:,0] = medianFilterFast(im[:,:,0], radius)
    median_fast[:,:,1] = medianFilterFast(im[:,:,1], radius)
    median_fast[:,:,2] = medianFilterFast(im[:,:,2], radius)

    median_constant[:,:,0] = medianFilterConstant(im[:,:,0], radius)
    median_constant[:,:,1] = medianFilterConstant(im[:,:,1], radius)
    median_constant[:,:,2] = medianFilterConstant(im[:,:,2], radius)

    assert (median_fast == median_fast).all()
    assert (median_fast == median_constant).all()

    plt.figure()
    im_base = Image.fromarray(median_base.astype('uint8'))
    plt.imshow(im_base)
    plt.axis('off')
    plt.show()

    plt.figure()
    im_fast = Image.fromarray(median_fast.astype('uint8'))
    plt.imshow(im_fast)
    plt.axis('off')
    plt.show()

    plt.figure()
    im_fast = Image.fromarray(median_constant.astype('uint8'))
    plt.imshow(im_fast)
    plt.axis('off')
    plt.show()
