#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : meanFilter.py
# @Author : xxxx
# @Mail   : xxxx@mail.com
# @Date   : 2021/12/11
# @Docs   : 均值滤波及其改进
'''

from matplotlib.pyplot import axis
import numpy as np
import sys
if '../imPadding/' not in sys.path:
    sys.path.append('../imPadding')
from imPadding import padding

def meanFilterBase(im, radius):
    '''
    ### Docs: 最基本的均值滤波
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - radius: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    im_pad = padding(im, radius)

    filtered = np.zeros((im_h, im_w), im.dtype)
    for r in range(radius, im_h+radius):
        blk_top = r-radius
        blk_bottom = r+radius+1
        for c in range(radius, im_w+radius):
            blk_left = c-radius
            blk_right = c+radius+1
            blk = im_pad[blk_top:blk_bottom, blk_left:blk_right]
            filtered[r-radius, c-radius] = np.mean(blk)
    return filtered

def meanFilterSat(im, radius):
    '''
    ### Docs: 使用积分图加速均值滤波
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - radius: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    # 向外扩充, 前面左边多增加1行1列, 方便计算
    im_pad = np.zeros((im_h+2*radius+1, im_w+2*radius+1), im.dtype)
    im_pad[1:, 1:] = padding(im, radius)
    im_pad0 = np.zeros((im_h+2*radius+1, im_w+2*radius+1), im.dtype)
    im_pad0[1:, 1:] = padding(np.ones((im_h, im_w)), radius)

    # 计算积分图, nums用于记录每个block像素个数
    sat = np.cumsum(im_pad, axis=0)
    sat = np.cumsum(sat, axis=1)
    nums = np.cumsum(im_pad0, axis=0)
    nums = np.cumsum(nums, axis=1)

    # 滤波
    filtered = np.zeros((im_h, im_w), im.dtype)
    blk_nums = np.zeros((im_h, im_w), im.dtype)
    for r in range(radius+1, im_h+radius+1):
        bottom_r = r+radius
        top_r = r-radius-1
        for c in range(radius+1, im_w+radius+1):
            right_c = c+radius
            left_c = c-radius-1
            blk = sat[bottom_r, right_c] - sat[bottom_r, left_c] - sat[top_r, right_c] + sat[top_r, left_c]
            blk_num = nums[bottom_r, right_c] - nums[bottom_r, left_c] - nums[top_r, right_c] + nums[top_r, left_c]
            filtered[r-radius-1, c-radius-1] = blk
            blk_nums[r-radius-1, c-radius-1] = blk_num
    filtered = filtered / blk_nums
    return filtered

def meanFilterSWF(im, radius):
    '''
    ### Docs: SWF(Side Window Filter)均值滤波, 使用积分图进行加速
        - "Side Window Filtering", 2019
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - radius: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    # 向外扩充, 前面左边多增加1行1列, 方便计算
    im_pad = np.zeros((im_h+2*radius+1, im_w+2*radius+1), im.dtype)
    im_pad[1:, 1:] = padding(im, radius)

    # 计算积分图
    sat = np.cumsum(im_pad, axis=0)
    sat = np.cumsum(sat, axis=1)

    size_L = size_R = size_U = size_D = (radius*2+1) * (radius+1)
    size_LU = size_LD = size_RD = size_RU = (radius+1) * (radius+1)

    # 滤波
    filtered = np.zeros((im_h, im_w), im.dtype)
    for r in range(radius+1, im_h+radius+1):
        bottom_r = r+radius
        top_r = r-radius-1
        for c in range(radius+1, im_w+radius+1):
            right_c = c+radius
            left_c = c-radius-1
            blk_L = sat[bottom_r, c] - sat[top_r, c] - sat[bottom_r, left_c] + sat[top_r, left_c]
            blk_R = sat[bottom_r, right_c] - sat[top_r, right_c] - sat[bottom_r, c-1] + sat[top_r, c-1]
            blk_U = sat[r, right_c] - sat[top_r, right_c] - sat[r, left_c] + sat[top_r, left_c]
            blk_D = sat[bottom_r, right_c] - sat[r-1, right_c] - sat[bottom_r, left_c] + sat[r-1, left_c]
            blk_LU = sat[r, c] - sat[top_r, c] - sat[r, left_c] + sat[top_r, left_c]
            blk_LD = sat[bottom_r, c] - sat[r-1, c] - sat[bottom_r, left_c] + sat[r-1, left_c]
            blk_RD = sat[bottom_r, right_c] - sat[r-1, right_c] - sat[bottom_r, c-1] + sat[r-1, c-1]
            blk_RU = sat[r, right_c] - sat[top_r, right_c] - sat[r, c-1] + sat[top_r, c-1]

            blk_L = blk_L / size_L
            blk_R = blk_R / size_R
            blk_U = blk_U / size_U
            blk_D = blk_D / size_D
            blk_LU = blk_LU / size_LU
            blk_LD = blk_LD / size_LD
            blk_RD = blk_RD / size_RD
            blk_RU = blk_RU / size_RU
            E = [blk_L, blk_R, blk_U, blk_D, blk_LU, blk_LD, blk_RD, blk_RU]
            E1 = abs(np.array(E) - im_pad[r, c])
            idx = np.argmin(E1)
            
            filtered[r-radius-1, c-radius-1] = E[idx]
    return filtered

def meanFilterSNN(im, radius):
    '''
    ### Docs: SNN(Symmetric Nearest Neighbor)均值滤波
        - "Smooth operator: Smoothing seismic interpretations and attributes", 2007
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - radius: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape
    win_half = 2 * radius * (radius + 1)

    im_pad = padding(im, radius)

    filtered = np.zeros((im_h, im_w), im.dtype)
    for r in range(radius, im_h+radius):
        blk_top = r-radius
        blk_bottom = r+radius+1
        for c in range(radius, im_w+radius):
            blk_left = c-radius
            blk_right = c+radius+1
            blk = im_pad[blk_top:blk_bottom, blk_left:blk_right].flatten()
            # 分成2半, 构成像素对
            blk_T = blk[:win_half]
            blk_B = blk[-1:win_half:-1] # 逆序
            pp = np.concatenate((blk_T, blk_B)).reshape((2, -1)).T
            idx = np.argmin(np.abs(pp - blk[win_half]), axis=1) # 查找与中间像素最相近的像素
            nn = pp[range(win_half), idx]
            filtered[r-radius, c-radius] = np.mean(nn)
    return filtered

def meanFilterKuwahara(im, radius):
    '''
    ### Docs: kuwahara均值滤波, 使用积分图加速
        - "Smooth operator: Smoothing seismic interpretations and attributes", 2007
        - 方差: 转换成一阶及二阶积分图, sum(xi^2)/n - u^2
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - radius: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape
    blk_size = (radius+1) * (radius+1)

    # 向外扩充, 前面左边多增加1行1列, 方便计算
    im_pad = np.zeros((im_h+2*radius+1, im_w+2*radius+1), im.dtype)
    im_pad[1:, 1:] = padding(im, radius)
    im_pad2 = im_pad ** 2

    # 计算积分图
    sat = np.cumsum(im_pad, axis=0)
    sat = np.cumsum(sat, axis=1)
    sat2 = np.cumsum(im_pad2, axis=0)
    sat2 = np.cumsum(sat2, axis=1)

    # 滤波
    filtered = np.zeros((im_h, im_w), im.dtype)
    for r in range(radius+1, im_h+radius+1):
        bottom_r = r+radius
        top_r = r-radius-1
        for c in range(radius+1, im_w+radius+1):
            right_c = c+radius
            left_c = c-radius-1
            blk_LU_std = sat2[r, c] - sat2[top_r, c] - sat2[r, left_c] + sat2[top_r, left_c]
            blk_LU = sat[r, c] - sat[top_r, c] - sat[r, left_c] + sat[top_r, left_c]
            blk_LU_std = blk_LU_std - blk_LU ** 2 / blk_size

            blk_LD_std = sat2[bottom_r, c] - sat2[r-1, c] - sat2[bottom_r, left_c] + sat2[r-1, left_c]
            blk_LD = sat[bottom_r, c] - sat[r-1, c] - sat[bottom_r, left_c] + sat[r-1, left_c]
            blk_LD_std = blk_LD_std - blk_LD ** 2 / blk_size

            blk_RD_std = sat2[bottom_r, right_c] - sat2[r-1, right_c] - sat2[bottom_r, c-1] + sat2[r-1, c-1]
            blk_RD = sat[bottom_r, right_c] - sat[r-1, right_c] - sat[bottom_r, c-1] + sat[r-1, c-1]
            blk_RD_std = blk_RD_std - blk_RD ** 2 / blk_size

            blk_RU_std = sat2[r, right_c] - sat2[top_r, right_c] - sat2[r, c-1] + sat2[top_r, c-1]
            blk_RU = sat[r, right_c] - sat[top_r, right_c] - sat[r, c-1] + sat[top_r, c-1]
            blk_RU_std = blk_RU_std - blk_RU ** 2 / blk_size

            stds = [blk_LU_std, blk_LD_std, blk_RD_std, blk_RU_std]
            means = [blk_LU, blk_LD, blk_RD, blk_RU]
            idx = np.argmin(stds)
            
            filtered[r-radius-1, c-radius-1] = means[idx]
    filtered = filtered / blk_size
    return filtered

if __name__ == '__main__':

    from PIL import Image
    import matplotlib.pyplot as plt

    img_path = '../src/lena.jpg'
    im = Image.open(img_path)
    im = np.array(im, dtype=float)

    radius = 3

    # base
    mean_base = np.zeros(im.shape, im.dtype)
    mean_base[:,:,0] = meanFilterBase(im[:,:,0], radius)
    mean_base[:,:,1] = meanFilterBase(im[:,:,1], radius)
    mean_base[:,:,2] = meanFilterBase(im[:,:,2], radius)

    plt.figure()
    im_base = Image.fromarray(mean_base.astype('uint8'))
    plt.imshow(im_base)
    plt.axis('off')
    plt.show()

    # 积分图
    mean_sat = np.zeros(im.shape, im.dtype)
    mean_sat[:,:,0] = meanFilterSat(im[:,:,0], radius)
    mean_sat[:,:,1] = meanFilterSat(im[:,:,1], radius)
    mean_sat[:,:,2] = meanFilterSat(im[:,:,2], radius)

    plt.figure()
    im_sat = Image.fromarray(mean_sat.astype('uint8'))
    plt.imshow(im_sat)
    plt.axis('off')
    plt.show()

    # 侧窗
    mean_swf = np.zeros(im.shape, im.dtype)
    mean_swf[:,:,0] = meanFilterSWF(im[:,:,0], radius)
    mean_swf[:,:,1] = meanFilterSWF(im[:,:,1], radius)
    mean_swf[:,:,2] = meanFilterSWF(im[:,:,2], radius)

    plt.figure()
    im_swf = Image.fromarray(mean_swf.astype('uint8'))
    plt.imshow(im_swf)
    plt.axis('off')
    plt.show()

    # SNN
    mean_snn = np.zeros(im.shape, im.dtype)
    mean_snn[:,:,0] = meanFilterSNN(im[:,:,0], radius)
    mean_snn[:,:,1] = meanFilterSNN(im[:,:,1], radius)
    mean_snn[:,:,2] = meanFilterSNN(im[:,:,2], radius)

    plt.figure()
    im_snn = Image.fromarray(mean_snn.astype('uint8'))
    plt.imshow(im_snn)
    plt.axis('off')
    plt.show()

    # Kuwahara
    mean_kuwahara = np.zeros(im.shape, im.dtype)
    mean_kuwahara[:,:,0] = meanFilterKuwahara(im[:,:,0], radius)
    mean_kuwahara[:,:,1] = meanFilterKuwahara(im[:,:,1], radius)
    mean_kuwahara[:,:,2] = meanFilterKuwahara(im[:,:,2], radius)

    plt.figure()
    im_kuwahara = Image.fromarray(mean_kuwahara.astype('uint8'))
    plt.imshow(im_kuwahara)
    plt.axis('off')
    plt.show()

