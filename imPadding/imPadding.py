#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : imPadding.py
# @Author : xxxx
# @Mail   : xxxx@mail.com
# @Date   : 2021/12/15
# @Docs   : 图像填充处理
'''

import numpy as np

def padding(im, radius):
    '''
    ### Docs: 进行填充处理, 使用最近邻数据填充
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - radius: int, 滤波窗口半径
    ### Returns:
        - im_pad: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    im_pad = np.zeros((im_h+2*radius, im_w+2*radius), im.dtype)
    im_pad[radius:im_h+radius, radius:im_w+radius] = im
    for idx in range(radius):
        im_pad[radius:im_h+radius, idx] = im[:, 0]
        im_pad[radius:im_h+radius, idx+im_w+radius] = im[:, -1]
    for idx in range(radius):
        im_pad[idx, :] = im_pad[radius, :]
        im_pad[idx+im_h+radius, :] = im_pad[im_h+radius-1, :]
    return im_pad

if __name__ == '__main__':

    im = np.array([[1,2],[3,4]])
    im_pad = padding(im, 1)

    print(im)
    print(im_pad)