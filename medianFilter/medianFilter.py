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

def medianFilterBase(im, win_half):
    '''
    ### Docs: 最基本的中值滤波
    ### Args:
        - im: H*W, numpy.array, float, 单通道图像数据
        - win_half: int, 滤波窗口半径
    ### Returns:
        - filtered: 与 im 相同格式
    ### Examples:
    '''

    im_h, im_w = im.shape

    im_pad = padding(im, win_half)

    mid = ((win_half*2+1) * (win_half*2+1)) // 2

    filtered = np.zeros((im_h, im_w), im.dtype)
    for r in range(win_half, im_h+win_half):
        blk_top = r-win_half
        blk_bottom = r+win_half+1
        for c in range(win_half, im_w+win_half):
            blk_left = c-win_half
            blk_right = c+win_half+1
            blk = im_pad[blk_top:blk_bottom, blk_left:blk_right].flatten()
            blk = np.sort(blk)
            filtered[r-win_half, c-win_half] = blk[mid]
    return filtered

if __name__ == '__main__':

    from PIL import Image
    import matplotlib.pyplot as plt

    img_path = '../src/lena.jpg'
    im = Image.open(img_path)
    im = np.array(im, dtype=float)

    median_base = np.zeros(im.shape, im.dtype)

    win_half = 3

    median_base[:,:,0] = medianFilterBase(im[:,:,0], win_half)
    median_base[:,:,1] = medianFilterBase(im[:,:,1], win_half)
    median_base[:,:,2] = medianFilterBase(im[:,:,2], win_half)

    plt.figure()
    im_base = Image.fromarray(median_base.astype('uint8'))
    plt.imshow(im_base)
    plt.axis('off')
    plt.show()

