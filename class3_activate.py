# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 22:38:08 2020

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io as io
import skimage.filters as filters
from skimage import filters
from skimage import feature
img=io.imread('data/cloud.jpg')
plt.imshow(img)
img_gray=skimage.color.rgb2gray(img)
plt.imshow(img_gray,plt.cm.gray)
#平滑滤波
img_e=filters.gaussian(img,sigma=5)
plt.imshow(img_e)
#robert 算子滤波
img_e=filters.roberts(img_gray)
plt.imshow(img_e)
#prewitt算子

#水平算子
img_e1=filters.prewitt_h(img_gray)

#垂直算子
img_e2=filters.prewitt_v(img_gray)

#结果(分块绘图)
plt.figure('prewitt',figsize=(8,8))
plt.subplot(121)
plt.imshow(img_e1)
plt.subplot(122)
plt.imshow(img_e2)
plt.show()

#直接调用
img_e3=filters.prewitt(img_gray)
plt.imshow(img_e3)

#sobel算子检测

#水平算子
img_e1=filters.sobel_h(img_gray)

#垂直算子
img_e2=filters.sobel_v(img_gray)

#结果（分块绘图）
plt.figure('sobel',figsize=(8,8))
plt.subplot(121)
plt.imshow(img_e1)
plt.subplot(122)
plt.imshow(img_e2)
plt.show()

#使用canny算子检测
img_e=feature.canny(img_gray,sigma=0.5)
plt.imshow(img_e)

#otsu阈值分割，不同nbins效果
theta=filters.threshold_otsu(img_gray,nbins=256)
img_seg=np.zeros(img_gray.shape)
img_seg[img_gray>theta]=1
#结果展示
plt.imshow(img_seg)






