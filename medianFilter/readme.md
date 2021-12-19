# Median Filter

## 原理

计算图像每个像素周围邻域所有像素的中值, 可以通过排序获取中值.

根据原理直接进行计算时, 算法耗时随邻域窗口大小增加而增加. 目前有较多改进.

## 改进

主要有以下几种改进:

1. 使用直方图代替排序, 提高算法执行效率, 参考: [A Fast Two-Dimensional Median Filtering Algorithm](https://ieeexplore.ieee.org/document/1163188);
2. 利用空间换时间, 进一步提高执行效率, 参考: [Median Filter in Constant Time](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.103.3173);

对于上述算法介绍也可以参看: [任意半径中值滤波（扩展至百分比滤波器）O(1)时间复杂度算法的原理、实现及效果](https://www.cnblogs.com/Imageshop/archive/2013/04/26/3045672.html)

上述算法耗时对比如下:

![median_filter_time_cmp](https://gitee.com/yfor1008/pictures/raw/master/median_filter_time_cmp.png)

