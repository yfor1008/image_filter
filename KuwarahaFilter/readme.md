# Kuwaraha 滤波

`Kuwahara filter`是由 `Kuwahara`等人提出的, 因而以其名字命名. 最开始是用于心血管系统的RI血管造影图像去噪, 原始文献可以查看: [Processing of RI-Angiocardiographic Images](https://link.springer.com/chapter/10.1007/978-1-4684-0769-3_13), 其优点是平滑时可以保留图像的边缘.

## 基本原理

`Kuwahara filter`属于`value-and-criterion`结构的滤波器, 由2部分构成:

- 值(value): 均值
- 评价标准(criterion): 方差

`Kuwahara filter`将当前滤波区域分成4个子区域, 左上, 右上, 右下, 左下, 计算每个子区域的`value`和`criterion`, 根据`criterion`来选择`value`. 

对于滤波半径`r`的滤波器, 其子区域由`(r+1)*(r+1)`的正方形构成, 中心像素是重叠的, 如下图所示为 3*3 滤波器的模版.

![kuwahara_filter_3x3](https://gitee.com/yfor1008/pictures/raw/master/kuwahara_filter_3x3.png)

选择`criterion(方差)`最小的子区域, 以其`value(均值)`作为该区域滤波后的值.

## 实现效果

如下图所示为[维基百科](https://en.wikipedia.org/wiki/Kuwahara_filter)上的一个示例, 图中沙滩被平滑, 但图像的边缘细节没有被平滑.

![Kuwa_iwias3](https://gitee.com/yfor1008/pictures/raw/master/Kuwa_iwias3.jpg)

## 有趣应用

`Kuwahara filter`应用最多的要数艺术图像及摄影领域了, `Kuwahara filter`可以去除图像的纹理而增强图像的边缘, 如下[维基百科](https://en.wikipedia.org/wiki/Kuwahara_filter)示例:

![Kuwahara_creates_artistic_photo](https://gitee.com/yfor1008/pictures/raw/master/Kuwahara_creates_artistic_photo.jpg)

![Lion_waiting_in_Namibia_-_Kuwahara_Filter](https://gitee.com/yfor1008/pictures/raw/master/Lion_waiting_in_Namibia_-_Kuwahara_Filter.jpg)

## 一些改进

`Kuwahara filter`也是存在一些问题的, 如下所示, 滤波后在图像中产生块状伪影, 破坏了图像的平滑度, 对人的感官产生了负面影响.

![Block_artifact_kuwahara](https://gitee.com/yfor1008/pictures/raw/master/Block_artifact_kuwahara.jpg)

### 使用不同的形状

容易想到的方法是使用不同的形状, 如下所示:

[Tomita-Tsuji filter](http://link.springer.com/10.1007/978-1-4471-6684-9_17):

![Tomita-Tsuji](https://gitee.com/yfor1008/pictures/raw/master/Tomita-Tsuji.png)

[Nagao-Matsuyama filter](http://link.springer.com/10.1007/978-1-4471-6684-9_17):

![Nagao-Matsuyama](https://gitee.com/yfor1008/pictures/raw/master/Nagao-Matsuyama.png)

### 去伪影

如使用子区域均值加权, 各向异性等, 详见: [Image and video abstraction by multi-scale anisotropic Kuwahara filtering](http://dl.acm.org/citation.cfm?doid=2024676.2024686), 如下所示:

![kuwahara_modify](https://gitee.com/yfor1008/pictures/raw/master/kuwahara_modify.png)

![kuwaraha_cmp](https://gitee.com/yfor1008/pictures/raw/master/kuwaraha_cmp.png)

### 加速

`Kuwahara filter`存在的另外一个比较严重的问题就是速度较慢, 因而需要进行优化加速. 可以查看这篇文章: [Free-size accelerated Kuwahara filter](https://link.springer.com/10.1007/s11554-021-01081-3).

参考:

1. [Kuwahara滤波](https://blog.csdn.net/lz0499/article/details/54646952)
2. [SSE图像算法优化系列二十三: 基于value-and-criterion structure 系列滤波器（如Kuwahara，MLV，MCV滤波器）的优化](https://www.cnblogs.com/Imageshop/p/9789754.html)
3. [https://en.wikipedia.org/wiki/Kuwahara_filter](https://en.wikipedia.org/wiki/Kuwahara_filter)

