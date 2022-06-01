# README

## 任务

编写一个图像分类系统，能够对输入图像进行类别预测。具体的说，利用数据库的2250张训练样本进行训练；对测试集中的2235张样本进行预测。



## 数据库说明

**scene_categories数据集包含**15个类别（文件夹名就是类别名），每个类中编号前150号的样本作为训练样本，15个类一共2250张训练样本；剩下的样本构成测试集合。

数据集下载地址：https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177



## 算法整体流程

### 引用库：

```python
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import math
import pickle
```

### 定义分类器类：

```python
class ImgClassifier
```

### 初始化：

```python
def __init__(self, 
             srcFilePath, # 数据库路径
             classNum,  # 数据分类个数
             n_clusters)  # kMeans分类的簇数
```

### 算法流程：

```python
def run(self)  # 调用run开始分类
```

#### 1. 读入数据集

```python
def initDataset(self)
```

使用cv2.imread来读取图片，并将所有图片转化为灰度图。

#### 2. 初始化sklearn的kMeans聚类器，传入初始化的n_clusters值

#### 3. 计算训练数据集的SIFT值

```python
temp = self.computeSIFT(self.trainDataset, stepSize=10)
```

此时的stepSize值代表在特征提取时，在原图像的横纵方向上取像素的步长，这么做是为了减小程序运行压力。在computeSIFT中调用self.SIFT逐个计算训练数据集中每一项图片的SIFT描述符，使用openCV的cv2.xfeatures2d进行特征提取（需要指定版本的openCV）。计算完成后会将结果展开成多个128维的向量形式的矩阵。返回值：SIFT的描述符

#### 4. 计算每幅图像的直方图、金字塔匹配

```python
train_histogram = self.getHistogramSPM(2, self.trainDataset, kMeansClf)
test_histogram = self.getHistogramSPM(2, self.testDataset, kMeansClf)
```

对原始训练数据集和测试数据集进行计算。在计算时，对每幅图片调用

```python
def getImageFeaturesSPM(self, Level, img, kMeans)
```

分析特征和金字塔匹配：对于给定的Level值，分析从0到level的情况，每层level下的步长为H/(2^level)、W/(2^level)，即要把原始图像等切分成2^(2*level)个子图像，逐一调用self.SIFT进行特征提取，再使用kMeans聚类器进程预测，并把预测结果展开成n_clusters维的向量。程序中给定的level值为2，即一共需要计算1+4+16=21次特征，得到21\*60的直方图矩阵。在程序最后，将直方图矩阵降维并标准化，返回(1\*1260)向量。

#### 5. 使用SVM预测结果

调用sklearn.svm中的LinearSVC进行测试集预测。对于LinearSVC中的正则化参数c，越小越有利于数据的正规化。程序中选用了c的值为0.001，将训练数据的直方图和训练数据label作为输入进行训练，并用测试数据的直方图矩阵进行测试，得到预测结果。

#### 6. 程序运行展示

```
Loading dataset:
Done.
[ WARN:0@3.210] DEPRECATED: cv.xfeatures2d.SIFT_create() is deprecated due SIFT tranfer to the main repository. https://github.com/opencv/opencv/issues/16736
Building vocabulary with kMeans...
Computing the SIFT...
2250/2250
Fitting classifier...
Finished.
Building histogram and processing Spatial Pyramid Matching...
Finished.
Train with SVM...
Accuracy: 73.33333333333333 %

Process finished with exit code 0

```

#### 7. 混淆矩阵

```python
[[ 35   0   2   4  11   0   0   0   0   0   0   1   0   8   5]
 [  0  89   0   0   0   0   0   0   0   0   0   0   0   1   1]
 [  1  23  75   3   4   1   0   0   3   2   1   0   1  16  31]
 [  6   0   2  40   5   0   0   0   0   0   0   1   0   3   3]
 [ 25   2   4  13  72   0   0   0   0   1   0   0   0   8  14]
 [  1   0   0   0   0 162   0  20   2   4  21   0   0   0   0]
 [  0   0   0   0   0   0 170   0   1   6   0   1   0   0   0]
 [  0   0   0   0   0  11   0  83   2   4   5   3   0   0   2]
 [  0   0   1   0   1   3   1   1 126   1   3   7  14   0   0]
 [  0   0   0   0   1  10   6   6   1 176  10   4  10   0   0]
 [  0   2   1   0   0  45  15  16   0  19 145  11   2   0   4]
 [  1   0   0   0   0   2   4   3   8   4   1 116   3   0   0]
 [  0   0   2   0   0   2   5   1  14   8   0   6 168   0   0]
 [  2   1   0   1   1   0   0   0   0   0   0   0   0  59   1]
 [  0   4   5  10  15   0   1   0   3   0   0   1   2   1 123]]

```

