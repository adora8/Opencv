# 背景建模

## 	帧差法

​		由于场景中目标在运动，目标的影像在不同图像帧中的位置不同

​		该算法对时间上连续的两帧图像进行差分运算

​		结果绝对值超过一定阈值T时可判断为运动目标

​		Dn(x,y)=|fn(x,y)-f(n-1)(x,y)|

​		R'n(x,y)=255,Dn(x,y)>T

​		R'n(x,y)=0,else

​		*问题：容易引入噪音和空洞问题*

## 	混合高斯模型

​		在进行前景检测前，先对背景进行训练，对图像中每个背景采用一个混合高斯模型进行模拟

​		混合高斯模型学习方法：

​			首先初始化每个高斯模型的矩阵参数

​			取视频中T帧数据图像用来训练高斯混合模型，来了第一个像素之后用它当作第一个高斯分布

​			当后面来的像素值时，与前面已有的高斯的均值比较，若该像素点的值与其模型均值差在3倍的方差内，则属于该分布，并对其进行参数更新

​			若下一次来的像素不满足当前高斯分布，用它来创建一个新的高斯分布

​		混合高斯模型测试方法：

​			在测试阶段，对新来像素的值与混合高斯模型中的每一个均值进行比较，若其差值在2倍的方差之间的话，则认为是背景，否则为前景

​		python-opencv实战

```python
import numpy as np
import cv2

#经典的测试视频
cap=cv2.VideoCapture('test.avi')
#形态学操作需要使用
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#创建混合高斯模型用于背景建模
fgbg=cv2.createBackgroundSubtractorMOG2()

while(1):
    ret,frame=cap.read()  #读取每一帧图像
    fgmask=fgbg.apply(frame)
    #形态学开运算去噪点
    fgmask=cv2.morphologyEx(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #寻找视频中的轮廓
  	im,contours,lierarchy=cv2.findContours(fgmask,cv2.RETR_EXTERNAL,
                                        	 cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        #计算各轮廓的周长
        perimeter=cv2.arcLength(c,true)
        if perimeter>188:
            #找到一个直矩形(不会旋转)
            x,y,w,h=cv2.boundingRect(c)
            #画出这个矩形
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0.255.0),2)
   	cv2.imshow('frame',frame)
    cv2.imshow('fgmask',fgmask)
    if k==27:  #27代表退出键
        break

cap.release()
cv2.destroyAllWindows()
```

