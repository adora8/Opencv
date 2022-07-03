# Opencv剩余的课程

## 	图像特征-harris角点检测

### 			基本原理

​						对于图像I(x,y)，当在点(x,y)处平移(△x,△y)后的自相似性：

​						![image-20220626211126619](C:\Users\2021级智科21-1班王诗琪\AppData\Roaming\Typora\typora-user-images\image-20220626211126619.png)

​						W(x,y)是以点(x.y)为中心的窗口，可以是常数或高斯加权函数

​						基于泰勒展开，对图像I(x,y)在平移(△x,△y)后进行一阶近似：

​						![image-20220627101601947](C:\Users\2021级智科21-1班王诗琪\AppData\Roaming\Typora\typora-user-images\image-20220627101601947.png)

​						近似可得：					![image-20220627102235204](C:\Users\2021级智科21-1班王诗琪\AppData\Roaming\Typora\typora-user-images\image-20220627102235204.png)

​						其中M

![image-20220627102453262](C:\Users\2021级智科21-1班王诗琪\AppData\Roaming\Typora\typora-user-images\image-20220627102453262.png)

​						化简可得：

![image-20220627102946459](C:\Users\2021级智科21-1班王诗琪\AppData\Roaming\Typora\typora-user-images\image-20220627102946459.png)

![image-20220627103126924](C:\Users\2021级智科21-1班王诗琪\AppData\Roaming\Typora\typora-user-images\image-20220627103126924.png)

​						二次项函数本质上就是一个椭圆函数，椭圆方程为：

![image-20220627103923199](C:\Users\2021级智科21-1班王诗琪\AppData\Roaming\Typora\typora-user-images\image-20220627103923199.png)

​						边界：一个特征值大，另一个特征值小

​						平面：两个特征值都小，且近似相等

​						角点：两个特征值都大，且近似相等

![image-20220627105318550](C:\Users\2021级智科21-1班王诗琪\AppData\Roaming\Typora\typora-user-images\image-20220627105318550.png)

​						R>0角点；R≈0平面；R<0边界