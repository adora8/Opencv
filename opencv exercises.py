import cv2
import numpy as np
pie=cv2.imread('D:/360/pie.jpg')
cv2.imshow('pic',pie)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(pie.shape)
pie_gray=cv2.imread('D:/360/pie.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('graypic',pie_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('graypic.png',pie_gray)
kernel=np.ones((3,3),np.uint8)
erosion=cv2.erode(pie,kernel,iterations=1)
cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
dilate=cv2.dilate(erosion,kernel,iterations=1)
cv2.imshow('dilate',dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
dilate=cv2.dilate(pie,kernel,iterations=1)
cv2.imshow('pie_dilate',dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
kernel=np.ones((5,5),np.uint8)
opening=cv2.morphologyEx(pie,cv2.MORPH_OPEN,kernel)
cv2.imshow('opening',opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
closing=cv2.morphologyEx(pie,cv2.MORPH_CLOSE,kernel)
cv2.imshow('closing',closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
pie=cv2.imread('D:/360/pie.jpg')
kernel=np.ones((3,3),np.uint8)
erosion=cv2.erode(pie,kernel,iterations=5)
dilate=cv2.dilate(pie,kernel,iterations=5)
res=np.hstack((pie,erosion,dilate))
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
gradient=cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel)
cv2.imshow('gradient',gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
tophat=cv2.morphologyEx(pie,cv2.MORPH_TOPHAT,kernel)
cv2.imshow('tophat',tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()
blackhat=cv2.morphologyEx(pie,cv2.MORPH_BLACKHAT,kernel)
cv2.imshow('blackhat',blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
compare=np.hstack((tophat,blackhat))
cv2.imshow('compare',compare)
cv2.waitKey(0)
cv2.destroyAllWindows()
img=cv2.imread('D:/360/pie.jpg')
img_noise=img
cv2.imshow('src',img)
rows,cols,chn=img_noise.shape
for i in range(5000):
    x=np.random.randint(0,rows)
    y=np.random.randint(0,cols)
    img_noise[x,y,:]=255
cv2.imshow('noise',img_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()
blur=cv2.blur(img_noise,(3,3))
median=cv2.medianBlur(img_noise,3)
aussian=cv2.GaussianBlur(img_noise,(3,3 ),0)
csv=np.hstack((blur,median,aussian))
cv2.imshow('csv',csv)
cv2.waitKey(0)
cv2.destroyAllWindows()
img=cv2.imread('D:/360Downloads/wpcache/srvsetwp/cut.jpg')#D:/360Downloads/wpcache/360wallpaper.jpg
pie=img[0:200,0:200]
cv2.imshow('cut',pie)
cv2.waitKey(0)
cv2.destroyAllWindows()
b,g,r=cv2.split(img)
img=cv2.merge((b,g,r))
cur_img=img.copy()
cur=cur_img[:,:,0]
cur=cur_img[:,:,1]
cv2.imshow('red',cur)
cv2.waitKey(0)
cv2.destroyAllWindows()
cug=cur_img[:,:,0]
cug=cur_img[:,:,2]
cv2.imshow('green',cug)
cv2.waitKey(0)
cv2.destroyAllWindows()
cub=cur_img[:,:,1]
cub=cur_img[:,:,2]
cv2.imshow('blue',cub)
cv2.waitKey(0)
cv2.destroyAllWindows()
res=np.hstack((cur,cug,cub))
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
top_size,bottom_size,left_size,right_size=(50,50,50,50)
border1=cv2.copyMakeBorder(pie,top_size,bottom_size,left_size,right_size,cv2.BORDER_REPLICATE)
border2=cv2.copyMakeBorder(pie,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT)
border3=cv2.copyMakeBorder(pie,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT_101)
border4=cv2.copyMakeBorder(pie,top_size,bottom_size,left_size,right_size,cv2.BORDER_WRAP)
border5=cv2.copyMakeBorder(pie,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=0)
border=np.hstack((border1,border2,border3,border4,border5))
cv2.imshow('BORDER',border)
cv2.waitKey(0)
cv2.destroyAllWindows()
girl=cv2.imread('D:/360Downloads/wpcache/srvsetwp/2054368.jpg')
cat=cv2.imread('D:/360Downloads/wpcache/srvsetwp/2055085.jpg')
cat2=cat+10
print(cat.shape)
print(girl.shape)
print(cat2[:5,:,0])
print((girl+cat)[:5,:,0])      #越界减255
print(cv2.add(girl,cat)[:5,:,0])      #越界取255
img=cv2.imread('D:/360Downloads/wpcache/srvsetwp/2054368.jpg')
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv_show(img,'src')
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx=cv2.convertScaleAbs(sobelx)
cv_show(sobelx,'sobelx')
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely=cv2.convertScaleAbs(sobely)
cv_show(sobely,'sobely')
sobelxy=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv_show(sobelxy,'sobelxy')
scharrx=cv2.Scharr(img,cv2.CV_64F,1,0)
scharrx=cv2.convertScaleAbs(scharrx)
cv_show(scharrx,'scharrx')
scharry=cv2.Scharr(img,cv2.CV_64F,0,1)
scharry=cv2.convertScaleAbs(scharry)
cv_show(scharry,'scharry')
scharrxy=cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
cv_show(scharrxy,'scharrxy')
laplacian=cv2.Laplacian(img,cv2.CV_64F)
laplacian=cv2.convertScaleAbs(laplacian)
cv_show(laplacian,'laplacian')
img=cv2.imread('D:/360Downloads/wpcache/srvsetwp/cut.jpg')
ret,dst1=cv2.threshold(img,127,255,cv2.THRESH_BINARY) #超过阈值取maxval,否则取0,结果非黑即白
ret,dst2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV) #与上面相反,结果非白即黑
res1=np.hstack((img,dst1,dst2))
cv_show(res1,'res1')
ret,dst3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC) #小于阈值的保持原色，大于阈值的为灰色
res2=np.hstack((img,dst3))
cv_show(res2,'res2')
ret,dst4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO) #小于阈值的为0，大于阈值的保持原色
ret,dst5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV) #小于阈值的保持原色，大于阈值的为0
res3=np.hstack((img,dst4,dst5))
cv_show(res3,'res3')
img=cv2.imread('D:/360Downloads/wpcache/srvsetwp/cut.jpg',cv2.IMREAD_GRAYSCALE)
v1=cv2.Canny(img,80,150)
v2=cv2.Canny(img,50,100)
res=np.hstack((img,v1,v2))
cv_show(res,'res')
img=cv2.imread('D:/360Downloads/wpcache/srvsetwp/cut.jpg')
img_large=cv2.pyrUp(img)
img_small=cv2.pyrDown(img)
cv_show(img,'img')
cv_show(img_large,'img_large')
cv_show(img_small,'img_small')
img_bl=cv2.pyrDown(img_large)
cv2.imshow('img',img)
cv2.imshow('img_bl',img_bl)
img_lb=cv2.pyrUp(img_small)
cv_show(img_lb,'img_lb')
down=cv2.pyrDown(img)
down_up=cv2.pyrUp(down)
img_1=cv2.resize(img,(612,758))
l_1=img_1-down_up
l_1=cv2.convertScaleAbs(l_1)
cv_show(l_1,'l_1')
img=cv2.imread('D:/360Downloads/wpcache/srvsetwp/examples.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv_show(thresh,'thresh')
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #旧版opencv会返回三个值
draw_img=img.copy()
res=cv2.drawContours(draw_img,contours,-1,(0,0,255),2) #-1表示画出所有轮廓,(B,G,R)用于轮廓颜色调节.2用于调节轮廓粗细
cv_show(res,'res')
cnt=contours[1]
area=cv2.contourArea(cnt) #算面积
length=cv2.arcLength(cnt,True) #算周长
print(area)
print(length)
epsilon=0.001*cv2.arcLength(cnt,True) #阈值
approx=cv2.approxPolyDP(cnt,epsilon,True)
draw_img=img.copy()
res=cv2.drawContours(draw_img,[approx],-1,(0,0,255),2)
cv_show(res,'res') #进行轮廓近似
x,y,w,h=cv2.boundingRect(cnt) #(x,y)是矩形左上点的坐标,w,h分别是矩形的宽和高
img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,100,0),2)
cv_show(img,'rectangle')
area=cv2.contourArea(cnt)
rec_area=w*h
extent=float(area)/rec_area #计算轮廓的面积与外接矩形的面积之比
print(extent)
(x,y),radius=cv2.minEnclosingCircle(cnt)
Center=(int(x),int(y))
radius=int(radius)
img=cv2.circle(img,Center,radius,(100,255,0),2)
cv_show(img,'circle')
target=cv2.imread('D:/360Downloads/wpcache/picture_test.jpg')
template=cv2.imread('D:/360Downloads/wpcache/srvsetwp/picture_test_cut.jpg')
theight,twidth=template.shape[:2] #获取模板的高和宽
result=cv2.matchTemplate(target,template,cv2.TM_SQDIFF_NORMED)
cv2.normalize(result,result,0,1,cv2.NORM_MINMAX,-1) #归一化处理
min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(result)
strmin_val=str(min_val)
import matplotlib.pyplot as plt
#cv2.rectangle对应参数的含义(原图,左上点坐标，右下点坐标，BGR颜色,线条的宽度)
cv2.rectangle(target,min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(255,100,0),2)
cv_show(target,'MatchResult--MatchingValue='+strmin_val)
img=cv2.imread('DS:/360Downloads/wpcache/picture_test.jpg',0)
#参数:[输入的图像]，[图像的通道]，掩膜，[使用多少个bin(柱子)],[像素值的范围]
hist=cv2.calcHist([img],[0],None,[256],[0,256]) #计算直方图
#img.ravel()指最终直方图要对数据集进行统计，后面的参数是:256是统计的区间分布，可能还会有一个显示的区间的参数[0.256]
plt.hist(img.ravel(),256)
plt.show()
equ=cv2.equalizeHist(img)
plt.hist(equ.ravel(),256)
plt.show()
import numpy as np
res=np.hstack((img,equ))
cv_show(res,'res')
clahe=cv2.createCLAHE(clipLimit=10,tileGridSize=(8,8))
res_clahe=clahe.apply()
plt.hist(res_clahe.ravel(),256)
plt.show()
res=np.hstack((equ,img,res_clahe))
cv_show(res,'res')
vc=cv2.VideoCapture('D:/360/video.mp4')
if vc.isOpened():       
    open,frame=vc.read()     #读取每一帧图像
else:
    open=False
while open:
    ret,frame=vc.read()
    if frame is None:
        break
    if ret==True:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray',gray)
        if cv2.waitKey(10)&0xFF==ord('q'):
            break
cv2.waitKey(0)
cv2.destroyAllWindows()

'''import matplotlib.pyplot as plt
img=cv2.imread('D:/360Downloads/wpcache/picture_test.jpg')
template=cv2.imread('D:/360Downloads/wpcache/srvsetwp/picture_test_cut.jpg')
h,w=template.shape[:2]
methods=['cv2.TM_SQDIFF','cv2.TM_CCORR','cv2.TM_CCOEFF',
        'cv2.TM_SQDIFF_NORMED','cv2.TM_CCORR_NORMED','cv2.TM_CCOEFF_NORMED']
res=cv2.matchTemplate(img,template,cv2.TM_SQDIFF)
for meth in methods:
    img2=img.copy()
    method=eval(meth)
    print(method)
    res=cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left=min_loc
    else:
        top_left=max_loc
    bottom_right=(top_left[0]+w,top_left[1]+h)
    cv2.rectangle(img2,top_left,bottom_right,255,2)
    plt.subplot(121),plt.imshow(res,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(img2,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.suptitle(meth)
    plt.show()'''
'''图像的融合?
img_bottom=cv2.imread('D:/360Downloads/wpcache/srvsetwp/2054368.jpg')
img_top=cv2.imread('D:/360Downloads/wpcache/srvsetwp/2055085.jpg')
ADD=cv2.addWeighted(img_bottom,0.5,img_top,0.5,0)
import matplotlib.pyplot as plt
plt.imshow('ADD',ADD)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
大律算法和三角形算法?
img=img.astype("uint8")
ret,dst6=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Otsu算法选择最优阈值，与前面的方法配合使用？
res4=np.hstack((img,dst6))
cv_show(res4,'res4')
ret,dst7=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE) #三角形算法选择最优阈值，与前面的方法配合使用？
res5=np.hstack((img,dst7))
cv_show(res5,'res5')
ret,dst8=cv2,threshold(img,127,255,cv2.THRESH_MASK) #用于和设置的阈值相与确保只有符合条件的数值保留，即保留低3位？
res6=np.hstack((img,dst8))
cv_show(res6,'res6')'''
