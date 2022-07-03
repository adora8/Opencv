import cv2
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
img=cv2.imread('D:/360Downloads/wpcache/srvsetwp/examples.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv_show(thresh,'thresh')
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
draw_img=img.copy()
res=cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
cv_show(res,'res')
cnt=contours[2]
area=cv2.contourArea(cnt)
length=cv2.arcLength(cnt,True)
print(area)
print(length)
epsilon=0.001*cv2.arcLength(cnt,True)
approx=cv2.approxPolyDP(cnt,epsilon,True)
draw_img=img.copy()
res=cv2.drawContours(draw_img,[approx],-1,(0,0,255),2)
cv_show(res,'res')
x,y,w,h=cv2.boundingRect(cnt)
img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv_show(img,'rectangle')
area=cv2.contourArea(cnt)
rect_area=w*h
extent=float(area)/rect_area
print(extent)
(x,y),radius=cv2.minEnclosingCircle(cnt)
Center=(int(x),int(y))
radius=int(radius)
img=cv2.circle(img,Center,radius,(0,255,0),2)
cv_show(img,'circle')
target=cv2.imread('D:/360Downloads/wpcache/picture_test.jpg')
template=cv2.imread('D:/360Downloads/wpcache/srvsetwp/picture_test_cut.jpg')
theight,twidth=template.shape[:2]
result=cv2.matchTemplate(target,template,cv2.TM_SQDIFF_NORMED)
cv2.normalize(result,result,0,1,cv2.NORM_MINMAX,-1)
min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(result)
strmin_val=str(min_val)
cv2.rectangle(target,min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(0,255,0),2)
cv_show(target,'MatchResult--MatchingValue='+strmin_val)
import matplotlib.pyplot as plt
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
    plt.show()

#Other Example:D:/360Downloads/wpcache/picture_test.jpg
