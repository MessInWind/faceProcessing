import cv2
from cv2.ximgproc import *
import numpy as np

# https://blog.csdn.net/wxplol/article/details/121832816 面部模板补光法

def BGR_Hist_Equalization(target):
    '''
    对亮度通道(Y 通道)进行直方图均衡化，然后将均衡化后的亮度通道与原始的 U、V 通道合并，
    最后转换回 BGR 颜色空间。这样可以对彩色图像进行直方图均衡化。
    转换为 YUV 颜色空间
    :param img:
    :return:
    '''

    yuv_img = cv2.cvtColor(target, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_img)

    # 对亮度通道进行直方图均衡化
    y_eq = cv2.equalizeHist(y)

    # 合并均衡化后的亮度通道和原始的 U、V 通道
    yuv_img_eq = cv2.merge((y_eq, u, v))

    # 转换回 BGR 颜色空间
    target = cv2.cvtColor(yuv_img_eq, cv2.COLOR_YUV2BGR)

    return target

def lightness_layer_decomposition(img,conf,sigma):
    '''
    将图片分离为颜色层和亮度层
    :param img:
    :return:
    '''
    # wsl滤波
    large_scale_img=cv2.ximgproc.fastGlobalSmootherFilter(img, img, conf, sigma)
    detail_img=img/large_scale_img

    return large_scale_img,detail_img

def face_illumination_transfer(target=None, reference=None):
    '''
    将标签光照迁移到目标人脸上
    :param target:
    :param reference:
    :return:
    '''
    h,w=reference.shape[:2]
    target=cv2.resize(target,(w,h))

    target = BGR_Hist_Equalization(target)    

    # lab颜色转换
    # 提取颜色(a,b)和亮度层l
    lab_img = cv2.cvtColor(target, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_img)

    lab_rimg = cv2.cvtColor(reference, cv2.COLOR_BGR2Lab)
    lr, ar,br = cv2.split(lab_rimg)

    # 将亮度层进行分层:大尺度层和细节层
    large_scale_img, detail_img= lightness_layer_decomposition(l,600,20)
    large_scale_rimg, detail_rimg = lightness_layer_decomposition(lr,600,20)

    # 通过引导波滤波将模板亮度迁移到目标图上
    large_scale_rimg = large_scale_rimg.astype('float32')
    large_scale_img = large_scale_img.astype('float32')
    out=cv2.ximgproc.guidedFilter(large_scale_img,large_scale_rimg,18,1e-3)

    out=out*detail_img
    out=out.astype(np.uint8)
    res=cv2.merge((out,a,b))
    res=cv2.cvtColor(res,cv2.COLOR_Lab2BGR)

    return res

if __name__=="__main__":
    img_file=r"./darkFace.png"
    reference=r"./lightFace.png"

    # 读取人脸图片
    img = cv2.imread(img_file)
    rimg=cv2.imread(reference)

    res=face_illumination_transfer(img,rimg)
    cv2.imshow("res",res)
    cv2.waitKey(0)
