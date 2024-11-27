import cv2
import numpy as np

# 彩图直方图均衡化

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

if __name__=="__main__":
    img_file=r"./darkFace.png"

    # 读取人脸图片
    img = cv2.imread(img_file)

    res=BGR_Hist_Equalization(img)
    cv2.imshow("res",res)
    cv2.waitKey(0)