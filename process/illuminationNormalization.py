import cv2
import numpy as np

# https://developer.baidu.com/article/detail.html?id=3339137 光照归一化

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

def MSRCR(img, scales=[15, 80, 250], sigma=125, G=192, b=30, alpha=125, beta=-46):
    img = BGR_Hist_Equalization(img)
    img = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)
    for scale in scales:
        retinex += cv2.log(img) - cv2.log(cv2.GaussianBlur(img, (0, 0), scale))
    retinex = retinex / len(scales)
    
    for i in range(img.shape[2]):
        unique, count = np.unique(img[:, :, i], return_counts=True)
        s = np.cumsum(count) / float(count.sum())
        s = np.interp(img[:, :, i].flatten(), unique, s)
        img[:, :, i] = s.reshape(img[:, :, i].shape)
    
    img = G * (img - b)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img

def invoke(img):
    img = BGR_Hist_Equalization(img)
    img = MSRCR(img)
    return img

if __name__ == '__main__':
    img = cv2.imread('./darkFace.png')
    result = MSRCR(img)
    
    cv2.imshow('Original Image', img)
    cv2.imshow('Illumination Normalized Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()