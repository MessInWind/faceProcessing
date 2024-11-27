import cv2
import numpy as np

# https://blog.csdn.net/weixin_38742868/article/details/142649707 不均匀光照补偿（黑白）
 
def uneven_light_compensate(image, block_size=32):
    # 将图像转换为灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算图像的平均灰度值
    average = np.mean(image)
    
    # 计算新的行列数
    rows_new = int(np.ceil(image.shape[0] / block_size))
    cols_new = int(np.ceil(image.shape[1] / block_size))
    
    # 初始化子块亮度矩阵
    block_image = np.zeros((rows_new, cols_new), dtype=np.float32)
    
    # 遍历子块并计算每个子块的平均灰度值
    for i in range(rows_new):
        for j in range(cols_new):
            row_min = i * block_size
            row_max = min((i + 1) * block_size, image.shape[0])
            col_min = j * block_size
            col_max = min((j + 1) * block_size, image.shape[1])
            
            image_roi = image[row_min:row_max, col_min:col_max]
            temaver = np.mean(image_roi)
            block_image[i, j] = temaver
    
    # 计算子块的亮度差值矩阵
    block_image = block_image - average
    
    # 将差值矩阵差值成与原图一样大小的亮度分布矩阵
    block_image_resized = cv2.resize(block_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 计算矫正后的图像
    image_float = image.astype(np.float32)
    dst = image_float - block_image_resized
    dst = np.clip(dst, 0, 255)  # 限制像素值在0-255之间
    dst = dst.astype(np.uint8)
    
    return dst
 
# 读取图像
image = cv2.imread('./darkFace.png')
 
# 应用不均匀光照补偿算法
compensated_image = uneven_light_compensate(image)
 
# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Compensated Image', compensated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()