# import cv2
# import numpy as np

# # https://blog.csdn.net/weixin_38742868/article/details/142649707 不均匀光照补偿（黑白）
 
# def uneven_light_compensate(image, block_size=32):
#     # 将图像转换为灰度图
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # 计算图像的平均灰度值
#     average = np.mean(image)
    
#     # 计算新的行列数
#     rows_new = int(np.ceil(image.shape[0] / block_size))
#     cols_new = int(np.ceil(image.shape[1] / block_size))
    
#     # 初始化子块亮度矩阵
#     block_image = np.zeros((rows_new, cols_new), dtype=np.float32)
    
#     # 遍历子块并计算每个子块的平均灰度值
#     for i in range(rows_new):
#         for j in range(cols_new):
#             row_min = i * block_size
#             row_max = min((i + 1) * block_size, image.shape[0])
#             col_min = j * block_size
#             col_max = min((j + 1) * block_size, image.shape[1])
            
#             image_roi = image[row_min:row_max, col_min:col_max]
#             temaver = np.mean(image_roi)
#             block_image[i, j] = temaver
    
#     # 计算子块的亮度差值矩阵
#     block_image = block_image - average
    
#     # 将差值矩阵差值成与原图一样大小的亮度分布矩阵
#     block_image_resized = cv2.resize(block_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
#     # 计算矫正后的图像
#     image_float = image.astype(np.float32)
#     dst = image_float - block_image_resized
#     dst = np.clip(dst, 0, 255)  # 限制像素值在0-255之间
#     dst = dst.astype(np.uint8)
    
#     return dst

# def invoke(img):
#     return uneven_light_compensate(img)

# if __name__ == '__main__':
#     # 读取图像
#     image = cv2.imread('./darkFace.png')
    
#     # 应用不均匀光照补偿算法
#     compensated_image = uneven_light_compensate(image)
    
#     # 显示结果
#     cv2.imshow('Original Image', image)
#     cv2.imshow('Compensated Image', compensated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# ↑↑↑↑↑ 不均匀光照补偿 ↑↑↑↑↑ 效果不好 使用普通的光照补偿算法
# 解释
# 调整参数：

# alpha：控制补偿强度。较小的 alpha 值可以减少过度增强的效果。
# beta：控制亮度偏移。可以根据需要调整亮度。
# 限制增强幅度：

# 在计算矫正后的图像时，使用 alpha 和 beta 参数来控制增强的幅度，避免过度增强。
# 平滑处理：

# 在补偿后对图像进行平滑处理，以减少过度增强带来的不自然效果。
# 通过调整这些参数和方法，你可以优化光照补偿效果，使图像看起来更加自然。


import cv2
import numpy as np

def light_compensation(image, block_size, alpha=1.0, beta=0.0):
    # 分离 BGR 通道
    b, g, r = cv2.split(image)
    
    # 对每个通道进行处理
    b = process_channel(b, block_size, alpha, beta)
    g = process_channel(g, block_size, alpha, beta)
    r = process_channel(r, block_size, alpha, beta)
    
    # 合并处理后的通道
    compensated_image = cv2.merge((b, g, r))
    
    return compensated_image

def process_channel(channel, block_size, alpha, beta):
    # 计算图像的平均灰度值
    average = np.mean(channel)
    
    # 计算新的行列数
    rows_new = int(np.ceil(channel.shape[0] / block_size))
    cols_new = int(np.ceil(channel.shape[1] / block_size))
    
    # 初始化子块亮度矩阵
    block_image = np.zeros((rows_new, cols_new), dtype=np.float32)
    
    # 遍历子块并计算每个子块的平均灰度值
    for i in range(rows_new):
        for j in range(cols_new):
            row_min = i * block_size
            row_max = min((i + 1) * block_size, channel.shape[0])
            col_min = j * block_size
            col_max = min((j + 1) * block_size, channel.shape[1])
            
            image_roi = channel[row_min:row_max, col_min:col_max]
            temaver = np.mean(image_roi)
            block_image[i, j] = temaver
    
    # 计算子块的亮度差值矩阵
    block_image = block_image - average
    
    # 将差值矩阵差值成与原图一样大小的亮度分布矩阵
    block_image_resized = cv2.resize(block_image, (channel.shape[1], channel.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 计算矫正后的图像
    channel_float = channel.astype(np.float32)
    dst = channel_float - alpha * block_image_resized + beta
    dst = np.clip(dst, 0, 255)  # 限制像素值在0-255之间
    dst = dst.astype(np.uint8)
    
    return dst

def invoke(img):
    return light_compensation(img, 16, 0.4, 10.0)

if __name__ == '__main__':
    image = cv2.imread('darkFace.png')
    block_size = 16  # 你可以根据需要调整块大小
    alpha = 0.5  # 调整补偿强度
    beta = 10.0  # 调整亮度偏移
    compensated_image = light_compensation(image, block_size, alpha, beta)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Compensated Image', compensated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()