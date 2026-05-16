import cv2
import numpy as np


def gen_mask():
    # 加载两个图像
    image1 = cv2.imread('Whole.bmp')
    # image2 = cv2.imread('Mask-BOSS.png')
    image2 = cv2.imread('Mask-BOSS-Tiger.png')
    # 转换为灰度图
    gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # 应用阈值化
    _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    # 转换为单通道
    mask = mask.astype('uint8')
    # 保存遮罩
    cv2.imwrite('mask.png', mask)
    # 应用遮罩
    masked_image = cv2.bitwise_and(image1, image1, mask=mask)
    # 保存遮罩后的图像
    cv2.imwrite('masked_image.png', masked_image)


# gen_mask()

def mask_image_show():
    import cv2
    # 加载遮罩图像
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    # 确保遮罩是二值图像
    if mask.max() > 1:
        mask = mask // 255
    # 加载原图像
    image = cv2.imread('WHOLE.bmp')
    # 应用遮罩
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    # 保存遮罩后的图像
    cv2.imwrite('masked_image.jpg', masked_image)
    # 显示原图像
    cv2.imshow('Original Image', image)
    # 显示遮罩后的图像
    cv2.imshow('Masked Image', masked_image)
    # 等待键盘输入，然后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


mask_image_show()