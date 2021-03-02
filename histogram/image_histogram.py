# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/3/2 9:49
# @FileName : image_histogram.py
# @Software : PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 全局设置
np.set_printoptions(suppress=True)


# 绘制直方图
def draw_his():
    img = cv2.imread("../resources/image/DJI_0030.JPG")

    # 通过CV统计直方图
    # hist = cv2.calcHist([img], [0], None, [256], [0, 256]).astype(np.int32).ravel()

    # 通过numpy统计直方图
    # hist, bins = np.histogram(img[:, :, 0].ravel(), 256, [0, 256])

    # hist 统计并绘制直方图
    # plt.hist(img.ravel(), 256, (0, 256))

    # opencv通道顺序为bgr
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])

    # 掩膜
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[400:900, 400:1200] = 255
    mask_img = cv2.bitwise_and(img, img, mask=mask)
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(mask_img, 'gray')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.xlim([0, 256])

    plt.show()


# 直方图均衡化
def equalization_his():
    img = cv2.imread("../resources/image/300px-Unequalized_Hawkes_Bay_NZ.jpg", 0)
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    # 计算累积分布图
    cdf = hist.cumsum()
    cdf_normalization = cdf * hist.max() / cdf.max()

    # 图像均衡化
    # 设置忽略0元素的掩码数组

    cdf_mask = np.ma.masked_equal(cdf, 0)
    cdf_mask = (cdf_mask - cdf_mask.min()) * 255 / (cdf_mask.max() - cdf_mask.min())
    h = np.ma.filled(cdf_mask, 0).astype('uint8')

    # 通过建立的列表进行均衡化,并显示直方图和累积分布
    img_hist = h[img]
    hist2, bins2 = np.histogram(img_hist.ravel(), 256, [0, 256])
    cdf_hist = hist2.cumsum()
    cdf_hist_normalization = cdf_hist * hist2.max() / cdf_hist.max()

    plt.subplot(121), plt.plot(cdf_normalization, color='b'), plt.hist(img.ravel(), 256, [0, 256], color='r')
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.xlim([0, 255])
    plt.subplot(122), plt.plot(cdf_hist_normalization, color='b'), plt.hist(img_hist.ravel(), 256, [0, 256], color='r')
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.xlim([0, 255])
    plt.show()


if __name__ == '__main__':
    equalization_his()
