# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/3/2 9:49
# @FileName : image_histogram.py
# @Software : PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 全局设置
np.set_printoptions(suppress=True)


# 绘制直方图
def draw_his():
    img = cv2.imread("../../resources/image/DJI_0030.JPG")

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
    cdf_normalization = cdf * hist.max(initial=None) / cdf.max()

    # 图像均衡化
    # 设置忽略0元素的掩码数组

    cdf_mask = np.ma.masked_equal(cdf, 0)
    cdf_mask = (cdf_mask - cdf_mask.min()) * 255 / (cdf_mask.max() - cdf_mask.min())
    h = np.ma.filled(cdf_mask, 0).astype('uint8')

    # 通过建立的列表进行均衡化,并显示直方图和累积分布
    img_hist = h[img]
    hist2, bins2 = np.histogram(img_hist.ravel(), 256, [0, 256])
    cdf_hist = hist2.cumsum()
    cdf_hist_normalization = cdf_hist * hist2.max(initial=None) / cdf_hist.max()

    plt.subplot(121), plt.plot(cdf_normalization, color='b'), plt.hist(img.ravel(), 256, [0, 256], color='r')
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.xlim([0, 255])
    plt.subplot(122), plt.plot(cdf_hist_normalization, color='b'), plt.hist(img_hist.ravel(), 256, [0, 256], color='r')
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.xlim([0, 255])
    plt.show()

    # opencv 中的直方图均衡
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))
    cv2.imwrite('../resources/result/equalization_his_res.jpg', res)


# 2_D直方图
def his_2d(filepath):
    img = cv2.imread(filepath)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # # numpy 统计2d直方图
    # hist, xbins, ybins = np.histogram2d(img_hsv[:, :, 0].ravel(),
    #                                     img_hsv[:, :, 1].ravel(),
    #                                     [180, 256],
    #                                     [[0, 180], [0, 256]])

    # 绘制直方图
    plt.imshow(hist, interpolation='nearest')
    plt.show()


# 直方图反向投影
def back_projection():
    image = cv2.imread('../../resources/image/DJI_0030.JPG')
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sample = image[800:1000, 800:1000, :]
    sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

    M = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    N = cv2.calcHist([sample_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # 进行归一化
    cv2.normalize(N, N, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([img_hsv], [0, 1], N, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dst = cv2.filter2D(dst, -1, disc)
    # threshold and binary AND
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    # # 别忘了是三通道图像，因此这里使用merge 变成3 通道
    thresh = cv2.merge((thresh, thresh, thresh))
    print(thresh.shape)
    # # 按位操作
    res = cv2.bitwise_and(image, thresh)
    # res = np.hstack((image, thresh, res))
    cv2.imwrite('../resources/result/res.jpg', res)
    # 显示图像
    plt.imshow(res)
    plt.show()


if __name__ == '__main__':
    path = "../../resources/image/"
    filename = 'home.jpg'
    filepath = os.path.join(path, filename)
    # his_2d(filepath)
    back_projection()
