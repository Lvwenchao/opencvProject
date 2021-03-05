# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/3/3 19:53
# @FileName : fourier_transform.py
# @Software : PyCharm
# 傅里叶变换
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def dft(file_path):
    img = cv2.imread(file_path, 0)

    # numpy 傅里叶变换
    f = np.fft.fft2(img)  # f[w,h]
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    # 去除低频分量建立高通滤波
    row, col = img.shape
    rows = int(row / 2)
    cols = int(col / 2)
    f_shift[rows - 40:rows + 40, cols - 40:cols + 40] = 0
    f_ishift = np.fft.ifftshift(f_shift)

    # 取绝对值
    img_back = np.abs(np.fft.ifft2(f_ishift))

    plt.subplot(121), plt.imshow(img, cmap='gray'),
    plt.title("input_image"), plt.xticks([]), plt.yticks([]),
    plt.subplot(122), plt.imshow(img_back, cmap='gray'),
    plt.title("Result in JET"), plt.xticks([]), plt.yticks([])
    plt.show()

    # opencv 傅里叶变换
    # 输入图像要先转为float32 opencv 得到的是双通道dft[h,w,2]
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum_cv = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    plt.subplot(121), plt.imshow(img, cmap='gray'),
    plt.title("image"), plt.xticks([]), plt.yticks([]),
    plt.subplot(122), plt.imshow(magnitude_spectrum_cv, cmap='gray'),
    plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.show()

    # 建立低通滤波掩膜
    mask = np.zeros((row, col, 2), np.uint8)
    mask[rows - 40:rows + 40, cols - 40:cols + 40] = 1
    f_shift = dft_shift * mask

    # 逆频移
    f_ishift = np.fft.ifftshift(f_shift)
    # 逆变换
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121), plt.imshow(img, cmap='gray'),
    plt.title("image"), plt.xticks([]), plt.yticks([]),
    plt.subplot(122), plt.imshow(img_back, cmap='gray'),
    plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.show()

    # cv2.getOptimalDFTSize 进行DFT的西南性能优化
    print(row, col)
    nrow = cv2.getOptimalDFTSize(row)
    ncol = cv2.getOptimalDFTSize(col)
    print(nrow, ncol)


if __name__ == '__main__':
    file_name = 'home.JPG'
    path = "../resources/image/"
    file_path = os.path.join(path, file_name)
    dft(file_path)
