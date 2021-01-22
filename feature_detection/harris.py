# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2020/12/23 10:25
# @FileName : harris.py
# @Software : PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import lib


# 自定义角点检测
def harris(img, ksize=3, k=0.04):
    """

    :param img: input_img
    :param ksize: size of sobel kernal
    :param k: k of R
    """
    threhold = 0.005
    h, w = img.shape[:2]
    # 计算在x和y方向的梯度
    grads = np.zeros((h, w, 2), dtype=np.float32)
    grads[:, :, 0] = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize)
    grads[:, :, 1] = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize)

    # 计算Ix^2,IxIy,Iy^2
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[:, :, 0] = grads[:, :, 0] ** 2
    m[:, :, 1] = grads[:, :, 1] ** 2
    m[:, :, 2] = grads[:, :, 0] * grads[:, :, 1]

    # 计算每个图像的M 矩阵 ，M=w(x,y)grads
    # 对梯度进行高斯滤波处理
    m[:, :, 0] = cv2.GaussianBlur(m[:, :, 0], (3, 3), sigmaX=2)
    m[:, :, 1] = cv2.GaussianBlur(m[:, :, 1], (3, 3), sigmaX=2)
    m[:, :, 2] = cv2.GaussianBlur(m[:, :, 2], (3, 3), sigmaX=2)
    M = [np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]
    M = np.array(M)

    # 计算特征值和R
    D, T = list(map(np.linalg.det, M)), list(map(np.trace, M))
    R = np.array([d - k * t ** 2 for d, t in zip(D, T)])
    R = np.reshape(R, (h, w))
    # 获取最大值
    R_max = R.max()
    corner = np.zeros_like(R, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            # 进行阈值检测
            if R[i, j] > R_max * threhold:
                corner[i, j] = 255

    return corner


def main():
    filename = r"../resources/process/man.jpg"
    img1 = cv2.imread(filename)
    img2 = img1.copy()
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    img2[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite("../resources/result/harris_opencv.jpg", img2)

    # 使用自定义的harris函数
    dst = harris(gray)
    print(dst.shape)
    # 对R分数进行检测，相当于两次阈值检测
    img2[dst > 0.005 * dst.max()] = [0, 0, 255]
    cv2.imwrite("../resources/result/harris.jpg", img2)
    cv2.imshow('harris', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
