# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2020/12/29 14:07
# @FileName : sift.py
# @Software : PyCharm
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    img1 = cv2.imread('../resources/process/DJI_0020.JPG')
    img2 = cv2.imread('../resources/process/DJI_0040.JPG')

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    hmerge = np.hstack((gray1, gray2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    sift = cv2.SIFT_create()
    kp = sift.detect(gray1, None)


if __name__ == '__main__':
    main()
