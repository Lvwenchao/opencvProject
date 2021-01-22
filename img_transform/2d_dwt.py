# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/1/18 9:00
# @FileName : dwt.py
# @Software : PyCharm

import numpy as np
import cv2
from tools.file_tools import read_tiff_multi
import gdal
import os.path

if __name__ == '__main__':
    img = read_tiff_multi('../resources/process/1.tiff')
    print(img.shape)
