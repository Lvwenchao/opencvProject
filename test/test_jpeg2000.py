# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/4/14 19:41
# @FileName : test_jpeg2000.py
# @Software : PyCharm
from src.tools.img_tools import load_img
import os
import matplotlib.pyplot as plt
import cv2
from src.compression_transmission import jpeg2000
from PIL import Image

FOLDER = '../resources/image'
FILENAME = 'lena.png'


class TestJpeg2000:

    def test_extract_bgr_coeff(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        coeffs_b, coeffs_g, coeffs_r = jpeg2000.extract_bgr_coeff(img)
        ca_b, ch_b, cv_b, cd_b = coeffs_b
        print(ca_b.max())
        print(ch_b.min())
        plt.subplot(221), plt.imshow(ca_b, cmap='gray')
        plt.subplot(222), plt.imshow(ch_b, cmap='gray')
        plt.subplot(223), plt.imshow(cv_b, cmap='gray')
        plt.subplot(224), plt.imshow(cd_b, cmap='gray')
        plt.show()

    def test_extract_yuv_coeff(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        img_yuv = jpeg2000.bgr_yuv(img)
        print(img_yuv.shape)
        coeffs_yuv = jpeg2000.extract_yuv_coeff(img_yuv)
        assert coeffs_yuv is not None

    def test_quantization(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        img_yuv = jpeg2000.bgr_yuv(img)
        coeffs_yuv = jpeg2000.extract_yuv_coeff(img_yuv)
        quatization_yuv = jpeg2000.quatization(coeffs_yuv, 30)
        assert quatization_yuv is not None

    def test_img_from_dwt_coeff(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        coeffs_dwt = jpeg2000.extract_bgr_coeff(img)
        re_img = jpeg2000.img_from_dwt_coeff(coeffs_dwt)
        print(re_img.max())
        print(re_img.shape)
        plt.imshow(re_img)
        plt.show()

    def test_reconstruct_help(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        coeffs_dwt = jpeg2000.extract_bgr_coeff(img)
        img = jpeg2000.reconstruct_help(coeff=coeffs_dwt.coeff_r, color_value=160)
        plt.imshow(img)
        plt.show()

    def test_forward(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        quatization_yuv = jpeg2000.forward(img)
