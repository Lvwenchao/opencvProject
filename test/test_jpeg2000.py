# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/4/14 19:41
# @FileName : test_jpeg2000.py
# @Software : PyCharm
from src.tools.img_tools import load_img, show_img, get_bits
import os
import matplotlib.pyplot as plt
from src.compression_transmission import jpeg2000
import cv2

FOLDER = '../resources/image'
FILENAME = 'lena.png'


class TestJpeg2000:

    def test_dc_offset(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        B = get_bits(img)
        # img_dc = jpeg2000.dc_offset(img, B)
        img_dc = jpeg2000.dc_offset(img, B)
        plt.imshow(img_dc)
        plt.show()

    def test_rgb_yuv(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        img_yuv = jpeg2000.rgb_yuv(img)
        print(img_yuv.min(), img_yuv.max())

    def test_yuv_rgb(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        img = jpeg2000.dc_offset(img, 8)
        img_yuv = jpeg2000.rgb_yuv(img)
        print(img_yuv.min(), img_yuv.max())
        img_rgb = jpeg2000.yuv_rgb(img_yuv)
        show_img([img_yuv, img_rgb], ["yuv", "rgb"], 2, 1)

    def test_extract_yuv_coeff(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        img = jpeg2000.dc_offset(img, 8)
        img_yuv = jpeg2000.rgb_yuv(img)
        coeffs_yuv = jpeg2000.extract_yuv_coeff(img_yuv)
        assert coeffs_yuv is not None

    def test_quantization(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        img_yuv = jpeg2000.rgb_yuv(img)
        coeffs_yuv = jpeg2000.extract_yuv_coeff(img_yuv)
        quatization_yuv = jpeg2000.quatization(coeffs_yuv, 30)
        assert quatization_yuv is not None

    def test_img_from_dwt_coeff(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        coeffs_dwt = jpeg2000.extract_rgb_coeff(img)
        re_img = jpeg2000.img_from_dwt_coeff(coeffs_dwt)
        print(re_img.max())
        print(re_img.shape)
        plt.imshow(re_img)
        plt.show()

    def test_reconstruct_help(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        coeffs_dwt = jpeg2000.extract_rgb_coeff(img)
        img = jpeg2000.reconstruct_help(coeff=coeffs_dwt.coeff_r, color_value=160)
        plt.imshow(img)
        plt.show()

    def test_forward(self):
        img = load_img(os.path.join(FOLDER, FILENAME))
        quatization_yuv = jpeg2000.forward(img)
        # quatization_y, quatization_u, quatization_v = quatization_yuv
        # img_titles = ['CA', 'CH', 'CV', 'CD']
        # show_img(quatization_y, img_titles, 2, 2)

