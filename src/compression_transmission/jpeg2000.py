# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/4/14 18:35
# @FileName : jpeg2000.py
# @Software : PyCharm
import numpy as np
import cv2
from collections import namedtuple

from PIL import Image

from src.tools.img_tools import max_ndarray, dwt, idwt


# convert bgr to yuv
def bgr_yuv(img):
    """
    :param img: BRG
    :return:
    """
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    except Exception:
        return None


# bgr_coeff
def extract_bgr_coeff(img):
    coeffs_b = dwt(img[:, :, 0])
    coeffs_g = dwt(img[:, :, 1])
    coeffs_r = dwt(img[:, :, 2])
    return namedtuple("coeffs_bgr", ['coeff_b', 'coeff_g', 'coeff_r'])(coeffs_b, coeffs_g, coeffs_r)


# yuv_coeff
def extract_yuv_coeff(img):
    """
    Returns BRG dwt applied coefficients tuple
    :param img: [H,W,3] YUV
    :returns:(coeffs_b, coeffs_r, coeffs_g)
    """
    coeffs_y = dwt(img[:, :, 0])
    coeffs_u = dwt(img[:, :, 1])
    coeffs_v = dwt(img[:, :, 2])

    return namedtuple("coeffs_yuv", ['coeff_y', 'coeff_u', 'coeff_v'])(coeffs_y, coeffs_u, coeffs_v)


# quantization
def quatization_math(img, step):
    (h, w) = img.shape
    i_quatization_img = np.empty_like(img)

    # loop through ever coefficient in img
    for i in range(0, w):
        for j in range(0, h):
            # save un-quatized coefficient
            i_quatization_img[j][i] = img[j][i] * step
    return i_quatization_img


def quatization_coeff(coeff, step):
    ca = quatization_math(coeff[0], step)
    ch = quatization_math(coeff[1][0], step)
    cv = quatization_math(coeff[1][1], step)
    cd = quatization_math(coeff[1][2], step)
    return namedtuple('quatization', ['ca', 'ch', 'cv', 'cd'])(ca, ch, cv, cd)


def quatization(coeffs, step):
    quatization_y = quatization_coeff(coeffs.coeff_y, step)
    quatization_u = quatization_coeff(coeffs.coeff_u, step)
    quatization_v = quatization_coeff(coeffs.coeff_v, step)
    return namedtuple('quatization_yuv',
                      ['quatization_y',
                       'quatization_u',
                       'quatization_v'])(quatization_y, quatization_u, quatization_v)

    # inverse_quantization


# i_quatization
def i_quatization_math(img, step):
    (h, w) = img.shape
    quantization_img = np.empty_like(img)
    for i in range(0, w):
        for j in range(0, h):
            # save the sign
            if img[j][i] >= 0:
                sign = 1
            else:
                sign = -1
            # save quantized coeffcicient
            quantization_img[j][i] = sign * np.floor(abs(img[j][i]) / step)
    return quantization_img


# 图像重建
def reconstruct_help(coeff, color_value):
    c_a = coeff[0]
    c_h = coeff[1][0]
    c_v = coeff[1][1]
    c_d = coeff[1][2]
    c_a = (c_a / max_ndarray(c_a)) * color_value
    c_h = (c_h / max_ndarray(c_h)) * color_value
    c_v = (c_v / max_ndarray(c_v)) * color_value
    c_d = (c_d / max_ndarray(c_d)) * color_value
    img = np.vstack((np.hstack((c_a, c_h)), np.hstack((c_v, c_d)))).astype(dtype=np.int16)
    return img


def img_from_dwt_coeff(coeff_dwt):
    """
    Returns Image recreated from dwt coefficients
    Parameters
    ----------
    (coeffs_r, coeffs_g, coeffs_b):
        RGB coefficients with Discrete Wavelet Transform Applied
    Returns
    -------
    Image from dwt coefficients
    """
    # Channel Red

    img_r = reconstruct_help(coeff_dwt.coeff_r, 160)
    img_g = reconstruct_help(coeff_dwt.coeff_g, 85)
    img_b = reconstruct_help(coeff_dwt.coeff_b, 100)

    re_img = np.stack((img_r, img_g, img_b), axis=-1).astype(np.int16)
    return re_img


def forward(img):
    img_yuv = bgr_yuv(img)
    coeffs_yuv = extract_yuv_coeff(img_yuv)
    quatization_yuv = quatization(coeffs_yuv, 30)
    return quatization_yuv
