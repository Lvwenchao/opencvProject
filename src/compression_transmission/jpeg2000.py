# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/4/14 18:35
# @FileName : jpeg2000.py
# @Software : PyCharm
import re

import numpy as np
from collections import namedtuple
from src.tools import img_tools
from src.tools.img_tools import max_ndarray, dwt, idwt


# dc平移
def dc_offset(img, B):
    """

    :param B: 图像位数
    :type img: np.ndarray
    """
    img_dc = np.array(img - 2 ** (B - 1), dtype=np.int8)
    return img_dc


def i_dc_offset(img, B):
    img_i_dc = np.array(img + 2 ** (B - 1), dtype=np.uint8)
    return img_i_dc


# convert bgr to yuv
def rgb_yuv(img):
    """
    :param img: BRG
    :return:
    """
    img_yuv = np.empty_like(img)
    h, w = img_yuv.shape[:2]
    for row in range(h):
        for col in range(w):
            R, G, B = img[row, col, :]
            Y = 0.25 * R + 0.5 * G + 0.25 * B
            U = B - G + 128
            V = R - G + 128
            img_yuv[row, col, :] = Y, U, V
    return img_yuv


def yuv_rgb(img):
    img_rgb = np.empty_like(img)
    h, w = img.shape[:2]
    for row in range(h):
        for col in range(w):
            Y, U, V = img[row, col, :]
            G = Y - 0.25 * (U - 128) - 0.25 * (V - 128)
            B = U + G - 128
            R = V + G - 128
            img_rgb[row, col, :] = R, G, B
    return img_rgb


def extract_rgb_coeff(img):
    coeffs_r = dwt(img[:, :, 0])
    coeffs_g = dwt(img[:, :, 1])
    coeffs_b = dwt(img[:, :, 2])
    return namedtuple("coeffs_rgb", ['coeff_r', 'coeff_g', 'coeff_b'])(coeffs_r, coeffs_g, coeffs_b)


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
    B = img_tools.get_bits(img)
    img = dc_offset(img, B)
    print(img.shape)
    img_yuv = rgb_yuv(img)

    # print(img_yuv.shape)
    # print(type(img_yuv))
    # print(img_yuv.shape)
    # coeffs_yuv = extract_yuv_coeff(img_yuv)
    # quatization_yuv = quatization(coeffs_yuv, 30)
    # return quatization_yuv
