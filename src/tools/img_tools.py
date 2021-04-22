# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/1/19 12:42
# @FileName : read_img.py
# @Software : PyCharm
import numpy as np
import cv2
import gdal
import os.path
import glymur
import tempfile
from PIL import Image


# 对文件排序
# 返回文件列表名
def file_sort(dir_name):
    """

    :param dir_name: 文件夹名称
    """
    file_names = os.listdir(dir_name)
    file_names.sort(key=lambda x: int(x.split('.')[0]))
    tiff_paths = []
    for path in file_names:
        tiff_path = os.path.join(dir_name, path)
        tiff_paths.append(tiff_path)
    return tiff_paths


# 读取tiff多波段数据
def read_tiff_multi(filename):
    img = np.array([])
    drive = gdal.GetDriverByName('GTiff')
    drive.Register()

    dataset = gdal.Open(filename)
    width = dataset.RasterXSize
    heigth = dataset.RasterYSize
    bands = dataset.RasterCount

    for band in range(bands):
        band += 1
        srcband = dataset.GetRasterBand(band)
        if srcband is None:
            continue
        data = srcband.ReadAsArray(0, 0, width, heigth).astype(np.float32)
        if band == 1:
            img = data.reshape((heigth, width, 1))
        else:
            img = np.append(img, data.reshape((heigth, width, 1)), axis=2)
    return img


# 读取多波段数据
def read_tiff_single(path_list):
    bands = len(path_list)
    image = np.array([])
    for i in range(bands):
        file_path = path_list[i]
        # 读取单波段图像
        data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if i == 0:
            image = np.expand_dims(data, axis=2)
        else:
            image = np.append(image, np.expand_dims(data, axis=2), axis=2)

    return image


# 存储tiff
def write_tiff(im_data, im_width, im_height, im_bands, path):
    """

    :param im_data: 输入图像，格式为[heigth,width,bands]
    :param im_width:
    :param im_height:
    :param im_bands:
    :param path: 存储路径
    """
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


# loadpng
def load_img(path):
    try:
        return cv2.imread(path)
    except IOError:
        return None


def max_ndarray(mat):
    return np.amax(mat) if type(mat).__name__ == 'ndarray' else 0


def jpeg_2000(img, save_path):
    im = Image.fromarray(img)
    im.save(save_path)


# 小波正变换
def dwt(img):
    """

    :type img: np.ndarray
    """
    h0 = np.array([1 / 2, 1 / 2])
    h1 = np.array([1 / 2, -1 / 2])

    L = np.array([])
    H = np.array([])
    L_down = []
    H_down = []

    h, w = img.shape[:2]

    # convolve the rows
    for row in range(h):
        if row == 0:
            L = np.convolve(img[row, :], h0[::-1], "valid")
            H = np.convolve(img[row, :], h1[::-1], "valid")
        else:
            L = np.vstack((L, np.convolve(img[row, :], h0[::-1], "valid")))
            H = np.vstack((H, np.convolve(img[row, :], h1[::-1], "valid")))

    # downSample
    L_down = L[:, ::2]
    H_down = H[:, ::2]

    # colum
    (h, w) = L_down.shape
    ll = np.array([])
    lh = np.array([])
    hl = np.array([])
    hh = np.array([])

    # convolute the columns, appending each column as a sub array to the given array
    for col in range(w):
        if col == 0:
            ll = np.convolve(L_down[:, col], h0[::-1], "valid")
            lh = np.convolve(L_down[:, col], h1[::-1], "valid")
            hl = np.convolve(H_down[:, col], h0[::-1], "valid")
            hh = np.convolve(H_down[:, col], h1[::-1], "valid")
        else:
            ll = np.vstack((ll, np.convolve(L_down[:, col], h0[::-1], "valid")))
            lh = np.vstack((lh, np.convolve(L_down[:, col], h1[::-1], "valid")))
            hl = np.vstack((hl, np.convolve(H_down[:, col], h0[::-1], "valid")))
            hh = np.vstack((hh, np.convolve(H_down[:, col], h1[::-1], "valid")))

    # turn the arrays to np.arrays and transpose them
    # (since columns were appended as rows in above step)

    # 前面在append的时候直接添加所有需要transpose
    hh = np.transpose(np.asarray(hh))[::2, :]
    hl = np.transpose(np.asarray(hl))[::2, :]
    lh = np.transpose(np.asarray(lh))[::2, :]
    ll = np.transpose(np.asarray(ll))[::2, :]
    return ll, hl, lh, hh


# 小波逆变换
def idwt(coffes: tuple):
    """
    :type coffes:numpy.ndarray
    :param coffes:
    :return:
    """

    h0 = np.array([1, 1])
    h1 = np.array([1, -1])

    ll, hl, lh, hh = coffes

    # column upsample
    h, w = ll.shape[:2]

    for row in range(h):
        ll = np.insert(ll, [row * 2], np.zeros(w), axis=0)
        hl = np.insert(hl, [row * 2], np.zeros(w), axis=0)
        lh = np.insert(lh, [row * 2], np.zeros(w), axis=0)
        hh = np.insert(hh, [row * 2], np.zeros(w), axis=0)

    # in_con
    L = np.array([])
    H = np.array([])

    for col in range(w):
        conv_l = np.convolve(ll[:, col], h0) + np.convolve(lh[:, col], h1)
        conv_h = np.convolve(hl[:, col], h0) + np.convolve(hh[:, col], h1)
        if col == 0:
            L = conv_l
            H = conv_h
        else:
            L = np.vstack((L, conv_l))
            H = np.vstack((H, conv_h))

    L = np.transpose(L)[1:]
    H = np.transpose(H)[1:]

    # upsample\
    h, w = L.shape

    for column in range(w):
        L = np.insert(L, column * 2, 0, axis=1)
        H = np.insert(H, column * 2, 0, axis=1)

    original_img = np.array([])

    for row in range(h):
        conv_value = np.convolve(L[row, :], h0) + np.convolve(H[row, :], h1)
        conv_value = conv_value[1:]
        if row == 0:
            original_img = conv_value
        else:
            original_img = np.vstack((original_img, conv_value))

    return original_img
