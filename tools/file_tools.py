# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/1/19 12:42
# @FileName : read_img.py
# @Software : PyCharm
import numpy as np
import cv2

import gdal
import os.path


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
