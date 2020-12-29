# -*- using coding=utf-8 -*-

import os
import numpy as np
import gdal

import glob


class GRID:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件

        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

        del dataset
        return im_proj, im_geotrans, im_data

    # 写文件，以写成tif为例
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

            # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)


        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset

    # data 传入数据，path 生成统计文件的路径,
    # name 生成的文件名,
    # bandinfo波段信息-含有中心波长，list格式
    def statistics(self, data, path, name, bandinfo=None):
        if len(data.shape) == 3:
            print()
            channel, width, height = data.shape
            with open(path + '/' + name, 'w') as file_to_write:
                for i in range(channel):
                    im_mean = np.mean(data[i, :, :])  # 图像的平均值
                    im_std = np.std(data[i, :, :])
                    # 是否
                    # 将波段号转换为从1开始
                    file_to_write.write(str(i + 1) + ' ' + str(im_mean) + '\n')
        else:
            channel, (width, height) = 1, zip(data.shape)
            with open(path + '/' + name, 'w') as file_to_write:
                for i in range(channel):
                    im_mean = np.mean(data)  # 图像的平均值
                    im_std = np.std(data)
                    # 是否
                    # 将波段号转换为从1开始
                    file_to_write.write(str(i + 1) + ' ' + str(im_mean) + '\n')

    # path 查找的路径 ;type 文件的类型
    def searchtiff(self, path, type):
        # 使用上type
        files = glob(os.path.join(path, '*/*.jpg'))

    # 求一阶导数,
    # file 已经统计好的文件名
    # interval 高光谱假设间隔是固定的
    # bandinfo,波段信息 list格式，中心波段波长
    def spectr_diff(self, file, interval=None, bandinfo=None):
        print()
        # 考虑到如果遇到没有数据的情况，跳过，两个中有一个没有
        if interval != None:
            print()

    # 包络线，求吸收
    def continum_remove(self):
        print()
