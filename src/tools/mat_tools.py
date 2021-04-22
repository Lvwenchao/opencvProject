# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/3/31 17:45
# @FileName : mat_tools.py
# @Software : PyCharm
import numpy as np


# 获取索引数组
def get_index(h, w):
    """

    :param w: width
    :param h: heigth
    :return:
    """
    x_y = []
    for index in np.ndindex((h, w)):
        x_y.append(index)

    return np.asarray(x_y)


# 矩阵拆分
def split_mat(M):
    LL_H = M.shape[0] // 2
    LL_W = M.shape[1] // 2
    mats = M.reshape((2, LL_H, -1, LL_W, 2)).swapaxes(1, 2).reshape(-1, LL_H, LL_W, 2)
    return mats
