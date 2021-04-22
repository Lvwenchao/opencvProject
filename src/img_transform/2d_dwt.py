# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/1/18 9:00
# @FileName : dwt.py
# @Software : PyCharm

from src.tools import read_tiff_multi

if __name__ == '__main__':
    img = read_tiff_multi('../resources/process/1.tiff')
    print(img.shape)
