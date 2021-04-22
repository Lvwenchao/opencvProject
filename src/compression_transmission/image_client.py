# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/3/6 11:06
# @FileName : client.py
# @Software : PyCharm
import socket
import os
import cv2
import numpy as np


# 检测是否释放ESC
# 图像传输客户端：
def client_side(file_path):
    # 压缩编码
    encode_param = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), 15]
    # # 建立客户端socket
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 建立与服务端的连接需要指定前面的连接
    client.connect(('127.0.0.1', 8000))

    # 向服务端发送信息
    img_name = os.path.basename(file_path)
    img = cv2.imread(file_path)
    img_data = img.copy()
    # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)

    # 将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输。
    result, img_encode = cv2.imencode(".jpg", img_data, encode_param)

    # 转换成字符形式便于传输
    str_encode = img_encode.tostring()

    # 发送数据长度
    client.send(str.encode(str(len(str_encode)).ljust(16)))

    # 发送数据
    client.send(str_encode)


if __name__ == '__main__':
    file_path = "../../resources/image/DJI_0020.JPG"
    client_side(file_path=file_path)

    # 多线程传输
    filelist = []
