# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/3/19 21:27
# @FileName : image_server.py
# @Software : PyCharm
import socket
import os
import sys
import struct
import threading

import socketserver
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 根据数据长度获取数据流
def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        # 所接收的数据一次不会超过最大数据量
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


class MyServer(socketserver.BaseRequestHandler):

    def setup(self):
        pass

    def handle(self):
        # connect
        sock_ = self.request
        #  获取长度信息
        length = recvall(sock_, 16)

        # 接收图像数据
        img_string = recvall(sock_, int(length))
        img_data = np.frombuffer(img_string, np.uint8)
        img_data = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        cv2.imwrite("../resources/image/11.jpg", img_data)

    def finish(self):
        pass


if __name__ == '__main__':
    # 创建多线程
    server = socketserver.ThreadingTCPServer(('0.0.0.0', 8000), MyServer)

    # 开启多线程异步
    server.serve_forever()

