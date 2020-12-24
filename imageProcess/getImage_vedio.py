# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2020/12/17 17:42
# @FileName : getImage_vedio.py
# @Software : PyCharm
import cv2
import numpy as np

# 视频帧率
timeF = 25

# 帧数
flag = 0

# 打开文件
cap = cv2.VideoCapture(r"E:\pythonProject\opencvProject\resources\candle.avi")
while cap.isOpened():
    cap.set(cv2.CAP_PROP_FPS, flag)
    rval, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    flag += 1
    cv2.imwrite("candle_{}.jpg".format(flag), frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
