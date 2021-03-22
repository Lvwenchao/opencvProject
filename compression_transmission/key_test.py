# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/3/20 10:14
# @FileName : key_test.py
# @Software : PyCharm
from pynput import keyboard
import sys

with keyboard.Events() as events:
    # Block at most one second
    for event in events:
        while True:
            if event.key == keyboard.Key.esc:
                print('Received event {}'.format(event))
            print(1)