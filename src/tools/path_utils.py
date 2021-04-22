# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/4/22 21:50
# @FileName : path_utils.py
# @Software : PyCharm
import os
import sys


class PathUtil(object):
    """路径处理工具类"""

    def __init__(self):
        # 判断调试模式
        debug_vars = dict((a, b) for a, b in os.environ.items()
                          if a.find('IPYTHONENABLE') >= 0)
        # 根据不同场景获取根目录
        if len(debug_vars) > 0:
            """当前为debug运行时"""
            self.root_path = sys.path[2]
        elif getattr(sys, 'frozen', False):
            """当前为exe运行时"""
            self.root_pathr = os.getcwd()
        else:
            """正常执行"""
            self.root_path = sys.path[1]
        # 替换斜杠
        self.root_path = self.root_path.replace("\\", "/")

    def getPathFromResources(self, fileName):
        """按照文件名拼接资源文件路径"""
        file_path = "%s/resources/%s" % (self.root_path, fileName)
        return file_path
