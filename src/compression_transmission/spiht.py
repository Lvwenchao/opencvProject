# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/3/23 11:01
# @FileName : spiht .py
# @Software : PyCharm
import pywt
import numpy as np
import time


class Spiht(object):
    def __init__(self, img, im_dim, code_dim):
        """

        :param img:
        :param im_dim:  小波变换级数
        :param code_dim:  spiht编码级数
        """
        self.img = img
        self.im_dim = im_dim
        self.code_dim = code_dim
        self.lip_init = np.empty((0, 2), dtype=np.int8)
        self.dec_im = np.array([])

    # 小波变换
    def get_wavemat(self):
        wave_data = pywt.wavedec2(self.img, 'haar', level=self.im_dim)
        CA3, (CH3, CV3, CD3), (CH2, CV2, CD2), (CH1, CV1, CD1) = wave_data

        # 计算CA3
        AH3 = np.hstack((CA3, (CH3 + 510)))
        VD3 = np.hstack((CV3 + 510, CD3 + 510))
        CA2 = np.vstack((AH3, VD3))

        # CA1
        AH2 = np.hstack((CA2, CH2 + 510))
        VD2 = np.hstack((CV2 + 510, CD2 + 510))
        CA1 = np.vstack((AH2, VD2))

        # wave_mat
        AH1 = np.hstack((CA1, (CH1 + 255) * 2))
        VD1 = np.hstack(((CV1 + 255) * 2, (CD1 + 255) * 2))

        dec_im = np.vstack((AH1, VD1))

        return dec_im

    # compute the threshold for wave_mat
    def get_threhold(self):
        """
        :return: T: 阈值列表
        """
        T = []
        T0 = 2 ** np.floor(np.log2(self.dec_im.max(initial=None)))
        T.append(T0)
        for i in range(1, self.code_dim):
            Ti = T[i - 1] / 2
            T.append(Ti)

        return T

    # 重要性判别（包括单个元素和元素集合）
    def sn_out(self, T, coffset):
        """
        :param T: 阈值
        :param coffset: 坐标集
        :returns:
            is_imt 重要性(1,0)表示
            sign: 正负系数(1,0)表示
        """
        is_imt = 0
        sign = 0
        allmat = []
        # 如果其单点
        if len(coffset.shape) == 1:
            coffset = np.expand_dims(coffset, axis=0)
        for i, (x, y) in enumerate(coffset):
            allmat.append(self.dec_im[x, y])
            if abs(self.dec_im[x, y]) >= T:
                is_imt = 1
                break
        if len(allmat) and allmat[0] >= 0:
            sign = 1

        return is_imt, sign

    # EZW 扫描
    def list_order(self, mh, mw, x, y):
        """

        :param mh: 扫描矩阵长
        :param mw: 扫描矩阵宽
        :param x: 左上角坐标x
        :param y: 左上角坐标y
        """
        # 矩阵分为四块右上角点
        iso = np.asarray([[x, y], [x, y + mw // 2], [x + mh // 2, y], [x + mh // 2, y + mw // 2]])

        mh = mh // 2
        mw = mw // 2
        tm = np.empty((0, 2), dtype=np.int8)
        if mh > 1 or mw > 1:
            for i, (x, y) in enumerate(iso):
                ls = self.list_order(mh, mw, x, y)
                self.lip_init = np.vstack((self.lip_init, ls))

        return iso

    def child_mat(self, x, y):
        """

        :param x: 坐标x
        :param y: 坐标y
        :return: 所有子坐标和孩子坐标
        """
        tr_all = self.tree_mat(x, y)
        tr_zero = 1
        for i, (x, y) in enumerate(tr_all):
            if self.dec_im[x, y] != 0:
                tr_zero = 0
                break
        if tr_zero:
            tr_all = np.array([])
            tr_cho = np.array([])
        else:
            tr_cho = tr_all[:4]

        return tr_all, tr_cho

    # 子树生成
    def coef_dol(self, x, y, tree_type):
        """

        :param x:
        :param y:
        :param tree_type: 子树类型{D,L,O}
        :return: ch_tree: np.array,子树集合
        """
        ch_tree = []
        ch_d, ch_o = self.child_mat(x, y)
        ch_l = ch_d[ch_o.shape[0]:]
        if tree_type == "D":
            ch_tree = ch_d
        elif tree_type == "O":
            ch_tree = ch_o
        elif tree_type == "L":
            ch_tree = ch_l
        else:
            raise ValueError

        return ch_tree

    # 根据坐标生成树
    def tree_mat(self, x, y):
        h, w = self.dec_im.shape
        cp = np.empty((0, 2), dtype=np.int8)
        if x < h // 2 and y < h // 2:
            tp = np.asarray([[2 * x, 2 * y], [2 * x, 2 * y + 1],
                             [2 * x + 1, 2 * y], [2 * x + 1, 2 * y + 1]])
            tm = np.empty((0, 2), dtype=np.int8)
            if tp[3, 0] < h // 2 and tp[3, 1] < w // 2:
                for idx, (r, c) in enumerate(tp):
                    tm = np.append(tm, self.tree_mat(r, c), axis=0)

            cp = np.append(tp, tm, axis=0)

        return cp

    # 扫描LIP
    def lip_scan(self, LIP, T, LSP, sn):
        """
        :param LIP: 不重要坐标
        :param T: 阈值
        :param LSP: 重要坐标
        :param sn: 排序位流
        :returns:
        LIP: 扫描后LIP
        LSP：加入重要元素后
        sn：扫描后排序位值
        """
        r = 0
        while r < LIP.shape[0]:
            # 获取重要标志和正负符号位
            is_lmt, sign = self.sn_out(T, LIP[r])
            # 如果重要，输入1到sn
            if is_lmt:
                sn.append(1)

                # 输入符号位
                sn.append(sign)
                # 加入坐标到LSP
                LSP = np.vstack((LSP, LIP[r]))
                # 从LIS中删除坐标
                LIP = np.delete(LIP, r, axis=0)
            # 不重要，输入0到sn
            else:
                sn.append(0)
                r += 1

        return LIP, LSP, sn

    # 扫描LIS
    def lis_scan(self, T, sn, LIS, LSP, LIS_flag):
        """

        :param T: 阈值
        :param sn: 扫描输出流
        :param LIS: 不重要系数表
        :param LSP: 重要系数表
        :param LIS_flag:
        :returns:
        sn:扫描输出
        LIs：更新后LIS
        LSP:更新后LSP
        LIS_flag:更新后flag
        """
        r = 0
        while r < LIS.shape[0]:
            x, y = LIS[r]
            # 如果为'D'型
            if LIS_flag[r] == 'D':
                ch_d = self.coef_dol(x, y, 'D')
                is_imt, sign = self.sn_out(T, ch_d)
                # 如果重要分为L和O
                if is_imt:
                    sn.append(1)
                    ch_o = self.coef_dol(x, y, 'O')
                    ch_l = self.coef_dol(x, y, 'L')
                    # 判断O中每个系数的重要性
                    for i in range(4):
                        is_lmt, sign = self.sn_out(T, ch_o[i])
                        if is_lmt:
                            sn.append(1)
                            # 输入符号位
                            sn.append(sign)
                            LSP = np.vstack((LSP, ch_o[i]))
                        else:
                            sn.append(0)
                            # 加入到LIS
                            LIS = np.vstack((LIS, ch_o[i]))
                    # 判断L(r,c)是否为空
                    if ch_l.size != 0:
                        LIS = np.vstack((LIS, LIS[r]))
                        LIS_flag = np.append('L')
                        LIS = np.delete()
                    LIS = np.delete(LIS, r, axis=0)
                    LIS_flag = np.delete(LIS_flag, r)

                    # 将D修改L后或者为空均要删除
                # 不重要则下一项：
                else:
                    sn.append(0)
                    r += 1
            # 如果为'L'型
            elif LIS_flag[r] == 'L':
                ch_l = self.coef_dol(x, y, 'L')
                is_lmt, sign = self.sn_out(T, ch_l)
                # 如果L树重要
                if is_lmt:
                    sn.append(1)
                    ch_o = self.coef_dol(x, y, 'O')
                    # '将该L类移除
                    LIS = np.delete(LIS, r, axis=0)
                    LIS_flag = np.delete(LIS_flag, r)
                    # '将所属的O类子树坐标作为类加入
                    LIS = np.vstack((LIS, ch_o))
                    LIS_flag = np.append(LIS_flag, ['D' for _ in range(4)])
                else:
                    LIS = np.delete(LIS, r, axis=0)
                    LIS_flag = np.delete(LIS_flag, r)

        return sn, LIS, LSP, LIS_flag

        # 精细扫描

    def refine_scan(self):
        pass

    # 编码器
    def spiht_encoder(self):
        # 计算小波变化矩阵
        self.dec_im = self.get_wavemat()

        # 计算阈值
        T = self.get_threhold()

        # 初始化
        # LIP
        LL_H = int(self.img.shape[0] / 2 ** self.im_dim)
        LL_W = int(self.img.shape[0] / 2 ** self.im_dim)

        re = self.list_order(LL_H * 2, LL_W * 2, 0, 0)

        # 去除LIP中的重复元素 [LL_H*LL_W*4,2]
        _, idx = np.unique(self.lip_init, axis=0, return_index=True)
        LIP = self.lip_init[np.sort(idx)]

        # LIS [LL_H*LL_W*3,2]
        LIS = LIP[LIP.shape[0] // 4:, :]
        LIS_flag = np.zeros(LIS.shape[0], dtype=str)
        LIS_flag[:] = 'D'
        print(LIS_flag.shape)

        # LSP
        LSP = np.empty((0, 2), dtype=np.int8)

        # 扫描编码
        # for i in range(self.code_dim):
        sn = []
        LIP, LSP, sn = self.lip_scan(LIP, T[0], LSP, sn)
        #
        # sn, LIS, LSP, LIS_flag = self.lis_scan(T[0], sn, LIS, LSP, LIS_flag)
        lis = LIS[:2]
        lis_flag = LIS_flag[:2]
        print(lis)
        print(lis_flag)
        lsp = np.empty((0, 2), dtype=np.int8)
        sn, LIS, LSP, LIS_flag = self.lis_scan(T[0] / 2, sn, lis, LSP, lis_flag)


if __name__ == '__main__':
    img = pywt.data.camera()
    spiht = Spiht(img, 3, 3)

    # 起始时间
    startime = time.time()

    # spiht.spiht_encoder()
    spiht.spiht_encoder()
    # 结束时间
    endtime = time.time()

    print(endtime - startime)
