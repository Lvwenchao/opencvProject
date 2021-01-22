# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

# 设置全局参数
np.set_printoptions(suppress=True, precision=1)


def compute_p(l_list, N):
    l_len = len(l_list)
    P = np.mat(np.zeros(shape=(l_len, l_len)))
    for i in range(l_len):
        P[i, i] = N / l_list[i]
    return P


# 间接平差计算
def aooe(l_list, B, L, N):
    """
    v=bx-l
    :param N: 单位权重
    :param l_list: 距离的集合
    :param B: 公式中的b
    :param L: 公式中的l
    """

    # B.T*P*B*X=B.T*P*L ，NX=W
    # X=N1.T * Q11 *W1
    P = compute_p(l_list, N)
    # 计算N,N1,N11,W1
    N = B.T * P * B
    W = B.T * P * L
    N1 = N[:-1, :]
    N11 = N1[:, :-1]
    W1 = W[:-1, :]

    # 计算Q11
    np.set_printoptions(precision=5)
    Q11 = (N1 * N1.T).I

    # 计算x
    x = N1.T * Q11 * W1

    # 计算协议数阵
    Qxx = N1.T * Q11 * N11 * Q11 * N1

    return x, Qxx


B = np.mat([[-1, 0, 1, 0], [-1, 0, 0, 1], [0, -1, 1, 0], [0, -1, 0, 1], [0, 0, -1, 1], [0, 1, -1, 0]])
l_list = [1.1, 1.7, 2.3, 2.7, 2.4, 4.0]
L = np.mat([0, 0, 4, 3, 7, 2]).T

# 第二题


if __name__ == '__main__':
    # p = np.mat(np.zeros((4, 4)))
    # print(p[:-1, :].shape)
    x, Qxx = aooe(l_list, B, L, 10)
    print("计算结果如下:")
    print("X{}\n Qxx{}".format(x, Qxx))
