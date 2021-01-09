# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

l_list = [1.1, 1.7, 2.3, 2.7, 2.4, 4.0]
B = np.mat([[-1, 0, 1, 0], [-1, 0, 0, 1], [0, -1, 1, 0], [0, -1, 0, 1], [0, 0, -1, 1], [0, 1, -1, 0]])
x_list = np.mat([])
l = np.mat([0, 0, 4, 3, 7, 2])
wight = 10


def compute(l_list, B, l):
    global x_list
    l_list = np.array(l_list)
    l_len = len(l_list)
    P = np.mat(np.zeros(shape=(l_len, l_len)))
    for i in range(l_len):
        P[i, i] = 10 / l_list[i]
    N = B.T * P * B
    N = N.I
    print(N)


if __name__ == '__main__':
    compute(l_list, B, l)
