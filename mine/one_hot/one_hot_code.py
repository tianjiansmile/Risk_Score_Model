import pandas as pd
import numpy as np
import os
from functools import reduce

def one_hot(df,old_field):
    pass


if __name__ == '__main__':
    data = pd.read_excel('temp.xlsx')
    # 共5个变量，2个连续变量，3个分类变量
    print(data)
    # 连续变量 年纪 收入，不需要one-hot编码

