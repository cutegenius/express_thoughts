# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:32:50 2020

@author: 37212
"""
import numpy as np
import matplotlib.pyplot as plt
 
# data = [100, 200, 50, 300, 150, 100, 200]
# print(np.maximum.accumulate(data))  # [100 200 200 300 300 300 300]
 
data = np.random.randn(100).cumsum()
data2=np.maximum.accumulate(data)
data3=np.maximum.accumulate(data) - data


index_j = np.argmax(np.maximum.accumulate(data) - data)  # 结束位置
print(index_j)
index_i = np.argmax(data[:index_j])  # 开始位置
print(index_i)
d = data[index_j] - data[index_i]  # 最大回撤
print(d)
 
# 绘制图像
plt.plot(data)
plt.plot(data2)
plt.plot(data3)


plt.plot([index_i, index_j], [data[index_i], data[index_j]], 'o', color="r", markersize=10)
plt.show()