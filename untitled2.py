# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:20:04 2020

@author: 37212
"""

from pyfinance.ols import PandasRollingOLS
 
results = PandasRollingOLS(model_frame.iloc[:,0], model_frame.iloc[:,-1], min(252,len(model_frame.iloc[:,0])))  # window 是滚动回归的自变量个数
results.solution  # 每一步估计的截距与斜率
results.beta  # 每一步估计的斜率
results.alpha  # 每一步估计的截距
results.predicted  # 每一步估计的样本内预测值