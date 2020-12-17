# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:04:24 2020

@author: 37212
"""
import numpy as np
import pandas as pd

#For simplicity, we will assume a risk-free rate of 0% and target return of 0%. I have imported some sample strategy returns to a dataframe labeled ‘Returns’.

rfr = 0
target = 0

df=pd.DataFrame()

df['Returns'] = df_temp[(   0,   '混合型基金')]


returns = df['Returns']
sharpe_ratio = ((returns.mean() - rfr) / returns.std())
print(sharpe_ratio)

#For the Sortino Ratio, we calculate the downside deviation of the expected returns by taking the difference between each period’s return and the target return. If a period’s return is greater than the target return, the difference is simply set to 0. Then, we square the value of the difference. Next, we calculate the average of all squared differences. The square root of the average is the downside deviation.

df['downside_returns'] = 0
df.loc[df['Returns'] < target, 'downside_returns'] = df['Returns']**2
expected_return = df['Returns'].mean()
down_stdev = np.sqrt(df['downside_returns'].mean())
sortino_ratio = (expected_return - rfr)/down_stdev
print(sortino_ratio)