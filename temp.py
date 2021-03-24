#from copy import deepcopy
#import sys
#import os
#from itertools import chain
#from functools import reduce
#from pyfinance.ols import PandasRollingOLS as rolling_ols
#from pandas.core.window import ewm
#from tushare.stock.indictor import macd
import copy
import statsmodels.api as sm
from utility.factor_data_preprocess import adjust_months, add_to_panels, align, append_df
from pyfinance.utils import rolling_windows
from utility.tool0 import Data, scaler
import utility.tool3 as tool3
from datetime import *
import pandas as pd
import numpy as np
import re
import os
from utility.relate_to_tushare import generate_months_ends
from dateutil.relativedelta import relativedelta
from utility.constant import date_dair
from utility.tool1 import CALFUNC, _calculate_su_simple, parallelcal,  lazyproperty, time_decorator, \
    get_signal_season_value, get_fill_vals, linear_interpolate, get_season_mean_value
from sqlalchemy import create_engine
import pymysql


token_path2 = r'D:\文档\OneDrive\earning money\入职后资料\token_mysql.txt'
if os.path.exists(token_path2):
    f = open(token_path2)
    token_mysql = f.read()
    f.close()


START_YEAR = 2009
try:
    os.makedirs(date_dair + './fund' + './factor_data')
    basic_path = os.path.join(date_dair, 'fund', 'factor_data')
except Exception as e:
    basic_path = os.path.join(date_dair, 'fund', 'factor_data')
data=Data()
pct_chg_of_index_price_daily=data.pct_chg_of_index_price_daily
grouped_df=data.grouped_df

fund_manager_collection = pd.read_excel(os.path.join(basic_path, 'fund_manager_collection.xlsx'), index_col=0)
fund_manager_dict = dict(zip(fund_manager_collection['manager_ID'], fund_manager_collection['fund_manager']))
type_match_dict = {
    '混合型基金': '混合基金',
    '股票型基金': '股票基金',
    '债券型基金': '债券基金',
    '货币市场型基金': '货币基金',
    '国际(QDII)基金': 'QDII基金'
}


def plug_in_index_data(t):
    '''
    函数的主要目的是按照一级投资分类，把对应的基金指数塞入投资经理指数空窗期没有数据的地方
    其中，另类投资基金没有找到合适的基金指数进行填充，暂不进行处理
    '''
    # t=grouped_df.iloc[0]
    if t['firstinvesttype'] == '另类投资基金':
        return t
    else:
        type_matched = type_match_dict[t['firstinvesttype']]
        if len(np.where(t.notnull())[0])>=3:
            first_num_ix = np.where(t.notnull())[0][2]
            last_num_ix = np.where(t.notnull())[0][-1]
            first_num_date = t.index.values[first_num_ix]
            last_num_date = t.index.values[last_num_ix]
            for ele in t.index.values:
                if isinstance(ele, pd._libs.tslibs.timestamps.Timestamp) and (ele >= first_num_date) and (
                        ele <= last_num_date):
                    if pd.isnull(t[ele]):
                        try:
                            t[ele] = pct_chg_of_index_price_daily.loc[type_matched, ele]
                        except Exception as e:
                            t[ele] = 0
                    else:
                        continue
            return t
        else:
            return t
grouped_df2 = grouped_df.apply(plug_in_index_data, axis=1)