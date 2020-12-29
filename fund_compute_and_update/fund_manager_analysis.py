# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:06:42 2020

@author: 37212
"""
#from copy import deepcopy
#import sys
#import os
#from itertools import chain
#from functools import reduce
from pyfinance.ols import PandasRollingOLS as rolling_ols
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
import empyrical
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from utility.relate_to_tushare import trade_days
#For simplicity, we will assume a risk-free rate of 0% and target return of 0%.


try:
    os.makedirs(date_dair + './fund' + './factor_data')
    basic_path = os.path.join(date_dair, 'fund', 'factor_data')
except Exception as e:
    basic_path = os.path.join(date_dair, 'fund', 'factor_data')



try:
    os.makedirs(date_dair + './fund'+'./fund_manager_plots')
    plot_path = os.path.join(date_dair, 'fund','fund_manager_plots')
except Exception as e:
    plot_path = os.path.join(date_dair, 'fund','fund_manager_plots')


fund_manager_factor_dict = {
                    'fund_manager_1y_return': '基金近一年的绝对收益',
                    'fund_manager_3y_return': '基金近三年的绝对收益',
                    'fund_manager_5y_return': '基金近五年的绝对收益',
                    'fund_manager_1y_div_payout': '基金近一年的分红情况',
                    'fund_manager_3y_div_payout': '基金近三年的分红情况',
                    'fund_manager_5y_div_payout':'基金近五年的分红情况',
                    'fund_manager_alpha':'alpha',
                    'fund_manager_sharpe':'夏普比率',
                    'fund_manager_treynor':'特雷诺',
                    'fund_manager_ir':'信息比',
                    'fund_manager_sortino':'sortino',
                    'fund_manager_beta':'beta',
                    'fund_manager_r2':'R squared',
                    'fund_manager_excessreturn':'超额收益',
                    'fund_manager_prt_fundnetasset_total':'规模',
                    'fund_manager_prt_fundnetasset_total_growthrate':'年增长率',
                    'fund_manager_stddev':'年波动率',
                    'fund_manager_maxdrawdown_3m':'季度最大回撤',
                    'fund_manager_maxdrawdown_1y':'年度最大回撤',
                    'fund_manager_drawdown_1y':'回撤幅度',
                    'fund_manager_maxdrawdown_duration_1y':'回撤时长',
                    'fund_manager_managementfeeratio':'管理费',
                    'fund_manager_purchasefeeratio':'最高申购费',
                    'fund_manager_custodianfeeratio':'托管费',
                    'fund_manager_fund_corp_teamstability':'基金经理流动性',
                    'fund_manager_fund_liquidation':'基金公司合规性',
                    'fund_manager_fund_corp_fivestarfundsprop':'外部评级',
                    'fund_manager_holder_pct':'股权结构',
                    'fund_manager_prt_fundcototalnetassets':'资产管理规模',
                    'fund_manager_education':'学历',
                    'fund_manager_fund_averageworkingyears':'从业经验',
                    'fund_manager_fund_manager_awardrecord':'研究深度',
                    'fund_manager_holder_mngemp_holdingpct':'激励机制',
                    'fund_manager_risk_annutrackerror':'跟踪误差',
                    'fund_manager_style_stylecoefficient':'投资风格',
                    'fund_manager_prt_hkstocktonav':'投资地域敞口',
                    'fund_manager_prt_stocktonav':'仓位管理',
                    'fund_manager_prt_fundnoofsecurities':'底层资产研究',
                    'fund_manager_rating_shanghaioverall3y':'透明度',
                    'fund_manager_style_averagepositiontime':'流动性',
                    'fund_manager_prt_corporatebondtobond':'交易对手风险',
                    'fund_manager_prt_topsectosec':'市场波动风险',
                    'fund_manager_1y_periodreturnranking':'相较同业表现',
                    'fund_manager_1y_periodreturnranking_type':'相较基金经理其他产品类型表现',
                    'fund_manager_employee_board':'董事会人数',
                    'fund_manager_fund_corp_fundmanagermaturity':'员工素质',
                    'fund_manager_corp_productnopermanager':'基金公司人均管理产品数',
                    'fund_manager_regcapital':'注册资本',
                    'fund_manager_qanal_totalincome':'财务状况',
                    'fund_manager_prt_totalasset':'基金总资产值',
                    'fund_manager_div_accumulatedperunit':'基金历年分红情况',
                    'fund_manager_total_expense':'基金各种费用及成本'
                    }
d_freq_factor_list=[
    'fund_manager_alpha',
    'fund_manager_sharpe',
    'fund_manager_treynor',
    'fund_manager_ir',
    'fund_manager_sortino',
    'fund_manager_beta',
    'fund_manager_r2',
    'fund_manager_excessreturn',
    'fund_manager_stddev',
    'fund_manager_maxdrawdown_3m',
    'fund_manager_maxdrawdown_1y',
    'fund_manager_drawdown_1y',
    'fund_manager_maxdrawdown_duration_1y',
    'fund_manager_managementfeeratio',
    'fund_manager_purchasefeeratio',
    'fund_manager_custodianfeeratio',
    'fund_manager_fund_corp_fivestarfundsprop',
    'fund_manager_holder_pct',
    'fund_manager_prt_fundcototalnetassets',
    'fund_manager_education',
    'fund_manager_fund_averageworkingyears',
    'fund_manager_fund_manager_awardrecord',
    'fund_manager_employee_board',
    'fund_manager_fund_corp_fundmanagermaturity',
    'fund_manager_corp_productnopermanager',
    'fund_manager_regcapital'
    ]


def correlation(x, y):
    x_columns=x.columns.values.tolist()
    y_columns=y.columns.values.tolist()
    #cond = [val for val in x_columns if val in y_columns]
    cond=list(set(x_columns).intersection(set(y_columns)))
    co=pd.Series()
    for val in cond:
        x_p=x.loc[:,val]
        y_p=y.loc[:,val]
        temp=x_p.corr(y_p)
        co[val]=temp
    return np.mean(co)

def c_index(df):
    if isinstance(df.columns,pd.core.indexes.datetimes.DatetimeIndex):
        df.columns=df.columns.astype('str')
        return df
    else:
        t=df.columns.astype('str').values.tolist()
        t_cut=[col[0:10] for col in t]
        df.columns=t_cut
        return df







class Factor_Manager_Analysis(CALFUNC):

    def __init__(self):
        super().__init__()
        self._mes = generate_months_ends()

        
    @staticmethod
    def corr_matrix_typestock():
        data=Data()
        corr_matrix=pd.DataFrame()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        #exemption_list=['fund_manager_education',]考虑不把有序变量标准化，但是觉得同样不合理这样
        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=factor1.loc['股票型基金']
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1=c_index(factor1)
                    t=correlation(factor1, factor1)
                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                else:
                    factor1=c_index(factor1)
                    t=correlation(factor1, factor1)
                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
            else:
                #factor=tool3.cleaning(factor)
                factor1=c_index(factor1)
                t=correlation(factor1, factor1)
                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
            if i >0:
                for j in range(i):
                    key2=keys[j]
                    factor=eval('data.' + key2)
                    #factor = eval('self.' + key2)
                    factor2=factor.copy()
                    if 'firstinvesttype' in factor2.columns.values : 
                        factor2.reset_index(inplace=True)
                        factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                        factor2=factor2.loc['股票型基金']
                        if key2 in d_freq_factor_list:
                            factor2=tool3.d_freq_to_m_freq(factor2)
                            factor2=c_index(factor2)
                            #factor1=tool3.cleaning(factor1)
                            tt=correlation(factor1, factor2)
                            corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                            corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                        else:
                            #factor1=tool3.cleaning(factor1)
                            factor2=c_index(factor2)
                            tt=correlation(factor1, factor2)
                            corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                            corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                    else:
                        if key2 in d_freq_factor_list:
                            factor2=tool3.d_freq_to_m_freq(factor2)
                            factor2=c_index(factor2)
                            #factor1=tool3.cleaning(factor1)
                            tt=correlation(factor1, factor2)
                            corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                            corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                        else:
                            #factor1=tool3.cleaning(factor1)
                            factor2=c_index(factor2)
                            tt=correlation(factor1, factor2)
                            corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                            corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            
        corr_matrix.to_excel(os.path.join(plot_path, 'corr_matrix_typestock.xlsx'))




    @staticmethod
    def corr_matrix_typeQDII():
        data=Data()
        corr_matrix=pd.DataFrame()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        #exemption_list=['fund_manager_education',]考虑不把有序变量标准化，但是觉得同样不合理这样
        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                try:
                    factor1=factor1.loc['国际(QDII)基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    else:
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            factor=eval('data.' + key2)
                            #factor = eval('self.' + key2)
                            factor2=factor.copy()
                            if 'firstinvesttype' in factor2.columns.values : 
                                factor2.reset_index(inplace=True)
                                factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                                try:
                                    factor2=factor2.loc['国际(QDII)基金']
                                    if key2 in d_freq_factor_list:
                                        factor2=tool3.d_freq_to_m_freq(factor2)
                                        factor2=c_index(factor2)
                                        #factor1=tool3.cleaning(factor1)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                    else:
                                        #factor1=tool3.cleaning(factor1)
                                        factor2=c_index(factor2)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                except Exception as e:
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                            else:
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                except Exception as e:
                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=np.nan
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                            corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
            else:
                #factor=tool3.cleaning(factor)
                factor1=c_index(factor1)
                t=correlation(factor1, factor1)
                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                if i >0:
                    for j in range(i):
                        key2=keys[j]
                        factor=eval('data.' + key2)
                        #factor = eval('self.' + key2)
                        factor2=factor.copy()
                        if 'firstinvesttype' in factor2.columns.values : 
                            factor2.reset_index(inplace=True)
                            factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                            try:
                                factor2=factor2.loc['国际(QDII)基金']
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            except Exception as e:
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                        else:
                            if key2 in d_freq_factor_list:
                                factor2=tool3.d_freq_to_m_freq(factor2)
                                factor2=c_index(factor2)
                                #factor1=tool3.cleaning(factor1)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            else:
                                #factor1=tool3.cleaning(factor1)
                                factor2=c_index(factor2)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
            
        corr_matrix.to_excel(os.path.join(plot_path, 'corr_matrix_typeQDII.xlsx'))




    @staticmethod
    def corr_matrix_typealternative():
        data=Data()
        corr_matrix=pd.DataFrame()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        #exemption_list=['fund_manager_education',]考虑不把有序变量标准化，但是觉得同样不合理这样
        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                try:
                    factor1=factor1.loc['另类投资基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    else:
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            factor=eval('data.' + key2)
                            #factor = eval('self.' + key2)
                            factor2=factor.copy()
                            if 'firstinvesttype' in factor2.columns.values : 
                                factor2.reset_index(inplace=True)
                                factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                                try:
                                    factor2=factor2.loc['另类投资基金']
                                    if key2 in d_freq_factor_list:
                                        factor2=tool3.d_freq_to_m_freq(factor2)
                                        factor2=c_index(factor2)
                                        #factor1=tool3.cleaning(factor1)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                    else:
                                        #factor1=tool3.cleaning(factor1)
                                        factor2=c_index(factor2)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                except Exception as e:
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                            else:
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                except Exception as e:
                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=np.nan
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                            corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
            else:
                #factor=tool3.cleaning(factor)
                factor1=c_index(factor1)
                t=correlation(factor1, factor1)
                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                if i >0:
                    for j in range(i):
                        key2=keys[j]
                        factor=eval('data.' + key2)
                        #factor = eval('self.' + key2)
                        factor2=factor.copy()
                        if 'firstinvesttype' in factor2.columns.values : 
                            factor2.reset_index(inplace=True)
                            factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                            try:
                                factor2=factor2.loc['另类投资基金']
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            except Exception as e:
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                        else:
                            if key2 in d_freq_factor_list:
                                factor2=tool3.d_freq_to_m_freq(factor2)
                                factor2=c_index(factor2)
                                #factor1=tool3.cleaning(factor1)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            else:
                                #factor1=tool3.cleaning(factor1)
                                factor2=c_index(factor2)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
            
        corr_matrix.to_excel(os.path.join(plot_path, 'corr_matrix_typealternative.xlsx'))



    @staticmethod
    def corr_matrix_typebond():
        data=Data()
        corr_matrix=pd.DataFrame()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        #exemption_list=['fund_manager_education',]考虑不把有序变量标准化，但是觉得同样不合理这样
        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                try:
                    factor1=factor1.loc['债券型基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    else:
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            factor=eval('data.' + key2)
                            #factor = eval('self.' + key2)
                            factor2=factor.copy()
                            if 'firstinvesttype' in factor2.columns.values : 
                                factor2.reset_index(inplace=True)
                                factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                                try:
                                    factor2=factor2.loc['债券型基金']
                                    if key2 in d_freq_factor_list:
                                        factor2=tool3.d_freq_to_m_freq(factor2)
                                        factor2=c_index(factor2)
                                        #factor1=tool3.cleaning(factor1)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                    else:
                                        #factor1=tool3.cleaning(factor1)
                                        factor2=c_index(factor2)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                except Exception as e:
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                            else:
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                except Exception as e:
                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=np.nan
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                            corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
            else:
                #factor=tool3.cleaning(factor)
                factor1=c_index(factor1)
                t=correlation(factor1, factor1)
                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                if i >0:
                    for j in range(i):
                        key2=keys[j]
                        factor=eval('data.' + key2)
                        #factor = eval('self.' + key2)
                        factor2=factor.copy()
                        if 'firstinvesttype' in factor2.columns.values : 
                            factor2.reset_index(inplace=True)
                            factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                            try:
                                factor2=factor2.loc['债券型基金']
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            except Exception as e:
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                        else:
                            if key2 in d_freq_factor_list:
                                factor2=tool3.d_freq_to_m_freq(factor2)
                                factor2=c_index(factor2)
                                #factor1=tool3.cleaning(factor1)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            else:
                                #factor1=tool3.cleaning(factor1)
                                factor2=c_index(factor2)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
            
        corr_matrix.to_excel(os.path.join(plot_path, 'corr_matrix_typebond.xlsx'))






    @staticmethod
    def corr_matrix_typehybrid():
        data=Data()
        corr_matrix=pd.DataFrame()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        #exemption_list=['fund_manager_education',]考虑不把有序变量标准化，但是觉得同样不合理这样
        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                try:
                    factor1=factor1.loc['混合型基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    else:
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            factor=eval('data.' + key2)
                            #factor = eval('self.' + key2)
                            factor2=factor.copy()
                            if 'firstinvesttype' in factor2.columns.values : 
                                factor2.reset_index(inplace=True)
                                factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                                try:
                                    factor2=factor2.loc['混合型基金']
                                    if key2 in d_freq_factor_list:
                                        factor2=tool3.d_freq_to_m_freq(factor2)
                                        factor2=c_index(factor2)
                                        #factor1=tool3.cleaning(factor1)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                    else:
                                        #factor1=tool3.cleaning(factor1)
                                        factor2=c_index(factor2)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                except Exception as e:
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                            else:
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                except Exception as e:
                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=np.nan
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                            corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
            else:
                #factor=tool3.cleaning(factor)
                factor1=c_index(factor1)
                t=correlation(factor1, factor1)
                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                if i >0:
                    for j in range(i):
                        key2=keys[j]
                        factor=eval('data.' + key2)
                        #factor = eval('self.' + key2)
                        factor2=factor.copy()
                        if 'firstinvesttype' in factor2.columns.values : 
                            factor2.reset_index(inplace=True)
                            factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                            try:
                                factor2=factor2.loc['混合型基金']
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            except Exception as e:
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                        else:
                            if key2 in d_freq_factor_list:
                                factor2=tool3.d_freq_to_m_freq(factor2)
                                factor2=c_index(factor2)
                                #factor1=tool3.cleaning(factor1)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            else:
                                #factor1=tool3.cleaning(factor1)
                                factor2=c_index(factor2)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
            
        corr_matrix.to_excel(os.path.join(plot_path, 'corr_matrix_typehybrid.xlsx'))




    @staticmethod
    def corr_matrix_typemoney():
        data=Data()
        corr_matrix=pd.DataFrame()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        #exemption_list=['fund_manager_education',]考虑不把有序变量标准化，但是觉得同样不合理这样
        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                try:
                    factor1=factor1.loc['货币市场型基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    else:
                        factor1=c_index(factor1)
                        t=correlation(factor1, factor1)
                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            factor=eval('data.' + key2)
                            #factor = eval('self.' + key2)
                            factor2=factor.copy()
                            if 'firstinvesttype' in factor2.columns.values : 
                                factor2.reset_index(inplace=True)
                                factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                                try:
                                    factor2=factor2.loc['货币市场型基金']
                                    if key2 in d_freq_factor_list:
                                        factor2=tool3.d_freq_to_m_freq(factor2)
                                        factor2=c_index(factor2)
                                        #factor1=tool3.cleaning(factor1)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                    else:
                                        #factor1=tool3.cleaning(factor1)
                                        factor2=c_index(factor2)
                                        tt=correlation(factor1, factor2)
                                        corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                        corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                except Exception as e:
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                            else:
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                except Exception as e:
                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=np.nan
                    if i >0:
                        for j in range(i):
                            key2=keys[j]
                            corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                            corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
            else:
                #factor=tool3.cleaning(factor)
                factor1=c_index(factor1)
                t=correlation(factor1, factor1)
                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key1]]=t
                if i >0:
                    for j in range(i):
                        key2=keys[j]
                        factor=eval('data.' + key2)
                        #factor = eval('self.' + key2)
                        factor2=factor.copy()
                        if 'firstinvesttype' in factor2.columns.values : 
                            factor2.reset_index(inplace=True)
                            factor2.set_index(['firstinvesttype','manager_ID'],inplace=True)
                            try:
                                factor2=factor2.loc['货币市场型基金']
                                if key2 in d_freq_factor_list:
                                    factor2=tool3.d_freq_to_m_freq(factor2)
                                    factor2=c_index(factor2)
                                    #factor1=tool3.cleaning(factor1)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                                else:
                                    #factor1=tool3.cleaning(factor1)
                                    factor2=c_index(factor2)
                                    tt=correlation(factor1, factor2)
                                    corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                    corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            except Exception as e:
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=np.nan
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=np.nan
                        else:
                            if key2 in d_freq_factor_list:
                                factor2=tool3.d_freq_to_m_freq(factor2)
                                factor2=c_index(factor2)
                                #factor1=tool3.cleaning(factor1)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
                            else:
                                #factor1=tool3.cleaning(factor1)
                                factor2=c_index(factor2)
                                tt=correlation(factor1, factor2)
                                corr_matrix.loc[fund_manager_factor_dict[key1],fund_manager_factor_dict[key2]]=tt
                                corr_matrix.loc[fund_manager_factor_dict[key2],fund_manager_factor_dict[key1]]=tt
            
            corr_matrix.to_excel(os.path.join(plot_path, 'corr_matrix_typemoney.xlsx'))



    @staticmethod
    def efficient_test_typeQDII(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['国际(QDII)基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['国际(QDII)基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['国际(QDII)基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['国际(QDII)基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['国际(QDII)基金',month_end]
            for i in range(dict_len):
                key=keys[i]
                try:
                    t=fund_manager_factor_dict[key][month_end]
                    IC_3m=t.corr(n3m)
                    IC_6m=t.corr(n6m)
                    IC_1y=t.corr(n1y)
                    IC_2y=t.corr(n2y)
                    IC_dataframe.loc[key,'3m']=IC_3m
                    IC_dataframe.loc[key,'6m']=IC_6m
                    IC_dataframe.loc[key,'1y']=IC_1y
                    IC_dataframe.loc[key,'2y']=IC_2y
                except Exception as e:
                    IC_dataframe.loc[key,'3m']=np.nan
                    IC_dataframe.loc[key,'6m']=np.nan
                    IC_dataframe.loc[key,'1y']=np.nan
                    IC_dataframe.loc[key,'2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        
        IC_mean.to_excel(os.path.join(plot_path, 'IC_mean_QDII.xlsx'))
        IC_std.to_excel(os.path.join(plot_path, 'IC_std_QDII.xlsx'))
        IR.to_excel(os.path.join(plot_path, 'IR_QDII.xlsx'))
        IC_win.to_excel(os.path.join(plot_path, 'IC_win_QDII.xlsx'))






    @staticmethod
    def efficient_test_typestock(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['股票型基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['股票型基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['股票型基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['股票型基金',month_end]
            for i in range(dict_len):
                key=keys[i]
                try:
                    t=fund_manager_factor_dict[key][month_end]
                    IC_3m=t.corr(n3m)
                    IC_6m=t.corr(n6m)
                    IC_1y=t.corr(n1y)
                    IC_2y=t.corr(n2y)
                    IC_dataframe.loc[key,'3m']=IC_3m
                    IC_dataframe.loc[key,'6m']=IC_6m
                    IC_dataframe.loc[key,'1y']=IC_1y
                    IC_dataframe.loc[key,'2y']=IC_2y
                except Exception as e:
                    IC_dataframe.loc[key,'3m']=np.nan
                    IC_dataframe.loc[key,'6m']=np.nan
                    IC_dataframe.loc[key,'1y']=np.nan
                    IC_dataframe.loc[key,'2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        
        IC_mean.to_excel(os.path.join(plot_path, 'IC_mean_stock.xlsx'))
        IC_std.to_excel(os.path.join(plot_path, 'IC_std_stock.xlsx'))
        IR.to_excel(os.path.join(plot_path, 'IR_stock.xlsx'))
        IC_win.to_excel(os.path.join(plot_path, 'IC_win_stock.xlsx'))




    @staticmethod
    def efficient_test_typealternative(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['另类投资基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['另类投资基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['另类投资基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['另类投资基金',month_end]
            for i in range(dict_len):
                key=keys[i]
                try:
                    t=fund_manager_factor_dict[key][month_end]
                    IC_3m=t.corr(n3m)
                    IC_6m=t.corr(n6m)
                    IC_1y=t.corr(n1y)
                    IC_2y=t.corr(n2y)
                    IC_dataframe.loc[key,'3m']=IC_3m
                    IC_dataframe.loc[key,'6m']=IC_6m
                    IC_dataframe.loc[key,'1y']=IC_1y
                    IC_dataframe.loc[key,'2y']=IC_2y
                except Exception as e:
                    IC_dataframe.loc[key,'3m']=np.nan
                    IC_dataframe.loc[key,'6m']=np.nan
                    IC_dataframe.loc[key,'1y']=np.nan
                    IC_dataframe.loc[key,'2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        
        IC_mean.to_excel(os.path.join(plot_path, 'IC_mean_alternative.xlsx'))
        IC_std.to_excel(os.path.join(plot_path, 'IC_std_alternative.xlsx'))
        IR.to_excel(os.path.join(plot_path, 'IR_alternative.xlsx'))
        IC_win.to_excel(os.path.join(plot_path, 'IC_win_alternative.xlsx'))





    @staticmethod
    def efficient_test_typebond(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['债券型基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['债券型基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['债券型基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['债券型基金',month_end]
            for i in range(dict_len):
                key=keys[i]
                try:
                    t=fund_manager_factor_dict[key][month_end]
                    IC_3m=t.corr(n3m)
                    IC_6m=t.corr(n6m)
                    IC_1y=t.corr(n1y)
                    IC_2y=t.corr(n2y)
                    IC_dataframe.loc[key,'3m']=IC_3m
                    IC_dataframe.loc[key,'6m']=IC_6m
                    IC_dataframe.loc[key,'1y']=IC_1y
                    IC_dataframe.loc[key,'2y']=IC_2y
                except Exception as e:
                    IC_dataframe.loc[key,'3m']=np.nan
                    IC_dataframe.loc[key,'6m']=np.nan
                    IC_dataframe.loc[key,'1y']=np.nan
                    IC_dataframe.loc[key,'2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        
        IC_mean.to_excel(os.path.join(plot_path, 'IC_mean_bond.xlsx'))
        IC_std.to_excel(os.path.join(plot_path, 'IC_std_bond.xlsx'))
        IR.to_excel(os.path.join(plot_path, 'IR_bond.xlsx'))
        IC_win.to_excel(os.path.join(plot_path, 'IC_win_bond.xlsx'))











    @staticmethod
    def efficient_test_typehybird(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['混合型基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['混合型基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['混合型基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['混合型基金',month_end]
            for i in range(dict_len):
                key=keys[i]
                try:
                    t=fund_manager_factor_dict[key][month_end]
                    IC_3m=t.corr(n3m)
                    IC_6m=t.corr(n6m)
                    IC_1y=t.corr(n1y)
                    IC_2y=t.corr(n2y)
                    IC_dataframe.loc[key,'3m']=IC_3m
                    IC_dataframe.loc[key,'6m']=IC_6m
                    IC_dataframe.loc[key,'1y']=IC_1y
                    IC_dataframe.loc[key,'2y']=IC_2y
                except Exception as e:
                    IC_dataframe.loc[key,'3m']=np.nan
                    IC_dataframe.loc[key,'6m']=np.nan
                    IC_dataframe.loc[key,'1y']=np.nan
                    IC_dataframe.loc[key,'2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        
        IC_mean.to_excel(os.path.join(plot_path, 'IC_mean_hybrid.xlsx'))
        IC_std.to_excel(os.path.join(plot_path, 'IC_std_hybrid.xlsx'))
        IR.to_excel(os.path.join(plot_path, 'IR_hybrid.xlsx'))
        IC_win.to_excel(os.path.join(plot_path, 'IC_win_hybrid.xlsx'))






    @staticmethod
    def efficient_test_typemoney(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['货币市场型基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['货币市场型基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['货币市场型基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['货币市场型基金',month_end]
            for i in range(dict_len):
                key=keys[i]
                try:
                    t=fund_manager_factor_dict[key][month_end]
                    IC_3m=t.corr(n3m)
                    IC_6m=t.corr(n6m)
                    IC_1y=t.corr(n1y)
                    IC_2y=t.corr(n2y)
                    IC_dataframe.loc[key,'3m']=IC_3m
                    IC_dataframe.loc[key,'6m']=IC_6m
                    IC_dataframe.loc[key,'1y']=IC_1y
                    IC_dataframe.loc[key,'2y']=IC_2y
                except Exception as e:
                    IC_dataframe.loc[key,'3m']=np.nan
                    IC_dataframe.loc[key,'6m']=np.nan
                    IC_dataframe.loc[key,'1y']=np.nan
                    IC_dataframe.loc[key,'2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        
        IC_mean.to_excel(os.path.join(plot_path, 'IC_mean_money.xlsx'))
        IC_std.to_excel(os.path.join(plot_path, 'IC_std_money.xlsx'))
        IR.to_excel(os.path.join(plot_path, 'IR_money.xlsx'))
        IC_win.to_excel(os.path.join(plot_path, 'IC_win_money.xlsx'))






    @staticmethod
    def combinedfactor_test_typeQDII(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['国际(QDII)基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        combinedfactor_QDII=(((((((fund_manager_factor_dict['fund_manager_3y_return']*(1/9)).sub(fund_manager_factor_dict['fund_manager_5y_div_payout']*(1/9),fill_value=0)).add(fund_manager_factor_dict['fund_manager_alpha']*(1/9),fill_value=0).sub(fund_manager_factor_dict['fund_manager_sortino']*(1/9),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_fundnetasset_total']*(1/9),fill_value=0)).add(fund_manager_factor_dict['fund_manager_prt_stocktonav']*(1/9),fill_value=0)).add(fund_manager_factor_dict['fund_manager_prt_fundnoofsecurities']*(1/9),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_employee_board']*(1/9),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_totalasset']*(1/9),fill_value=0)
        
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['国际(QDII)基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['国际(QDII)基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['国际(QDII)基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['国际(QDII)基金',month_end]
            try:
                t=combinedfactor_QDII[month_end]
                IC_3m=t.corr(n3m)
                IC_6m=t.corr(n6m)
                IC_1y=t.corr(n1y)
                IC_2y=t.corr(n2y)
                IC_dataframe.loc['combinedfactor_QDII','3m']=IC_3m
                IC_dataframe.loc['combinedfactor_QDII','6m']=IC_6m
                IC_dataframe.loc['combinedfactor_QDII','1y']=IC_1y
                IC_dataframe.loc['combinedfactor_QDII','2y']=IC_2y
            except Exception as e:
                IC_dataframe.loc['combinedfactor_QDII','3m']=np.nan
                IC_dataframe.loc['combinedfactor_QDII','6m']=np.nan
                IC_dataframe.loc['combinedfactor_QDII','1y']=np.nan
                IC_dataframe.loc['combinedfactor_QDII','2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        #IC_max=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        #IR_max=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        combinedfactortest_QDII=pd.DataFrame()
        combinedfactortest_QDII['IC均值']=IC_mean.T['combinedfactor_QDII']
        combinedfactortest_QDII['IC标准差']=IC_std.T['combinedfactor_QDII']
        combinedfactortest_QDII['IR值']=IR.T['combinedfactor_QDII']
        combinedfactortest_QDII['IC>0']=IC_win.T['combinedfactor_QDII']
        
        
        combinedfactor_IC_QDII=pd.DataFrame()
        combinedfactor_IC_QDII['3m']=IC_3m_dataframe.T['combinedfactor_QDII']
        combinedfactor_IC_QDII['6m']=IC_6m_dataframe.T['combinedfactor_QDII']
        combinedfactor_IC_QDII['1y']=IC_1y_dataframe.T['combinedfactor_QDII']
        combinedfactor_IC_QDII['2y']=IC_2y_dataframe.T['combinedfactor_QDII']
        
        
        combinedfactor_QDII.to_excel(os.path.join(basic_path, 'combinedfactor_QDII.xlsx'))
        combinedfactortest_QDII.to_excel(os.path.join(plot_path, 'combinedfactortest_QDII.xlsx'))
        combinedfactor_IC_QDII.to_excel(os.path.join(plot_path, 'combinedfactor_IC_QDII.xlsx'))
        








    @staticmethod
    def combinedfactor_test_typestock(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['股票型基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        combinedfactor_stock=(((((((((((((fund_manager_factor_dict['fund_manager_5y_return']*(1/14)).sub(fund_manager_factor_dict['fund_manager_5y_div_payout']*(1/14),fill_value=0)).add(fund_manager_factor_dict['fund_manager_sortino']*(1/14),fill_value=0)).add(fund_manager_factor_dict['fund_manager_r2']*(1/14),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_maxdrawdown_1y']*(1/14),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_risk_annutrackerror']*(1/14),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_style_averagepositiontime']*(1/14),fill_value=0)).add(fund_manager_factor_dict['fund_manager_prt_topsectosec']*(1/14),fill_value=0)).add(fund_manager_factor_dict['fund_manager_employee_board']*(1/14),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_corp_productnopermanager']*(1/14),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_regcapital']*(1/14),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_totalasset']*(1/14),fill_value=0)).add(fund_manager_factor_dict['fund_manager_div_accumulatedperunit']*(1/14),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_total_expense']*(1/14),fill_value=0)
        
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['股票型基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['股票型基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['股票型基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['股票型基金',month_end]
            try:
                t=combinedfactor_stock[month_end]
                IC_3m=t.corr(n3m)
                IC_6m=t.corr(n6m)
                IC_1y=t.corr(n1y)
                IC_2y=t.corr(n2y)
                IC_dataframe.loc['combinedfactor_stock','3m']=IC_3m
                IC_dataframe.loc['combinedfactor_stock','6m']=IC_6m
                IC_dataframe.loc['combinedfactor_stock','1y']=IC_1y
                IC_dataframe.loc['combinedfactor_stock','2y']=IC_2y
            except Exception as e:
                IC_dataframe.loc['combinedfactor_stock','3m']=np.nan
                IC_dataframe.loc['combinedfactor_stock','6m']=np.nan
                IC_dataframe.loc['combinedfactor_stock','1y']=np.nan
                IC_dataframe.loc['combinedfactor_stock','2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        #IC_max=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        #IR_max=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        combinedfactortest_stock=pd.DataFrame()
        combinedfactortest_stock['IC均值']=IC_mean.T['combinedfactor_stock']
        combinedfactortest_stock['IC标准差']=IC_std.T['combinedfactor_stock']
        combinedfactortest_stock['IR值']=IR.T['combinedfactor_stock']
        combinedfactortest_stock['IC>0']=IC_win.T['combinedfactor_stock']
        
        
        combinedfactor_IC_stock=pd.DataFrame()
        combinedfactor_IC_stock['3m']=IC_3m_dataframe.T['combinedfactor_stock']
        combinedfactor_IC_stock['6m']=IC_6m_dataframe.T['combinedfactor_stock']
        combinedfactor_IC_stock['1y']=IC_1y_dataframe.T['combinedfactor_stock']
        combinedfactor_IC_stock['2y']=IC_2y_dataframe.T['combinedfactor_stock']
        
        
        combinedfactor_stock.to_excel(os.path.join(basic_path, 'combinedfactor_stock.xlsx'))
        combinedfactortest_stock.to_excel(os.path.join(plot_path, 'combinedfactortest_stock.xlsx'))
        combinedfactor_IC_stock.to_excel(os.path.join(plot_path, 'combinedfactor_IC_stock.xlsx'))
        
        








    @staticmethod
    def combinedfactor_test_typebond(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['债券型基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        combinedfactor_bond=((((((((((((((-fund_manager_factor_dict['fund_manager_3y_return']*(1/15)).add(fund_manager_factor_dict['fund_manager_5y_return']*(1/15),fill_value=0)).add(fund_manager_factor_dict['fund_manager_alpha']*(1/15),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_beta']*(1/15),fill_value=0)).add(fund_manager_factor_dict['fund_manager_r2']*(1/15),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_maxdrawdown_3m']*(1/15),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_maxdrawdown_duration_1y']*(1/15),fill_value=0)).add(fund_manager_factor_dict['fund_manager_managementfeeratio']*(1/15),fill_value=0)).add(fund_manager_factor_dict['fund_manager_purchasefeeratio']*(1/15),fill_value=0)).add(fund_manager_factor_dict['fund_manager_risk_annutrackerror']*(1/15),fill_value=0)).add(fund_manager_factor_dict['fund_manager_prt_stocktonav']*(1/15),fill_value=0)).add(fund_manager_factor_dict['fund_manager_corp_productnopermanager']*(1/15),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_regcapital']*(1/15),fill_value=0)).add(fund_manager_factor_dict['fund_manager_qanal_totalincome']*(1/15),fill_value=0)).add(fund_manager_factor_dict['fund_manager_total_expense']*(1/15),fill_value=0)
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['债券型基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['债券型基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['债券型基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['债券型基金',month_end]
            try:
                t=combinedfactor_bond[month_end]
                IC_3m=t.corr(n3m)
                IC_6m=t.corr(n6m)
                IC_1y=t.corr(n1y)
                IC_2y=t.corr(n2y)
                IC_dataframe.loc['combinedfactor_bond','3m']=IC_3m
                IC_dataframe.loc['combinedfactor_bond','6m']=IC_6m
                IC_dataframe.loc['combinedfactor_bond','1y']=IC_1y
                IC_dataframe.loc['combinedfactor_bond','2y']=IC_2y
            except Exception as e:
                IC_dataframe.loc['combinedfactor_bond','3m']=np.nan
                IC_dataframe.loc['combinedfactor_bond','6m']=np.nan
                IC_dataframe.loc['combinedfactor_bond','1y']=np.nan
                IC_dataframe.loc['combinedfactor_bond','2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        #IC_max=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        #IR_max=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        combinedfactortest_bond=pd.DataFrame()
        combinedfactortest_bond['IC均值']=IC_mean.T['combinedfactor_bond']
        combinedfactortest_bond['IC标准差']=IC_std.T['combinedfactor_bond']
        combinedfactortest_bond['IR值']=IR.T['combinedfactor_bond']
        combinedfactortest_bond['IC>0']=IC_win.T['combinedfactor_bond']
        
        
        combinedfactor_IC_bond=pd.DataFrame()
        combinedfactor_IC_bond['3m']=IC_3m_dataframe.T['combinedfactor_bond']
        combinedfactor_IC_bond['6m']=IC_6m_dataframe.T['combinedfactor_bond']
        combinedfactor_IC_bond['1y']=IC_1y_dataframe.T['combinedfactor_bond']
        combinedfactor_IC_bond['2y']=IC_2y_dataframe.T['combinedfactor_bond']
        
        
        combinedfactor_bond.to_excel(os.path.join(basic_path, 'combinedfactor_bond.xlsx'))
        combinedfactortest_bond.to_excel(os.path.join(plot_path, 'combinedfactortest_bond.xlsx'))
        combinedfactor_IC_bond.to_excel(os.path.join(plot_path, 'combinedfactor_IC_bond.xlsx'))
        







    @staticmethod
    def combinedfactor_test_typehybrid(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['混合型基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        combinedfactor_hybrid=(((((((((((fund_manager_factor_dict['fund_manager_3y_return']*(1/11)).sub(fund_manager_factor_dict['fund_manager_r2']*(1/11),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_fundnetasset_total']*(1/11),fill_value=0)).add(fund_manager_factor_dict['fund_manager_stddev']*(1/11),fill_value=0)).add(fund_manager_factor_dict['fund_manager_maxdrawdown_duration_1y']*(1/11),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_liquidation']*(1/11),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_corp_fivestarfundsprop']*(1/11),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_holder_pct']*(1/11),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_averageworkingyears']*(1/11),fill_value=0)).add(fund_manager_factor_dict['fund_manager_prt_stocktonav']*(1/11),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_1y_periodreturnranking']*(1/11),fill_value=0)).add(fund_manager_factor_dict['fund_manager_total_expense']*(1/11),fill_value=0)
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['混合型基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['混合型基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['混合型基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['混合型基金',month_end]
            try:
                t=combinedfactor_hybrid[month_end]
                IC_3m=t.corr(n3m)
                IC_6m=t.corr(n6m)
                IC_1y=t.corr(n1y)
                IC_2y=t.corr(n2y)
                IC_dataframe.loc['combinedfactor_hybrid','3m']=IC_3m
                IC_dataframe.loc['combinedfactor_hybrid','6m']=IC_6m
                IC_dataframe.loc['combinedfactor_hybrid','1y']=IC_1y
                IC_dataframe.loc['combinedfactor_hybrid','2y']=IC_2y
            except Exception as e:
                IC_dataframe.loc['combinedfactor_hybrid','3m']=np.nan
                IC_dataframe.loc['combinedfactor_hybrid','6m']=np.nan
                IC_dataframe.loc['combinedfactor_hybrid','1y']=np.nan
                IC_dataframe.loc['combinedfactor_hybrid','2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        #IC_max=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        #IR_max=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        combinedfactortest_hybrid=pd.DataFrame()
        combinedfactortest_hybrid['IC均值']=IC_mean.T['combinedfactor_hybrid']
        combinedfactortest_hybrid['IC标准差']=IC_std.T['combinedfactor_hybrid']
        combinedfactortest_hybrid['IR值']=IR.T['combinedfactor_hybrid']
        combinedfactortest_hybrid['IC>0']=IC_win.T['combinedfactor_hybrid']
        
        
        combinedfactor_IC_hybrid=pd.DataFrame()
        combinedfactor_IC_hybrid['3m']=IC_3m_dataframe.T['combinedfactor_hybrid']
        combinedfactor_IC_hybrid['6m']=IC_6m_dataframe.T['combinedfactor_hybrid']
        combinedfactor_IC_hybrid['1y']=IC_1y_dataframe.T['combinedfactor_hybrid']
        combinedfactor_IC_hybrid['2y']=IC_2y_dataframe.T['combinedfactor_hybrid']
        
        
        combinedfactor_hybrid.to_excel(os.path.join(basic_path, 'combinedfactor_hybrid.xlsx'))
        combinedfactortest_hybrid.to_excel(os.path.join(plot_path, 'combinedfactortest_hybrid.xlsx'))
        combinedfactor_IC_hybrid.to_excel(os.path.join(plot_path, 'combinedfactor_IC_hybrid.xlsx'))
        






# =============================================================================
#     @staticmethod
#     def combinedfactor_test_typemoney(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
#         data=Data()
#         start_date=datetime(year=2009, month=1, day=1)
#         end_date=end_date=datetime(year=2020, month=9, day=30)
#         IC=dict()
#         dict_len=len(fund_manager_factor_dict)
#         keys=list(fund_manager_factor_dict.keys())
#         pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
#         pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
#         pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
#         pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
#         pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         
#         pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
#         pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
#         pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
#         pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
#         
#         
#         
#         month_ends=generate_months_ends()
#         month_ends=[col for col in month_ends if col>=start_date and col<=end_date]
# 
#         for i in range(dict_len):
#             key1=keys[i]
#             factor=eval('data.' + key1)
#             #factor = eval('self.' + key1)
#             factor1=factor.copy()
#             if 'firstinvesttype' in factor1.columns.values : 
#                 factor1.reset_index(inplace=True)
#                 factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
#                 factor1=tool3.cleaning(factor1)
#                 try:
#                     factor1=factor1.loc['货币市场型基金']
#                     if key1 in d_freq_factor_list:
#                         factor1=tool3.d_freq_to_m_freq(factor1)
#                         factor1.columns=pd.DatetimeIndex(factor1.columns)
#                         fund_manager_factor_dict[key1]=factor1
#                     else:
#                         factor1.columns=pd.DatetimeIndex(factor1.columns)
#                         fund_manager_factor_dict[key1]=factor1
#                 except Exception as e:
#                     fund_manager_factor_dict[key1]=np.nan
#             else:
#                 factor1=tool3.cleaning(factor1)
#                 if key1 in d_freq_factor_list:
#                     factor1=tool3.d_freq_to_m_freq(factor1)
#                     factor1.columns=pd.DatetimeIndex(factor1.columns)
#                     fund_manager_factor_dict[key1]=factor1
#                 else:
#                     factor1.columns=pd.DatetimeIndex(factor1.columns)
#                     fund_manager_factor_dict[key1]=factor1
#                     fund_manager_factor_dict[key1]=factor1
#         combinedfactor_money=(((((((((((fund_manager_factor_dict['fund_manager_3y_return']*(1/13)).sub(fund_manager_factor_dict['fund_manager_r2']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_fundnetasset_total']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_stddev']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_maxdrawdown_duration_1y']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_liquidation']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_corp_fivestarfundsprop']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_holder_pct']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_averageworkingyears']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_prt_stocktonav']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_1y_periodreturnranking']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_total_expense']*(1/13),fill_value=0)
#         for month_end in month_ends:
#             IC_dataframe=pd.DataFrame()
#             n3m=pct_chg_of_fund_manager_index_n3m.loc['货币市场型基金',month_end]
#             n6m=pct_chg_of_fund_manager_index_n6m.loc['货币市场型基金',month_end]
#             n1y=pct_chg_of_fund_manager_index_n1y.loc['货币市场型基金',month_end]
#             n2y=pct_chg_of_fund_manager_index_n2y.loc['货币市场型基金',month_end]
#             try:
#                 t=combinedfactor_money[month_end]
#                 IC_3m=t.corr(n3m)
#                 IC_6m=t.corr(n6m)
#                 IC_1y=t.corr(n1y)
#                 IC_2y=t.corr(n2y)
#                 IC_dataframe.loc['combinedfactor_money','3m']=IC_3m
#                 IC_dataframe.loc['combinedfactor_money','6m']=IC_6m
#                 IC_dataframe.loc['combinedfactor_money','1y']=IC_1y
#                 IC_dataframe.loc['combinedfactor_money','2y']=IC_2y
#             except Exception as e:
#                 IC_dataframe.loc['combinedfactor_money','3m']=np.nan
#                 IC_dataframe.loc['combinedfactor_money','6m']=np.nan
#                 IC_dataframe.loc['combinedfactor_money','1y']=np.nan
#                 IC_dataframe.loc['combinedfactor_money','2y']=np.nan
#             IC[month_end]=IC_dataframe
#             
#         #tt=IC
#         IC_3m_dataframe=pd.DataFrame()
#         IC_6m_dataframe=pd.DataFrame()
#         IC_1y_dataframe=pd.DataFrame()
#         IC_2y_dataframe=pd.DataFrame()
#         for month_end in month_ends:
#             IC_3m_dataframe[month_end]=IC[month_end]['3m']
#             IC_6m_dataframe[month_end]=IC[month_end]['6m']
#             IC_1y_dataframe[month_end]=IC[month_end]['1y']
#             IC_2y_dataframe[month_end]=IC[month_end]['2y']
# 
# 
#         IC_mean=pd.DataFrame()
#         #IC_max=pd.DataFrame()
#         IC_std=pd.DataFrame()
#         IR=pd.DataFrame()
#         #IR_max=pd.DataFrame()
#         IC_win=pd.DataFrame()
#         
#         
#         
#         IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
#         IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
#         IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
#         IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
#         
#         
#         
#         
#         IC_std['3m']=IC_3m_dataframe.std(axis=1)
#         IC_std['6m']=IC_6m_dataframe.std(axis=1)
#         IC_std['1y']=IC_1y_dataframe.std(axis=1)
#         IC_std['2y']=IC_2y_dataframe.std(axis=1)
# 
# 
#         IR=IC_mean/IC_std
#         
#         
#         
#         IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
#         IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
#         IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
#         IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)
# 
#         
#         
#         combinedfactortest_money=pd.DataFrame()
#         combinedfactortest_money['IC均值']=IC_mean.T['combinedfactor_money']
#         combinedfactortest_money['IC标准差']=IC_std.T['combinedfactor_money']
#         combinedfactortest_money['IR值']=IR.T['combinedfactor_money']
#         combinedfactortest_money['IC>0']=IC_win.T['combinedfactor_money']
#         
#         
#         combinedfactor_IC_money=pd.DataFrame()
#         combinedfactor_IC_money['3m']=IC_3m_dataframe.T['combinedfactor_money']
#         combinedfactor_IC_money['6m']=IC_6m_dataframe.T['combinedfactor_money']
#         combinedfactor_IC_money['1y']=IC_1y_dataframe.T['combinedfactor_money']
#         combinedfactor_IC_money['2y']=IC_2y_dataframe.T['combinedfactor_money']
#         
#         
#         combinedfactor_money.to_excel(os.path.join(basic_path, 'combinedfactor_money.xlsx'))
#         combinedfactortest_money.to_excel(os.path.join(plot_path, 'combinedfactortest_money.xlsx'))
#         combinedfactor_IC_money.to_excel(os.path.join(plot_path, 'combinedfactor_IC_money.xlsx'))
#         
# =============================================================================







# =============================================================================
# 
#     #由于有效性太低被弃用的版本
#     @staticmethod
#     def combinedfactor_test_typemoney(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
#         data=Data()
#         start_date=datetime(year=2009, month=1, day=1)
#         end_date=end_date=datetime(year=2020, month=9, day=30)
#         IC=dict()
#         dict_len=len(fund_manager_factor_dict)
#         keys=list(fund_manager_factor_dict.keys())
#         pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
#         pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
#         pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
#         pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
#         pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         
#         pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
#         pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
#         pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
#         pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
#         
#         
#         
#         month_ends=generate_months_ends()
#         month_ends=[col for col in month_ends if col>=start_date and col<=end_date]
# 
#         for i in range(dict_len):
#             key1=keys[i]
#             factor=eval('data.' + key1)
#             #factor = eval('self.' + key1)
#             factor1=factor.copy()
#             if 'firstinvesttype' in factor1.columns.values : 
#                 factor1.reset_index(inplace=True)
#                 factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
#                 factor1=tool3.cleaning(factor1)
#                 try:
#                     factor1=factor1.loc['货币市场型基金']
#                     if key1 in d_freq_factor_list:
#                         factor1=tool3.d_freq_to_m_freq(factor1)
#                         factor1.columns=pd.DatetimeIndex(factor1.columns)
#                         fund_manager_factor_dict[key1]=factor1
#                     else:
#                         factor1.columns=pd.DatetimeIndex(factor1.columns)
#                         fund_manager_factor_dict[key1]=factor1
#                 except Exception as e:
#                     fund_manager_factor_dict[key1]=np.nan
#             else:
#                 factor1=tool3.cleaning(factor1)
#                 if key1 in d_freq_factor_list:
#                     factor1=tool3.d_freq_to_m_freq(factor1)
#                     factor1.columns=pd.DatetimeIndex(factor1.columns)
#                     fund_manager_factor_dict[key1]=factor1
#                 else:
#                     factor1.columns=pd.DatetimeIndex(factor1.columns)
#                     fund_manager_factor_dict[key1]=factor1
#                     fund_manager_factor_dict[key1]=factor1
#         combinedfactor_money=((((((((((((fund_manager_factor_dict['fund_manager_1y_return']*(1/13)).sub(fund_manager_factor_dict['fund_manager_treynor']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_stddev']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_maxdrawdown_duration_1y']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_custodianfeeratio']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_fund_corp_teamstability']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_fund_liquidation']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_corp_fivestarfundsprop']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_fundcototalnetassets']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_holder_mngemp_holdingpct']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_risk_annutrackerror']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_stocktonav']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_employee_board']*(1/13),fill_value=0)
#         for month_end in month_ends:
#             IC_dataframe=pd.DataFrame()
#             n3m=pct_chg_of_fund_manager_index_n3m.loc['货币市场型基金',month_end]
#             n6m=pct_chg_of_fund_manager_index_n6m.loc['货币市场型基金',month_end]
#             n1y=pct_chg_of_fund_manager_index_n1y.loc['货币市场型基金',month_end]
#             n2y=pct_chg_of_fund_manager_index_n2y.loc['货币市场型基金',month_end]
#             try:
#                 t=combinedfactor_money[month_end]
#                 IC_3m=t.corr(n3m)
#                 IC_6m=t.corr(n6m)
#                 IC_1y=t.corr(n1y)
#                 IC_2y=t.corr(n2y)
#                 IC_dataframe.loc['combinedfactor_money','3m']=IC_3m
#                 IC_dataframe.loc['combinedfactor_money','6m']=IC_6m
#                 IC_dataframe.loc['combinedfactor_money','1y']=IC_1y
#                 IC_dataframe.loc['combinedfactor_money','2y']=IC_2y
#             except Exception as e:
#                 IC_dataframe.loc['combinedfactor_money','3m']=np.nan
#                 IC_dataframe.loc['combinedfactor_money','6m']=np.nan
#                 IC_dataframe.loc['combinedfactor_money','1y']=np.nan
#                 IC_dataframe.loc['combinedfactor_money','2y']=np.nan
#             IC[month_end]=IC_dataframe
#             
#         #tt=IC
#         IC_3m_dataframe=pd.DataFrame()
#         IC_6m_dataframe=pd.DataFrame()
#         IC_1y_dataframe=pd.DataFrame()
#         IC_2y_dataframe=pd.DataFrame()
#         for month_end in month_ends:
#             IC_3m_dataframe[month_end]=IC[month_end]['3m']
#             IC_6m_dataframe[month_end]=IC[month_end]['6m']
#             IC_1y_dataframe[month_end]=IC[month_end]['1y']
#             IC_2y_dataframe[month_end]=IC[month_end]['2y']
# 
# 
#         IC_mean=pd.DataFrame()
#         #IC_max=pd.DataFrame()
#         IC_std=pd.DataFrame()
#         IR=pd.DataFrame()
#         #IR_max=pd.DataFrame()
#         IC_win=pd.DataFrame()
#         
#         
#         
#         IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
#         IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
#         IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
#         IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
#         
#         
#         
#         
#         IC_std['3m']=IC_3m_dataframe.std(axis=1)
#         IC_std['6m']=IC_6m_dataframe.std(axis=1)
#         IC_std['1y']=IC_1y_dataframe.std(axis=1)
#         IC_std['2y']=IC_2y_dataframe.std(axis=1)
# 
# 
#         IR=IC_mean/IC_std
#         
#         
#         
#         IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
#         IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
#         IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
#         IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)
# 
#         
#         
#         combinedfactortest_money=pd.DataFrame()
#         combinedfactortest_money['IC均值']=IC_mean.T['combinedfactor_money']
#         combinedfactortest_money['IC标准差']=IC_std.T['combinedfactor_money']
#         combinedfactortest_money['IR值']=IR.T['combinedfactor_money']
#         combinedfactortest_money['IC>0']=IC_win.T['combinedfactor_money']
#         
#         
#         combinedfactor_IC_money=pd.DataFrame()
#         combinedfactor_IC_money['3m']=IC_3m_dataframe.T['combinedfactor_money']
#         combinedfactor_IC_money['6m']=IC_6m_dataframe.T['combinedfactor_money']
#         combinedfactor_IC_money['1y']=IC_1y_dataframe.T['combinedfactor_money']
#         combinedfactor_IC_money['2y']=IC_2y_dataframe.T['combinedfactor_money']
#         
#         
#         combinedfactor_money.to_excel(os.path.join(basic_path, 'combinedfactor_money.xlsx'))
#         combinedfactortest_money.to_excel(os.path.join(plot_path, 'combinedfactortest_money.xlsx'))
#         combinedfactor_IC_money.to_excel(os.path.join(plot_path, 'combinedfactor_IC_money.xlsx'))
#         
# =============================================================================









    
    @staticmethod
    def combinedfactor_test_typemoney(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['货币市场型基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        combinedfactor_money=((((-fund_manager_factor_dict['fund_manager_treynor']*(1/5)).add(fund_manager_factor_dict['fund_manager_custodianfeeratio']*(1/5),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_fund_corp_teamstability']*(1/5),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_fundcototalnetassets']*(1/5),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_stocktonav']*(1/5),fill_value=0)
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['货币市场型基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['货币市场型基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['货币市场型基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['货币市场型基金',month_end]
            try:
                t=combinedfactor_money[month_end]
                IC_3m=t.corr(n3m)
                IC_6m=t.corr(n6m)
                IC_1y=t.corr(n1y)
                IC_2y=t.corr(n2y)
                IC_dataframe.loc['combinedfactor_money','3m']=IC_3m
                IC_dataframe.loc['combinedfactor_money','6m']=IC_6m
                IC_dataframe.loc['combinedfactor_money','1y']=IC_1y
                IC_dataframe.loc['combinedfactor_money','2y']=IC_2y
            except Exception as e:
                IC_dataframe.loc['combinedfactor_money','3m']=np.nan
                IC_dataframe.loc['combinedfactor_money','6m']=np.nan
                IC_dataframe.loc['combinedfactor_money','1y']=np.nan
                IC_dataframe.loc['combinedfactor_money','2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        #IC_max=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        #IR_max=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        combinedfactortest_money=pd.DataFrame()
        combinedfactortest_money['IC均值']=IC_mean.T['combinedfactor_money']
        combinedfactortest_money['IC标准差']=IC_std.T['combinedfactor_money']
        combinedfactortest_money['IR值']=IR.T['combinedfactor_money']
        combinedfactortest_money['IC>0']=IC_win.T['combinedfactor_money']
        
        
        combinedfactor_IC_money=pd.DataFrame()
        combinedfactor_IC_money['3m']=IC_3m_dataframe.T['combinedfactor_money']
        combinedfactor_IC_money['6m']=IC_6m_dataframe.T['combinedfactor_money']
        combinedfactor_IC_money['1y']=IC_1y_dataframe.T['combinedfactor_money']
        combinedfactor_IC_money['2y']=IC_2y_dataframe.T['combinedfactor_money']
        
        
        combinedfactor_money.to_excel(os.path.join(basic_path, 'combinedfactor_money.xlsx'))
        combinedfactortest_money.to_excel(os.path.join(plot_path, 'combinedfactortest_money.xlsx'))
        combinedfactor_IC_money.to_excel(os.path.join(plot_path, 'combinedfactor_IC_money.xlsx'))
        







# =============================================================================
#     #由于有效性太低被弃用的版本
#     @staticmethod
#     def combinedfactor_test_typemoney(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
#         data=Data()
#         start_date=datetime(year=2009, month=1, day=1)
#         end_date=end_date=datetime(year=2020, month=9, day=30)
#         IC=dict()
#         dict_len=len(fund_manager_factor_dict)
#         keys=list(fund_manager_factor_dict.keys())
#         pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
#         pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
#         pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
#         pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
#         pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
#         
#         pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
#         pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
#         pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
#         pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
#         
#         
#         
#         month_ends=generate_months_ends()
#         month_ends=[col for col in month_ends if col>=start_date and col<=end_date]
# 
#         for i in range(dict_len):
#             key1=keys[i]
#             factor=eval('data.' + key1)
#             #factor = eval('self.' + key1)
#             factor1=factor.copy()
#             if 'firstinvesttype' in factor1.columns.values : 
#                 factor1.reset_index(inplace=True)
#                 factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
#                 factor1=tool3.cleaning(factor1)
#                 try:
#                     factor1=factor1.loc['货币市场型基金']
#                     if key1 in d_freq_factor_list:
#                         factor1=tool3.d_freq_to_m_freq(factor1)
#                         factor1.columns=pd.DatetimeIndex(factor1.columns)
#                         fund_manager_factor_dict[key1]=factor1
#                     else:
#                         factor1.columns=pd.DatetimeIndex(factor1.columns)
#                         fund_manager_factor_dict[key1]=factor1
#                 except Exception as e:
#                     fund_manager_factor_dict[key1]=np.nan
#             else:
#                 factor1=tool3.cleaning(factor1)
#                 if key1 in d_freq_factor_list:
#                     factor1=tool3.d_freq_to_m_freq(factor1)
#                     factor1.columns=pd.DatetimeIndex(factor1.columns)
#                     fund_manager_factor_dict[key1]=factor1
#                 else:
#                     factor1.columns=pd.DatetimeIndex(factor1.columns)
#                     fund_manager_factor_dict[key1]=factor1
#                     fund_manager_factor_dict[key1]=factor1
#         combinedfactor_money=((((((((((((fund_manager_factor_dict['fund_manager_1y_return']*(1/13)).sub(fund_manager_factor_dict['fund_manager_treynor']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_stddev']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_maxdrawdown_duration_1y']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_custodianfeeratio']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_fund_corp_teamstability']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_fund_liquidation']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_corp_fivestarfundsprop']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_fundcototalnetassets']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_holder_mngemp_holdingpct']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_risk_annutrackerror']*(1/13),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_stocktonav']*(1/13),fill_value=0)).add(fund_manager_factor_dict['fund_manager_employee_board']*(1/13),fill_value=0)
#         for month_end in month_ends:
#             IC_dataframe=pd.DataFrame()
#             n3m=pct_chg_of_fund_manager_index_n3m.loc['货币市场型基金',month_end]
#             n6m=pct_chg_of_fund_manager_index_n6m.loc['货币市场型基金',month_end]
#             n1y=pct_chg_of_fund_manager_index_n1y.loc['货币市场型基金',month_end]
#             n2y=pct_chg_of_fund_manager_index_n2y.loc['货币市场型基金',month_end]
#             try:
#                 t=combinedfactor_money[month_end]
#                 IC_3m=t.corr(n3m)
#                 IC_6m=t.corr(n6m)
#                 IC_1y=t.corr(n1y)
#                 IC_2y=t.corr(n2y)
#                 IC_dataframe.loc['combinedfactor_money','3m']=IC_3m
#                 IC_dataframe.loc['combinedfactor_money','6m']=IC_6m
#                 IC_dataframe.loc['combinedfactor_money','1y']=IC_1y
#                 IC_dataframe.loc['combinedfactor_money','2y']=IC_2y
#             except Exception as e:
#                 IC_dataframe.loc['combinedfactor_money','3m']=np.nan
#                 IC_dataframe.loc['combinedfactor_money','6m']=np.nan
#                 IC_dataframe.loc['combinedfactor_money','1y']=np.nan
#                 IC_dataframe.loc['combinedfactor_money','2y']=np.nan
#             IC[month_end]=IC_dataframe
#             
#         #tt=IC
#         IC_3m_dataframe=pd.DataFrame()
#         IC_6m_dataframe=pd.DataFrame()
#         IC_1y_dataframe=pd.DataFrame()
#         IC_2y_dataframe=pd.DataFrame()
#         for month_end in month_ends:
#             IC_3m_dataframe[month_end]=IC[month_end]['3m']
#             IC_6m_dataframe[month_end]=IC[month_end]['6m']
#             IC_1y_dataframe[month_end]=IC[month_end]['1y']
#             IC_2y_dataframe[month_end]=IC[month_end]['2y']
# 
# 
#         IC_mean=pd.DataFrame()
#         #IC_max=pd.DataFrame()
#         IC_std=pd.DataFrame()
#         IR=pd.DataFrame()
#         #IR_max=pd.DataFrame()
#         IC_win=pd.DataFrame()
#         
#         
#         
#         IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
#         IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
#         IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
#         IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
#         
#         
#         
#         
#         IC_std['3m']=IC_3m_dataframe.std(axis=1)
#         IC_std['6m']=IC_6m_dataframe.std(axis=1)
#         IC_std['1y']=IC_1y_dataframe.std(axis=1)
#         IC_std['2y']=IC_2y_dataframe.std(axis=1)
# 
# 
#         IR=IC_mean/IC_std
#         
#         
#         
#         IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
#         IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
#         IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
#         IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)
# 
#         
#         
#         combinedfactortest_money=pd.DataFrame()
#         combinedfactortest_money['IC均值']=IC_mean.T['combinedfactor_money']
#         combinedfactortest_money['IC标准差']=IC_std.T['combinedfactor_money']
#         combinedfactortest_money['IR值']=IR.T['combinedfactor_money']
#         combinedfactortest_money['IC>0']=IC_win.T['combinedfactor_money']
#         
#         
#         combinedfactor_IC_money=pd.DataFrame()
#         combinedfactor_IC_money['3m']=IC_3m_dataframe.T['combinedfactor_money']
#         combinedfactor_IC_money['6m']=IC_6m_dataframe.T['combinedfactor_money']
#         combinedfactor_IC_money['1y']=IC_1y_dataframe.T['combinedfactor_money']
#         combinedfactor_IC_money['2y']=IC_2y_dataframe.T['combinedfactor_money']
#         
#         
#         combinedfactor_money.to_excel(os.path.join(basic_path, 'combinedfactor_money.xlsx'))
#         combinedfactortest_money.to_excel(os.path.join(plot_path, 'combinedfactortest_money.xlsx'))
#         combinedfactor_IC_money.to_excel(os.path.join(plot_path, 'combinedfactor_IC_money.xlsx'))
#         
# =============================================================================






    @staticmethod
    def combinedfactor_test_typealternative(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['另类投资基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        combinedfactor_alternative=(((((((((((fund_manager_factor_dict['fund_manager_1y_return']*(1/12)).sub(fund_manager_factor_dict['fund_manager_3y_return']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_treynor']*(1/12),fill_value=0)).add(fund_manager_factor_dict['fund_manager_sortino']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_r2']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_maxdrawdown_1y']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_custodianfeeratio']*(1/12),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_liquidation']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_holder_pct']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_fundcototalnetassets']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_1y_periodreturnranking']*(1/12),fill_value=0)).add(fund_manager_factor_dict['fund_manager_corp_productnopermanager']*(1/12),fill_value=0)
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['另类投资基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['另类投资基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['另类投资基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['另类投资基金',month_end]
            try:
                t=combinedfactor_alternative[month_end]
                IC_3m=t.corr(n3m)
                IC_6m=t.corr(n6m)
                IC_1y=t.corr(n1y)
                IC_2y=t.corr(n2y)
                IC_dataframe.loc['combinedfactor_alternative','3m']=IC_3m
                IC_dataframe.loc['combinedfactor_alternative','6m']=IC_6m
                IC_dataframe.loc['combinedfactor_alternative','1y']=IC_1y
                IC_dataframe.loc['combinedfactor_alternative','2y']=IC_2y
            except Exception as e:
                IC_dataframe.loc['combinedfactor_alternative','3m']=np.nan
                IC_dataframe.loc['combinedfactor_alternative','6m']=np.nan
                IC_dataframe.loc['combinedfactor_alternative','1y']=np.nan
                IC_dataframe.loc['combinedfactor_alternative','2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        #IC_max=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        #IR_max=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        combinedfactortest_alternative=pd.DataFrame()
        combinedfactortest_alternative['IC均值']=IC_mean.T['combinedfactor_alternative']
        combinedfactortest_alternative['IC标准差']=IC_std.T['combinedfactor_alternative']
        combinedfactortest_alternative['IR值']=IR.T['combinedfactor_alternative']
        combinedfactortest_alternative['IC>0']=IC_win.T['combinedfactor_alternative']
        
        
        combinedfactor_IC_alternative=pd.DataFrame()
        combinedfactor_IC_alternative['3m']=IC_3m_dataframe.T['combinedfactor_alternative']
        combinedfactor_IC_alternative['6m']=IC_6m_dataframe.T['combinedfactor_alternative']
        combinedfactor_IC_alternative['1y']=IC_1y_dataframe.T['combinedfactor_alternative']
        combinedfactor_IC_alternative['2y']=IC_2y_dataframe.T['combinedfactor_alternative']
        
        
        combinedfactor_alternative.to_excel(os.path.join(basic_path, 'combinedfactor_alternative.xlsx'))
        combinedfactortest_alternative.to_excel(os.path.join(plot_path, 'combinedfactortest_alternative.xlsx'))
        combinedfactor_IC_alternative.to_excel(os.path.join(plot_path, 'combinedfactor_IC_alternative.xlsx'))
        












    @staticmethod
    def combinedfactor_test_typealternative(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['另类投资基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        combinedfactor_alternative=(((((((((((fund_manager_factor_dict['fund_manager_1y_return']*(1/12)).sub(fund_manager_factor_dict['fund_manager_3y_return']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_treynor']*(1/12),fill_value=0)).add(fund_manager_factor_dict['fund_manager_sortino']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_r2']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_maxdrawdown_1y']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_custodianfeeratio']*(1/12),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_liquidation']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_holder_pct']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_fundcototalnetassets']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_1y_periodreturnranking']*(1/12),fill_value=0)).add(fund_manager_factor_dict['fund_manager_corp_productnopermanager']*(1/12),fill_value=0)
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['另类投资基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['另类投资基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['另类投资基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['另类投资基金',month_end]
            try:
                t=combinedfactor_alternative[month_end]
                IC_3m=t.corr(n3m)
                IC_6m=t.corr(n6m)
                IC_1y=t.corr(n1y)
                IC_2y=t.corr(n2y)
                IC_dataframe.loc['combinedfactor_alternative','3m']=IC_3m
                IC_dataframe.loc['combinedfactor_alternative','6m']=IC_6m
                IC_dataframe.loc['combinedfactor_alternative','1y']=IC_1y
                IC_dataframe.loc['combinedfactor_alternative','2y']=IC_2y
            except Exception as e:
                IC_dataframe.loc['combinedfactor_alternative','3m']=np.nan
                IC_dataframe.loc['combinedfactor_alternative','6m']=np.nan
                IC_dataframe.loc['combinedfactor_alternative','1y']=np.nan
                IC_dataframe.loc['combinedfactor_alternative','2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        #IC_max=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        #IR_max=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        combinedfactortest_alternative=pd.DataFrame()
        combinedfactortest_alternative['IC均值']=IC_mean.T['combinedfactor_alternative']
        combinedfactortest_alternative['IC标准差']=IC_std.T['combinedfactor_alternative']
        combinedfactortest_alternative['IR值']=IR.T['combinedfactor_alternative']
        combinedfactortest_alternative['IC>0']=IC_win.T['combinedfactor_alternative']
        
        
        combinedfactor_IC_alternative=pd.DataFrame()
        combinedfactor_IC_alternative['3m']=IC_3m_dataframe.T['combinedfactor_alternative']
        combinedfactor_IC_alternative['6m']=IC_6m_dataframe.T['combinedfactor_alternative']
        combinedfactor_IC_alternative['1y']=IC_1y_dataframe.T['combinedfactor_alternative']
        combinedfactor_IC_alternative['2y']=IC_2y_dataframe.T['combinedfactor_alternative']
        
        
        combinedfactor_alternative.to_excel(os.path.join(basic_path, 'combinedfactor_alternative.xlsx'))
        combinedfactortest_alternative.to_excel(os.path.join(plot_path, 'combinedfactortest_alternative.xlsx'))
        combinedfactor_IC_alternative.to_excel(os.path.join(plot_path, 'combinedfactor_IC_alternative.xlsx'))















    @staticmethod
    def backwardtest_stock(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
        data=Data()
        start_date=datetime(year=2009, month=1, day=1)
        end_date=end_date=datetime(year=2020, month=9, day=30)
        IC=dict()
        dict_len=len(fund_manager_factor_dict)
        keys=list(fund_manager_factor_dict.keys())
        pct_chg_of_fund_manager_index_n3m=data.pct_chg_of_fund_manager_index_n3m
        pct_chg_of_fund_manager_index_n3m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n3m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=data.pct_chg_of_fund_manager_index_n6m
        pct_chg_of_fund_manager_index_n6m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n6m.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=data.pct_chg_of_fund_manager_index_n1y
        pct_chg_of_fund_manager_index_n1y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n1y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=data.pct_chg_of_fund_manager_index_n2y
        pct_chg_of_fund_manager_index_n2y.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_n2y.set_index(['firstinvesttype','manager_ID'],inplace=True)
        
        pct_chg_of_fund_manager_index_n3m=tool3.cleaning(pct_chg_of_fund_manager_index_n3m)
        pct_chg_of_fund_manager_index_n6m=tool3.cleaning(pct_chg_of_fund_manager_index_n6m)
        pct_chg_of_fund_manager_index_n1y=tool3.cleaning(pct_chg_of_fund_manager_index_n1y)
        pct_chg_of_fund_manager_index_n2y=tool3.cleaning(pct_chg_of_fund_manager_index_n2y)
        
        
        
        month_ends=generate_months_ends()
        month_ends=[col for col in month_ends if col>=start_date and col<=end_date]

        for i in range(dict_len):
            key1=keys[i]
            factor=eval('data.' + key1)
            #factor = eval('self.' + key1)
            factor1=factor.copy()
            if 'firstinvesttype' in factor1.columns.values : 
                factor1.reset_index(inplace=True)
                factor1.set_index(['firstinvesttype','manager_ID'],inplace=True)
                factor1=tool3.cleaning(factor1)
                try:
                    factor1=factor1.loc['另类投资基金']
                    if key1 in d_freq_factor_list:
                        factor1=tool3.d_freq_to_m_freq(factor1)
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                    else:
                        factor1.columns=pd.DatetimeIndex(factor1.columns)
                        fund_manager_factor_dict[key1]=factor1
                except Exception as e:
                    fund_manager_factor_dict[key1]=np.nan
            else:
                factor1=tool3.cleaning(factor1)
                if key1 in d_freq_factor_list:
                    factor1=tool3.d_freq_to_m_freq(factor1)
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict[key1]=factor1
                    fund_manager_factor_dict[key1]=factor1
        combinedfactor_alternative=(((((((((((fund_manager_factor_dict['fund_manager_1y_return']*(1/12)).sub(fund_manager_factor_dict['fund_manager_3y_return']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_treynor']*(1/12),fill_value=0)).add(fund_manager_factor_dict['fund_manager_sortino']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_r2']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_maxdrawdown_1y']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_custodianfeeratio']*(1/12),fill_value=0)).add(fund_manager_factor_dict['fund_manager_fund_liquidation']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_holder_pct']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_prt_fundcototalnetassets']*(1/12),fill_value=0)).sub(fund_manager_factor_dict['fund_manager_1y_periodreturnranking']*(1/12),fill_value=0)).add(fund_manager_factor_dict['fund_manager_corp_productnopermanager']*(1/12),fill_value=0)
        for month_end in month_ends:
            IC_dataframe=pd.DataFrame()
            n3m=pct_chg_of_fund_manager_index_n3m.loc['另类投资基金',month_end]
            n6m=pct_chg_of_fund_manager_index_n6m.loc['另类投资基金',month_end]
            n1y=pct_chg_of_fund_manager_index_n1y.loc['另类投资基金',month_end]
            n2y=pct_chg_of_fund_manager_index_n2y.loc['另类投资基金',month_end]
            try:
                t=combinedfactor_alternative[month_end]
                IC_3m=t.corr(n3m)
                IC_6m=t.corr(n6m)
                IC_1y=t.corr(n1y)
                IC_2y=t.corr(n2y)
                IC_dataframe.loc['combinedfactor_alternative','3m']=IC_3m
                IC_dataframe.loc['combinedfactor_alternative','6m']=IC_6m
                IC_dataframe.loc['combinedfactor_alternative','1y']=IC_1y
                IC_dataframe.loc['combinedfactor_alternative','2y']=IC_2y
            except Exception as e:
                IC_dataframe.loc['combinedfactor_alternative','3m']=np.nan
                IC_dataframe.loc['combinedfactor_alternative','6m']=np.nan
                IC_dataframe.loc['combinedfactor_alternative','1y']=np.nan
                IC_dataframe.loc['combinedfactor_alternative','2y']=np.nan
            IC[month_end]=IC_dataframe
            
        #tt=IC
        IC_3m_dataframe=pd.DataFrame()
        IC_6m_dataframe=pd.DataFrame()
        IC_1y_dataframe=pd.DataFrame()
        IC_2y_dataframe=pd.DataFrame()
        for month_end in month_ends:
            IC_3m_dataframe[month_end]=IC[month_end]['3m']
            IC_6m_dataframe[month_end]=IC[month_end]['6m']
            IC_1y_dataframe[month_end]=IC[month_end]['1y']
            IC_2y_dataframe[month_end]=IC[month_end]['2y']


        IC_mean=pd.DataFrame()
        #IC_max=pd.DataFrame()
        IC_std=pd.DataFrame()
        IR=pd.DataFrame()
        #IR_max=pd.DataFrame()
        IC_win=pd.DataFrame()
        
        
        
        IC_mean['3m']=IC_3m_dataframe.mean(axis=1)
        IC_mean['6m']=IC_6m_dataframe.mean(axis=1)
        IC_mean['1y']=IC_1y_dataframe.mean(axis=1)
        IC_mean['2y']=IC_2y_dataframe.mean(axis=1)
        
        
        
        
        IC_std['3m']=IC_3m_dataframe.std(axis=1)
        IC_std['6m']=IC_6m_dataframe.std(axis=1)
        IC_std['1y']=IC_1y_dataframe.std(axis=1)
        IC_std['2y']=IC_2y_dataframe.std(axis=1)


        IR=IC_mean/IC_std
        
        
        
        IC_win['3m']=IC_3m_dataframe[IC_3m_dataframe>0].count(axis=1)/IC_3m_dataframe.count(axis=1)
        IC_win['6m']=IC_6m_dataframe[IC_6m_dataframe>0].count(axis=1)/IC_6m_dataframe.count(axis=1)
        IC_win['1y']=IC_1y_dataframe[IC_1y_dataframe>0].count(axis=1)/IC_1y_dataframe.count(axis=1)
        IC_win['2y']=IC_2y_dataframe[IC_2y_dataframe>0].count(axis=1)/IC_2y_dataframe.count(axis=1)

        
        
        combinedfactortest_alternative=pd.DataFrame()
        combinedfactortest_alternative['IC均值']=IC_mean.T['combinedfactor_alternative']
        combinedfactortest_alternative['IC标准差']=IC_std.T['combinedfactor_alternative']
        combinedfactortest_alternative['IR值']=IR.T['combinedfactor_alternative']
        combinedfactortest_alternative['IC>0']=IC_win.T['combinedfactor_alternative']
        
        
        combinedfactor_IC_alternative=pd.DataFrame()
        combinedfactor_IC_alternative['3m']=IC_3m_dataframe.T['combinedfactor_alternative']
        combinedfactor_IC_alternative['6m']=IC_6m_dataframe.T['combinedfactor_alternative']
        combinedfactor_IC_alternative['1y']=IC_1y_dataframe.T['combinedfactor_alternative']
        combinedfactor_IC_alternative['2y']=IC_2y_dataframe.T['combinedfactor_alternative']
        
        
        combinedfactor_alternative.to_excel(os.path.join(basic_path, 'combinedfactor_alternative.xlsx'))
        combinedfactortest_alternative.to_excel(os.path.join(plot_path, 'combinedfactortest_alternative.xlsx'))
        combinedfactor_IC_alternative.to_excel(os.path.join(plot_path, 'combinedfactor_IC_alternative.xlsx'))









if __name__ == "__main__":
    #factor_manager_analysis=Factor_Manager_Analysis()
    #factor_manager_analysis.corr_matrix_typestock()
    #factor_manager_analysis.corr_matrix_typeQDII()
    #factor_manager_analysis.corr_matrix_typealternative()
    #factor_manager_analysis.corr_matrix_typebond()
    #factor_manager_analysis.corr_matrix_typehybrid()
    #factor_manager_analysis.corr_matrix_typemoney()
    #factor_manager_analysis.efficient_test_typeQDII()
    #factor_manager_analysis.efficient_test_typestock()
    #factor_manager_analysis.efficient_test_typealternative()
    #factor_manager_analysis.efficient_test_typebond()
    #factor_manager_analysis.efficient_test_typehybrid()
    #factor_manager_analysis.efficient_test_typemoney()

