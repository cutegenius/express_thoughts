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






class Factor_Compute(CALFUNC):

    def __init__(self, status):
        super().__init__()
        # status = 'update' 表示仅对已有的因子值进行更新， ='all' 表示全部重新计算
        self._mes = generate_months_ends()
        self._status = status

    def _get_update_month(self, fn):
        factor_m = eval('self.' + fn)
        # factor_m = self.RETURN_12M
        last_dt = factor_m.columns[-1]
        to_update_month_list = [i for i in self._mes if i > last_dt]
        if len(to_update_month_list) == 0:
            print('没有更新必要')
            return None
            # sys.exit()
        else:
            return to_update_month_list
# =============================================================================
#     #todo 目前这个因子的原始数据从wind中手动提取更新后期有必要时应该接口导入
#     @lazyproperty
#     def fund_manager_basic_info(self):
#         #w.wss("000001.OF", "fund_fundmanageroftradedate,fund_corp_fundmanagementcompany,name_official,fund_manager_startdate,fund_manager_onthepostdays,fund_manager_enddate,fund_manager_gender,fund_manager_education,fund_manager_resume,fund_manager_age,fund_manager_managerworkingyears,NAV_periodicannualizedreturn","tradeDate=20201017;order=1;topNum=1")
#         #w.start()
#         #w.wss("320014.OF",  "fund_fullname,fund_mgrcomp,fund_setupdate,fund_predfundmanager,fund_fundmanager", usedf=True)[1]
#         '''
#         调用前先在wind中按照基金经理任职信息模板存储数据，保存为csv格式替换原数据！！！
#         '''
#         fund_manager_initial_info=self.fund_manager_initial_info
#         fund_manager_initial_info = fund_manager_initial_info.dropna(axis=0, how='all')
#         fund_manager_initial_info=fund_manager_initial_info[fund_manager_initial_info['是否初始基金']=='是']
#         fund_manager_initial_info=fund_manager_initial_info.drop('基金经理(历任)', axis=1).join(fund_manager_initial_info["基金经理(历任)"].str.split('\n',expand=True).stack().reset_index(level=1, drop=True).rename("基金经理"))
#         lst=list(fund_manager_initial_info['基金经理'])
#         lst_replaced=[str(sec).replace('至今','-*') for sec in lst]
#         fund_manager_initial_info['基金经理']=lst_replaced
#         lst2=list(fund_manager_initial_info['基金经理'])
#         lst2_replaced=[re.split('[()-]',str(sec)) for sec in lst2]
#         t=pd.DataFrame(lst2_replaced)
#         t.index=fund_manager_initial_info.index
#         fund_manager_initial_info=fund_manager_initial_info.drop(['基金经理','是否初始基金'], axis=1)
#         fund_manager_initial_info['基金经理']=t[0]
#         fund_manager_initial_info['任职日期']=t[1]
#         fund_manager_initial_info['离职日期']=t[2]
#         lst3=list(fund_manager_initial_info['基金成立日'])
#         lst3_replaced=[]
#         for day in lst3:
#             if pd.isnull(day):
#                 lst3_replaced.append(np.nan)
#             else:
#                 lst3_replaced.append(day.strftime("%Y%m%d"))
#         fund_manager_initial_info['基金成立日']=lst3_replaced
#         lst4=[]
#         for x,y in zip(fund_manager_initial_info['基金成立日'],fund_manager_initial_info['任职日期']):
#             if x==y:
#                 if pd.isnull(x):
#                     lst4.append(np.nan)
#                 else:
#                     lst4.append((datetime.strptime(str(x),'%Y%m%d')+relativedelta(months=6)).strftime("%Y%m%d"))
#             else:
#                 if pd.isnull(x):
#                     lst4.append(np.nan)
#                 else:
#                     lst4.append((datetime.strptime(str(x),'%Y%m%d')+relativedelta(months=3)).strftime("%Y%m%d"))
#         fund_manager_initial_info['有效任职日期']=lst4
#         fund_manager_basic_info=fund_manager_initial_info.reindex(columns=['证券简称','基金管理人中文名称','基金经理','任职日期','有效任职日期','离职日期'])
#
#         return fund_manager_basic_info
# =============================================================================



    @lazyproperty
    #就是填补了空值的refactor_net_value
    def ADJ_REFACTOR_NET_VALUE(self):
        #data=Data()
        #refactor_net_value=data.refactor_net_value
        refactor_net_value=self.refactor_net_value
        # 用前一列的值填补空值
        adj_refactor_net_value=refactor_net_value.fillna(method='ffill', axis=1, limit=4) #加一个limit=4的条件，减小填充误差
        #adj_refactor_net_value.to_excel(os.path.join(basic_path, 'adj_refactor_net_value.xlsx'))
        return adj_refactor_net_value


    @lazyproperty
    def PCT_CHG_OF_REFACTOR_NET_VALUE(self):
        #data=Data()
        #refactor_net_value=data.refactor_net_value
        refactor_net_value=self.refactor_net_value
        pct_chg_of_refactor_net_value = refactor_net_value/refactor_net_value.shift(periods=1, axis=1) - 1
        return pct_chg_of_refactor_net_value


    @lazyproperty
    def PCT_CHG_OF_ADJ_REFACTOR_NET_VALUE(self):
        #data=Data()
        #adj_refactor_net_value=data.adj_refactor_net_value
        adj_refactor_net_value=self.adj_refactor_net_value
        pct_chg_of_adj_refactor_net_value = adj_refactor_net_value/adj_refactor_net_value.shift(periods=1, axis=1) - 1
        #pct_chg_of_adj_refactor_net_value.to_excel(os.path.join(basic_path, 'pct_chg_of_adj_refactor_net_value.xlsx'))
        return pct_chg_of_adj_refactor_net_value


    @lazyproperty
    def PCT_CHG_OF_INDEX_PRICE_DAILY(self):
        #data=Data()
        #index_price_daily=data.index_price_daily
        index_price_daily=self.index_price_daily
        pct_chg_of_index_price_daily = index_price_daily/index_price_daily.shift(periods=1, axis=1) - 1
        return pct_chg_of_index_price_daily


    @lazyproperty
    def FUND_MANAGER_INDEX(self):
        #data=Data()
        #pct_chg_of_index_price_daily=data.pct_chg_of_index_price_daily
        #pct_chg_of_adj_refactor_net_value=data.pct_chg_of_adj_refactor_net_value
        pct_chg_of_index_price_daily=self.pct_chg_of_index_price_daily
        pct_chg_of_adj_refactor_net_value=self.pct_chg_of_adj_refactor_net_value
        pct_chg_of_adj_refactor_net_value.reset_index(inplace=True)
        #date_lst=pct_chg_of_refactor_net_value.columns.to_list()
        #date_lst_str=[day.strftime("%Y-%m-%d") for day in date_lst]
        #pct_chg_of_refactor_net_value.columns=date_lst_str
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,pct_chg_of_adj_refactor_net_value,left_on='wind_code',right_on='index',how='inner')
        def cut_data(t):
            #t=df_temp.iloc[0]
            if (isinstance(t.start_date_vaild,pd._libs.tslibs.nattype.NaTType))and(isinstance(t.end_date,pd._libs.tslibs.nattype.NaTType)):
                for ele in t.index.values:
                    if isinstance(ele,pd._libs.tslibs.timestamps.Timestamp):
                        t[ele]=np.nan
                return t
            elif (isinstance(t.start_date_vaild,pd._libs.tslibs.timestamps.Timestamp)) and (isinstance(t.end_date,pd._libs.tslibs.nattype.NaTType)):
                if t.start_date_vaild<t.index.values[11]:
                    return t
                elif t.start_date_vaild>t.index.values[-1]:
                    for ele in t.index.values:
                        if isinstance(ele,pd._libs.tslibs.timestamps.Timestamp):
                           t[ele]=np.nan
                    return t
                elif (t.start_date_vaild<=t.index.values[-1]) and (t.start_date_vaild>=t.index.values[11]):
                    for ele in t.index.values:
                        if isinstance(ele,pd._libs.tslibs.timestamps.Timestamp):
                            if ele<t.start_date_vaild:
                                t[ele]=np.nan
                    return t
            elif (isinstance(t.start_date_vaild,pd._libs.tslibs.timestamps.Timestamp)) and (isinstance(t.end_date,pd._libs.tslibs.timestamps.Timestamp)):
                if (t.start_date_vaild<t.index.values[11]) and (t.end_date<t.index.values[11]):
                    for ele in t.index.values:
                        if isinstance(ele,pd._libs.tslibs.timestamps.Timestamp):
                           t[ele]=np.nan
                    return t
                elif (t.start_date_vaild<t.index.values[11]) and (t.end_date>=t.index.values[11]) and (t.end_date<t.index.values[-1]):
                    for ele in t.index.values:
                        if isinstance(ele,pd._libs.tslibs.timestamps.Timestamp):
                           if ele>t.end_date:
                               t[ele]=np.nan
                    return t
                elif (t.start_date_vaild<t.index.values[11]) and (t.end_date>=t.index.values[-1]):
                    return t
                elif (t.start_date_vaild>=t.index.values[11]) and (t.start_date_vaild<=t.index.values[-1]) and (t.end_date>=t.index.values[11]) and (t.end_date<=t.index.values[-1]):
                    for ele in t.index.values:
                        if isinstance(ele,pd._libs.tslibs.timestamps.Timestamp):
                           if ele<t.start_date_vaild or ele>t.end_date:
                               t[ele]=np.nan
                    return t
                elif (t.start_date_vaild>=t.index.values[11]) and (t.start_date_vaild<=t.index.values[-1]) and (t.end_date>=t.index.values[-1]):
                    for ele in t.index.values:
                        if isinstance(ele,pd._libs.tslibs.timestamps.Timestamp):
                           if ele<t.start_date_vaild:
                               t[ele]=np.nan
                    return t
                elif (t.start_date_vaild>=t.index.values[-1]) and (t.end_date>=t.index.values[-1]):
                    for ele in t.index.values:
                        if isinstance(ele,pd._libs.tslibs.timestamps.Timestamp):
                            t[ele]=np.nan
                    return t
        df_temp=df_temp.apply(cut_data,axis=1)
        #df_temp2.to_excel(os.path.join(basic_path, 'df_temp2.xlsx'))
        grouped_df = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        grouped_df.reset_index(inplace=True)
        #grouped_df.to_excel(os.path.join(basic_path, 'grouped_df.xlsx'))
        fund_manager_collection=pd.read_excel(os.path.join(basic_path, 'fund_manager_collection.xlsx'),index_col=0)
        fund_manager_dict=dict(zip(fund_manager_collection['manager_ID'],fund_manager_collection['fund_manager']))
        type_match_dict={
            '混合型基金':'混合基金',
            '股票型基金':'股票基金',
            '债券型基金':'债券基金',
            '货币市场型基金':'货币基金',
            '国际(QDII)基金':'QDII基金'
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
                        if isinstance(ele, pd._libs.tslibs.timestamps.Timestamp) and (ele >= first_num_date) and (ele <= last_num_date):
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
        grouped_df = grouped_df.apply(plug_in_index_data, axis=1)
        grouped_df.set_index(['manager_ID','firstinvesttype'],inplace=True)
        grouped_df_plusone=grouped_df+1
        grouped_df_cumprod=grouped_df_plusone.cumprod(axis=1)
        #grouped_df.to_excel(os.path.join(basic_path, 'pct_chg_of_fund_manager_index.xlsx'))
        #todo 按理说这个数应该是以1000开头，但是现在就是1000*当天涨跌幅开头，留待日后解决
        fund_manager_index=grouped_df_cumprod*1000
        fund_manager_index.reset_index(inplace=True)
        fund_manager_index.insert(1, 'manager', fund_manager_index['manager_ID'].map(fund_manager_dict))
        fund_manager_index.set_index(['manager_ID','manager','firstinvesttype'],inplace=True)
        #fund_manager_index.to_excel(os.path.join(basic_path, 'fund_manager_index.xlsx'))
        res_dict = {'pct_chg_of_fund_manager_index': grouped_df,
                    'fund_manager_index': fund_manager_index
                    }
        return res_dict


    @lazyproperty
    def FUND_MANAGER_SEX(self):
        #男为1，女为0
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_sex=fund_manager_collection['gender']
        gender_dict={'男':1,'女':0}
        fund_manager_sex=fund_manager_sex.map(gender_dict)
        fund_manager_sex=pd.DataFrame(np.array([fund_manager_sex]).repeat(len(tds), axis=0)).T
        fund_manager_sex.index.name='manager_ID'
        fund_manager_sex.columns=tds
        #fund_manager_sex.to_excel(os.path.join(basic_path, 'fund_manager_sex.xlsx'))
        return fund_manager_sex


    @lazyproperty
    def FUND_MANAGER_EDUCATION(self):
        #2 表示博士， 1 表示硕士， 0 表示本科及以下
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_education=fund_manager_collection['education']
        education_dict={'博士':2,'硕士':1,'本科':0}
        fund_manager_education=fund_manager_education.map(education_dict)
        fund_manager_education=pd.DataFrame(np.array([fund_manager_education]).repeat(len(tds), axis=0)).T
        fund_manager_education.index.name='manager_ID'
        fund_manager_education.columns=tds
        #fund_manager_education.to_excel(os.path.join(basic_path, 'fund_manager_education.xlsx'))
        return fund_manager_education


    @lazyproperty
    def FUND_MANAGER_SELLSIDE(self):
        #是否曾有大陆卖方任职经历，1表示是0表示否
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_resume=fund_manager_collection['resume']
        sellside=pd.read_excel(date_dair+"\\fund\\download_from_wind\\证券公司.xlsx",header=0,index_col=0,usecols=None,squeeze=False)
        sellside=sellside[:-3]
        sellside_lst=list(sellside['HKS_INFO_NAME'])+list(sellside['COMP_NAME'])
        #t=fund_manager_resume[14]
        def is_sellside(t):
            for code in sellside_lst:
                if t.find(code) !=-1:
                    t=1
                    break
                else:
                    continue
            if t!=1:
                t=0
            return t
        #is_sellside(t)
        #t2 = fund_manager_resume[0]
        #is_sellside(t2)
        fund_manager_sellside=fund_manager_resume.apply(is_sellside)
        fund_manager_sellside=pd.DataFrame(np.array([fund_manager_sellside]).repeat(len(tds), axis=0)).T
        fund_manager_sellside.index.name='manager_ID'
        fund_manager_sellside.columns=tds
        #fund_manager_sellside.to_excel(os.path.join(basic_path, 'fund_manager_sellside.xlsx'))
        return fund_manager_sellside


    @lazyproperty
    def FUND_MANAGER_BANK(self):
        #是否曾有大陆银行任职经历，1表示是0表示否
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_resume=fund_manager_collection['resume']
        bank=pd.read_excel(date_dair+"\\fund\\download_from_wind\\银行.xlsx",header=0,index_col=0,usecols=None,squeeze=False)
        bank=bank[:-3]
        bank_lst=list(bank['HKS_INFO_NAME'])+list(bank['COMP_NAME'])
        #bank_lst去除0
        bank_list=[]
        for i  in bank_lst:
            if i!=0:
                bank_list.append(i)
        def is_bank(t):
            for code in bank_list:
                if t.find(code) !=-1:
                    t=1
                    break
                else:
                    continue
            if t!=1:
                t=0
            return t
        fund_manager_bank=fund_manager_resume.apply(is_bank)
        fund_manager_bank=pd.DataFrame(np.array([fund_manager_bank]).repeat(len(tds), axis=0)).T
        fund_manager_bank.index.name='manager_ID'
        fund_manager_bank.columns=tds
        #fund_manager_bank.to_excel(os.path.join(basic_path, 'fund_manager_bank.xlsx'))
        return fund_manager_bank
    
    
    
    @lazyproperty
    def FUND_MANAGER_INSURANCE(self):
        #是否曾有大陆保险任职经历，1表示是0表示否
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_resume=fund_manager_collection['resume']
        insurance=pd.read_excel(date_dair+"\\fund\\download_from_wind\\保险公司.xlsx",header=0,index_col=0,usecols=None,squeeze=False)
        insurance=insurance[:-3]
        insurance_lst=list(insurance['HKS_INFO_NAME'])+list(insurance['COMP_NAME'])
# =============================================================================
#         #bank_lst去除0
#         bank_list=[]
#         for i  in bank_lst:
#             if i!=0:
#                 bank_list.append(i)
# =============================================================================
        def is_insurance(t):
            for code in insurance_lst:
                if t.find(code) !=-1:
                    t=1
                    break
                else:
                    continue
            if t!=1:
                t=0
            return t
        fund_manager_insurance=fund_manager_resume.apply(is_insurance)
        fund_manager_insurance=pd.DataFrame(np.array([fund_manager_insurance]).repeat(len(tds), axis=0)).T
        fund_manager_insurance.index.name='manager_ID'
        fund_manager_insurance.columns=tds
        #fund_manager_insurance.to_excel(os.path.join(basic_path, 'fund_manager_insurance.xlsx'))
        return fund_manager_insurance





    @lazyproperty
    def FUND_MANAGER_FUTURE(self):
        #是否曾有大陆期货任职经历，1表示是0表示否
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_resume=fund_manager_collection['resume']
        future=pd.read_excel(date_dair+"\\fund\\download_from_wind\\期货公司.xlsx",header=0,index_col=0,usecols=None,squeeze=False)
        future=future[:-3]
        future_lst=list(future['HKS_INFO_NAME'])+list(future['COMP_NAME'])
# =============================================================================
#         #bank_lst去除0
#         bank_list=[]
#         for i  in bank_lst:
#             if i!=0:
#                 bank_list.append(i)
# =============================================================================
        def is_future(t):
            for code in future_lst:
                if t.find(code) !=-1:
                    t=1
                    break
                else:
                    continue
            if t!=1:
                t=0
            return t
        fund_manager_future=fund_manager_resume.apply(is_future)
        fund_manager_future=pd.DataFrame(np.array([fund_manager_future]).repeat(len(tds), axis=0)).T
        fund_manager_future.index.name='manager_ID'
        fund_manager_future.columns=tds
        #fund_manager_future.to_excel(os.path.join(basic_path, 'fund_manager_future.xlsx'))
        return fund_manager_future




    @lazyproperty
    def FUND_MANAGER_OTHER(self):
        #是否曾有大陆其他金融机构任职经历，1表示是0表示否
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_resume=fund_manager_collection['resume']
        other=pd.read_excel(date_dair+"\\fund\\download_from_wind\\其他金融机构.xlsx",header=0,index_col=0,usecols=None,squeeze=False)
        other=other[:-3]
        other_lst=list(other['HKS_INFO_NAME'])+list(other['COMP_NAME'])
# =============================================================================
#         #bank_lst去除0
#         bank_list=[]
#         for i  in bank_lst:
#             if i!=0:
#                 bank_list.append(i)
# =============================================================================
        def is_other(t):
            for code in other_lst:
                if t.find(code) !=-1:
                    t=1
                    break
                else:
                    continue
            if t!=1:
                t=0
            return t
        fund_manager_other=fund_manager_resume.apply(is_other)
        fund_manager_other=pd.DataFrame(np.array([fund_manager_other]).repeat(len(tds), axis=0)).T
        fund_manager_other.index.name='manager_ID'
        fund_manager_other.columns=tds
        #fund_manager_other.to_excel(os.path.join(basic_path, 'fund_manager_other.xlsx'))
        return fund_manager_other



    @lazyproperty
    def FUND_MANAGER_CPA(self):
        #是否为CPA，1表示是0表示否
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_resume=fund_manager_collection['resume']
        def is_cpa(t):
            if t.find('CPA') !=-1:
                t=1
            else:
                t=0
            return t
        fund_manager_cpa=fund_manager_resume.apply(is_cpa)
        fund_manager_cpa=pd.DataFrame(np.array([fund_manager_cpa]).repeat(len(tds), axis=0)).T
        fund_manager_cpa.index.name='manager_ID'
        fund_manager_cpa.columns=tds
        #fund_manager_cpa.to_excel(os.path.join(basic_path, 'fund_manager_cpa.xlsx'))
        return fund_manager_cpa



    @lazyproperty
    def FUND_MANAGER_CFA(self):
        #是否为CFA，1表示是0表示否
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_resume=fund_manager_collection['resume']
        def is_cfa(t):
            if t.find('CFA') !=-1:
                t=1
            else:
                t=0
            return t
        fund_manager_cfa=fund_manager_resume.apply(is_cfa)
        fund_manager_cfa=pd.DataFrame(np.array([fund_manager_cfa]).repeat(len(tds), axis=0)).T
        fund_manager_cfa.index.name='manager_ID'
        fund_manager_cfa.columns=tds
        #fund_manager_cfa.to_excel(os.path.join(basic_path, 'fund_manager_cfa.xlsx'))
        return fund_manager_cfa


    @lazyproperty
    def FUND_MANAGER_MBA(self):
        #是否为MBA，1表示是0表示否
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_manager_collection.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_resume=fund_manager_collection['resume']
        def is_mba(t):
            if t.find('MBA') !=-1:
                t=1
            else:
                t=0
            return t
        fund_manager_mba=fund_manager_resume.apply(is_mba)
        fund_manager_mba=pd.DataFrame(np.array([fund_manager_mba]).repeat(len(tds), axis=0)).T
        fund_manager_mba.index.name='manager_ID'
        fund_manager_mba.columns=tds
        #fund_manager_mba.to_excel(os.path.join(basic_path, 'fund_manager_mba.xlsx'))
        return fund_manager_mba


# =============================================================================
#     #todo 空值巨多认为误差过大，忽略此因子
#     @lazyproperty
#     def FUND_MANAGER_AGE(self):
#         #基金经理年龄
#         #data=Data()
#         #fund_manager_collection=data.fund_manager_collection
#         fund_manager_collection=self.fund_manager_collection
#         fund_manager_collection.reset_index(inplace=True)
#         fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
#         fund_manager_collection.set_index('manager_ID',inplace=True)
#         tds=tool3.get_trade_days()
#         fund_manager_education=fund_manager_collection['education']
#         fund_manager_education=fund_manager_education.map(education_dict)
#         fund_manager_education=pd.DataFrame(np.array([fund_manager_education]).repeat(len(tds), axis=0)).T
#         fund_manager_education.index.name='manager_ID'
#         fund_manager_education.columns=tds
#         #fund_manager_education.to_excel(os.path.join(basic_path, 'fund_manager_education.xlsx'))
#         return fund_manager_education
# =============================================================================


# =============================================================================
#     #todo 空值巨多认为误差过大，忽略此因子
#     @lazyproperty
#     def FUND_MANAGER_AGE(self):
#         #基金经理年龄
#         #data=Data()
#         #fund_manager_collection=data.fund_manager_collection
#         fund_manager_collection=self.fund_manager_collection
#         fund_manager_collection.reset_index(inplace=True)
#         fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
#         fund_manager_collection.set_index('manager_ID',inplace=True)
#         tds=tool3.get_trade_days()
#         fund_manager_education=fund_manager_collection['education']
#         fund_manager_education=fund_manager_education.map(education_dict)
#         fund_manager_education=pd.DataFrame(np.array([fund_manager_education]).repeat(len(tds), axis=0)).T
#         fund_manager_education.index.name='manager_ID'
#         fund_manager_education.columns=tds
#         #fund_manager_education.to_excel(os.path.join(basic_path, 'fund_manager_education.xlsx'))
#         return fund_manager_education
# =============================================================================

    @lazyproperty
    def PCT_CHG_OF_INDEX_PRICE_DAILY(self):
        #data=Data()
        #index_price_daily=data.index_price_daily
        index_price_daily=self.index_price_daily
        pct_chg_of_index_price_daily = index_price_daily/index_price_daily.shift(periods=1, axis=1) - 1
        return pct_chg_of_index_price_daily
    
    
    @lazyproperty
    def FUND_MANAGER_CUSTODIANFEERATIO(self):
        #托管费率
        data=Data()
        fund_fee=data.fund_fee
        #fund_fee=self.fund_fee
        fund_custodianfeeratio= fund_fee['FUND_CUSTODIANFEERATIO']
        tds=tool3.get_trade_days()
        fund_custodianfeeratio=pd.DataFrame(np.array([fund_custodianfeeratio]).repeat(len(tds), axis=0)).T
        fund_custodianfeeratio.index=fund_fee.index
        fund_custodianfeeratio.columns=tds
        fund_custodianfeeratio.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,fund_custodianfeeratio,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_custodianfeeratio = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        #fund_manager_custodianfeeratio.to_excel(os.path.join(basic_path, 'fund_manager_custodianfeeratio.xlsx'))

        
        return fund_manager_custodianfeeratio



    @lazyproperty
    def FUND_MANAGER_MANAGEMENTFEERATIO(self):
        #管理费
        data=Data()
        fund_fee=data.fund_fee
        #fund_fee=self.fund_fee
        fund_managementfeeratio= fund_fee['FUND_MANAGEMENTFEERATIO']
        tds=tool3.get_trade_days()
        fund_managementfeeratio=pd.DataFrame(np.array([fund_managementfeeratio]).repeat(len(tds), axis=0)).T
        fund_managementfeeratio.index=fund_fee.index
        fund_managementfeeratio.columns=tds
        fund_managementfeeratio.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,fund_managementfeeratio,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_managementfeeratio = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()

        #fund_manager_managementfeeratio.to_excel(os.path.join(basic_path, 'fund_manager_managementfeeratio.xlsx'))

        return fund_manager_managementfeeratio



    @lazyproperty
    def FUND_MANAGER_PURCHASEFEERATIO(self):
        #最高申购费
        data=Data()
        fund_fee=data.fund_fee
        #fund_fee=self.fund_fee
        fund_purchasefeeratio= fund_fee['FUND_PURCHASEFEERATIO']
        tds=tool3.get_trade_days()
        fund_purchasefeeratio=pd.DataFrame(np.array([fund_purchasefeeratio]).repeat(len(tds), axis=0)).T
        fund_purchasefeeratio.index=fund_fee.index
        fund_purchasefeeratio.columns=tds
        fund_purchasefeeratio.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,fund_purchasefeeratio,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_purchasefeeratio = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()

        #fund_manager_purchasefeeratio.to_excel(os.path.join(basic_path, 'fund_manager_purchasefeeratio.xlsx'))
        return fund_manager_purchasefeeratio



# =============================================================================
#     @lazyproperty
#     def FUND_MANAGER_TNA(self):
#         #基金经理最新季度的管理基金资产净值总和
#         #data=Data()
#         #total_tna=data.total_tna
#         total_tna=self.total_tna
#         total_tna.reset_index(inplace=True)
#         fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
#         def as_datetime(x):
#             if pd.isnull(x):
#                 return x
#             else:
#                 x_date=datetime.strptime(str(x),'%Y-%m-%d')
#                 return x_date
#         fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
#         fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
#         fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
#         df_temp=pd.merge(fund_manager_detail,total_tna,left_on='wind_code',right_on='index',how='inner')
#         df_temp=df_temp.apply(tool3.cut_data,axis=1)
#         fund_manager_tna = df_temp.groupby('manager_ID').sum()
#         #fund_manager_tna.to_excel(os.path.join(basic_path, 'fund_manager_tna.xlsx'))
#         return fund_manager_tna
# 
# 
# 
#     @lazyproperty
#     def FUND_MANAGER_TNA_GROWTHRATE(self):
#         #基金经理最新季度的管理基金资产净值总和的年增长率
#         #data=Data()
#         #fund_manager_tna=data.fund_manager_tna
#         fund_manager_tna=self.fund_manager_tna
#         fund_manager_tna[fund_manager_tna==0]=np.nan
#         fund_manager_tna_growthrate = fund_manager_tna/fund_manager_tna.shift(periods=4, axis=1)
#         #fund_manager_tna_growthrate.to_excel(os.path.join(basic_path, 'fund_manager_tna_growthrate.xlsx'))
#         return fund_manager_tna_growthrate
# =============================================================================




    @lazyproperty
    def FUND_MANAGER_PRT_FUNDNETASSET_TOTAL(self):
        #基金经理最新季度的管理基金资产净值总和
        #data=Data()
        #prt_fundnetasset_total=data.prt_fundnetasset_total
        prt_fundnetasset_total=self.prt_fundnetasset_total
        prt_fundnetasset_total.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,prt_fundnetasset_total,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_prt_fundnetasset_total = df_temp.groupby('manager_ID').sum()
        fund_manager_prt_fundnetasset_total[fund_manager_prt_fundnetasset_total==0]=np.nan
        fund_manager_prt_fundnetasset_total=tool3.adjust_months(fund_manager_prt_fundnetasset_total)
        fund_manager_prt_fundnetasset_total = tool3.append_df(fund_manager_prt_fundnetasset_total)
        #fund_manager_prt_fundnetasset_total.to_excel(os.path.join(basic_path, 'fund_manager_prt_fundnetasset_total.xlsx'))
        return fund_manager_prt_fundnetasset_total



    @lazyproperty
    def FUND_MANAGER_PRT_FUNDNETASSET_TOTAL_GROETHRATE(self):
        #基金经理最新季度的管理基金资产净值总和的年增长率
        data=Data()
        fund_manager_prt_fundnetasset_total=data.fund_manager_prt_fundnetasset_total
        fund_manager_prt_fundnetasset_total=self.fund_manager_prt_fundnetasset_total
        fund_manager_prt_fundnetasset_total_growthrate= fund_manager_prt_fundnetasset_total/fund_manager_prt_fundnetasset_total.shift(periods=12, axis=1)-1
        fund_manager_prt_fundnetasset_total_growthrate.to_excel(os.path.join(basic_path, 'fund_manager_prt_fundnetasset_total_growthrate.xlsx'))
        return fund_manager_prt_fundnetasset_total_growthrate






    @lazyproperty
    def FUND_MANAGER_QANAL_TOTALINCOME(self):
        #基金经理最新季度的在管基金基金利润的简单平均
        #data=Data()
        #qanal_totalincome=data.qanal_totalincome
        qanal_totalincome=self.qanal_totalincome
        qanal_totalincome.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,qanal_totalincome,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_qanal_totalincome = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        #fund_manager_qanal_totalincome=data.fund_manager_qanal_totalincome
        #fund_manager_qanal_totalincome.reset_index(inplace=True)
        #fund_manager_qanal_totalincome.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_qanal_totalincome=tool3.adjust_months(fund_manager_qanal_totalincome)
        fund_manager_qanal_totalincome = tool3.append_df(fund_manager_qanal_totalincome)
        #fund_manager_qanal_totalincome.to_excel(os.path.join(basic_path, 'fund_manager_qanal_totalincome.xlsx'))
        return fund_manager_qanal_totalincome


    @lazyproperty
    def FUND_MANAGER_PRT_TOTALASSET(self):
        #基金经理最新季度的管理基金资产总值总和
        #data=Data()
        #prt_totalasset=data.prt_totalasset
        prt_totalasset=self.prt_totalasset
        prt_totalasset.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,prt_totalasset,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_prt_totalasset = df_temp.groupby(['manager_ID', 'firstinvesttype']).sum()
        
        #fund_manager_prt_totalasset=data.fund_manager_prt_totalasset
        #fund_manager_prt_totalasset.reset_index(inplace=True)
        #fund_manager_prt_totalasset.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_prt_totalasset=tool3.adjust_months(fund_manager_prt_totalasset)
        fund_manager_prt_totalasset = tool3.append_df(fund_manager_prt_totalasset)
        fund_manager_prt_totalasset[fund_manager_prt_totalasset==0]=np.nan
        
        #fund_manager_prt_totalasset.to_excel(os.path.join(basic_path, 'fund_manager_prt_totalasset.xlsx'))
        return fund_manager_prt_totalasset


    @lazyproperty
    def FUND_MANAGER_DIV_ACCUMULATEDPERUNIT(self):
        #基金经理最新季度在管基金的单位累计分红的简单平均
        #data=Data()
        #div_accumulatedperunit=data.div_accumulatedperunit
        div_accumulatedperunit=self.div_accumulatedperunit
        div_accumulatedperunit.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,div_accumulatedperunit,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_div_accumulatedperunit = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        #fund_manager_div_accumulatedperunit=data.fund_manager_div_accumulatedperunit
        #fund_manager_div_accumulatedperunit.reset_index(inplace=True)
        #fund_manager_div_accumulatedperunit.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_div_accumulatedperunit=tool3.adjust_months(fund_manager_div_accumulatedperunit)
        fund_manager_div_accumulatedperunit = tool3.append_df(fund_manager_div_accumulatedperunit)

        #fund_manager_div_accumulatedperunit.to_excel(os.path.join(basic_path, 'fund_manager_div_accumulatedperunit.xlsx'))
        return fund_manager_div_accumulatedperunit



    @lazyproperty
    def FUND_MANAGER_TOTAL_EXPENSE(self):
        #基金经理最新季度在管基金费用合计的简单平均
        #data=Data()
        #total_expense=data.total_expense
        total_expense=self.total_expense
        total_expense.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,total_expense,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_total_expense = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        
        #fund_manager_total_expense=data.fund_manager_total_expense
        #fund_manager_total_expense.reset_index(inplace=True)
        #fund_manager_total_expense.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_total_expense=tool3.adjust_months(fund_manager_total_expense,'H')
        fund_manager_total_expense = tool3.append_df(fund_manager_total_expense)

        
        
        #fund_manager_total_expense.to_excel(os.path.join(basic_path, 'fund_manager_total_expense.xlsx'))
        return fund_manager_total_expense




    @lazyproperty
    def FUND_MANAGER_1Y_DIV_PAYOUT(self):
        #基金经理在管基金近一年年度分红总额的简单平均
        data=Data()
        div_payout=data.div_payout
        #div_payout=self.div_payout
        div_payout.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,div_payout,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_1y_div_payout = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        #fund_manager_1y_div_payout=data.fund_manager_1y_div_payout
        #fund_manager_1y_div_payout.reset_index(inplace=True)
        #fund_manager_1y_div_payout.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_1y_div_payout=tool3.adjust_months(fund_manager_1y_div_payout,'Y')
        fund_manager_1y_div_payout = tool3.append_df(fund_manager_1y_div_payout)
        
        fund_manager_1y_div_payout = CALFUNC.del_dat_early_than(fund_manager_1y_div_payout, START_YEAR)
        #fund_manager_1y_div_payout.to_excel(os.path.join(basic_path, 'fund_manager_1y_div_payout.xlsx'))
        return fund_manager_1y_div_payout



    @lazyproperty
    def FUND_MANAGER_3Y_DIV_PAYOUT(self):
        #基金经理在管基金近三年年度分红总额的简单平均
        data=Data()
        div_payout=data.div_payout
        #div_payout=self.div_payout
        div_payout.fillna(value=0, inplace=True)
        div_payout_3y=div_payout+div_payout.shift(periods=1, axis=1)+div_payout.shift(periods=2, axis=1)
        div_payout_3y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,div_payout_3y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_3y_div_payout = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
    
        fund_manager_3y_div_payout=tool3.adjust_months(fund_manager_3y_div_payout,'Y')
        fund_manager_3y_div_payout = tool3.append_df(fund_manager_3y_div_payout)
        
        fund_manager_3y_div_payout = CALFUNC.del_dat_early_than(fund_manager_3y_div_payout, START_YEAR)
        #fund_manager_3y_div_payout.to_excel(os.path.join(basic_path, 'fund_manager_3y_div_payout.xlsx'))
        return fund_manager_3y_div_payout


    @lazyproperty
    def FUND_MANAGER_5Y_DIV_PAYOUT(self):
        #基金经理在管基金近五年年度分红总额的简单平均
        #data=Data()
        #div_payout=data.div_payout
        div_payout=self.div_payout
        div_payout.fillna(value=0, inplace=True)
        div_payout_5y=div_payout+div_payout.shift(periods=1, axis=1)+div_payout.shift(periods=2, axis=1)+div_payout.shift(periods=3, axis=1)+div_payout.shift(periods=4, axis=1)
        div_payout_5y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,div_payout_5y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_5y_div_payout = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_5y_div_payout=tool3.adjust_months(fund_manager_5y_div_payout,'Y')
        fund_manager_5y_div_payout = tool3.append_df(fund_manager_5y_div_payout)
        
        
        fund_manager_5y_div_payout = CALFUNC.del_dat_early_than(fund_manager_5y_div_payout, START_YEAR)
        #fund_manager_5y_div_payout.to_excel(os.path.join(basic_path, 'fund_manager_5y_div_payout.xlsx'))
        return fund_manager_5y_div_payout



    @lazyproperty
    def FUND_MANAGER_1Y_return(self):
        #基金经理在管基金近一年年度回报的简单平均
        data=Data()
        return_y=data.return_y
        #return_y=self.return_y
        return_y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,return_y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_1y_return = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_1y_return=tool3.adjust_months(fund_manager_1y_return,'Y')
        fund_manager_1y_return = tool3.append_df(fund_manager_1y_return)
        
        fund_manager_1y_return = CALFUNC.del_dat_early_than(fund_manager_1y_return, START_YEAR)
        #fund_manager_1y_return.to_excel(os.path.join(basic_path, 'fund_manager_1y_return.xlsx'))
        return fund_manager_1y_return



    @lazyproperty
    def FUND_MANAGER_3Y_return(self):
        #基金经理在管基金近三年年度回报的简单平均
        data=Data()
        return_y=data.return_y
        #return_y=self.return_y
        return_3y=return_y+return_y.shift(periods=1, axis=1)+return_y.shift(periods=2, axis=1)
        return_3y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,return_3y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_3y_return = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_3y_return=tool3.adjust_months(fund_manager_3y_return,'Y')
        fund_manager_3y_return = tool3.append_df(fund_manager_3y_return)
        
        fund_manager_3y_return = CALFUNC.del_dat_early_than(fund_manager_3y_return, START_YEAR)
        #fund_manager_3y_return.to_excel(os.path.join(basic_path, 'fund_manager_3y_return.xlsx'))
        return fund_manager_3y_return



    @lazyproperty
    def FUND_MANAGER_5Y_return(self):
        #基金经理在管基金近五年年度回报的简单平均
        data=Data()
        return_y=data.return_y
        #return_y=self.return_y
        return_5y=return_y+return_y.shift(periods=1, axis=1)+return_y.shift(periods=2, axis=1)+return_y.shift(periods=3, axis=1)+return_y.shift(periods=4, axis=1)
        return_5y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,return_5y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_5y_return = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_5y_return=tool3.adjust_months(fund_manager_5y_return,'Y')
        fund_manager_5y_return = tool3.append_df(fund_manager_5y_return)
        
        fund_manager_5y_return = CALFUNC.del_dat_early_than(fund_manager_5y_return, START_YEAR)
        #fund_manager_5y_return.to_excel(os.path.join(basic_path, 'fund_manager_5y_return.xlsx'))
        return fund_manager_5y_return



    @lazyproperty
    def FUND_MANAGER_1Y_PERIODRETURNRANKING(self):
        #基金经理单年度回报排名的简单平均
        data=Data()
        periodreturnranking_y=data.periodreturnranking_y
        #periodreturnranking_y=self.periodreturnranking_y
        def str2float(s):
            if pd.isnull(s):
                return s
            else:
                s_float=int(s.split('/')[0])/int(s.split('/')[1])
                return s_float
        periodreturnranking_y_float=periodreturnranking_y.applymap(str2float)
        periodreturnranking_y_float.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,periodreturnranking_y_float,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_1y_periodreturnranking = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_1y_periodreturnranking=tool3.adjust_months(fund_manager_1y_periodreturnranking,'Y')
        fund_manager_1y_periodreturnranking = tool3.append_df(fund_manager_1y_periodreturnranking)
        
        fund_manager_1y_periodreturnranking = CALFUNC.del_dat_early_than(fund_manager_1y_periodreturnranking, START_YEAR)
        #fund_manager_1y_periodreturnranking.to_excel(os.path.join(basic_path, 'fund_manager_1y_periodreturnranking.xlsx'))
        return fund_manager_1y_periodreturnranking
    
    
    @lazyproperty
    def FUND_MANAGER_1Y_PERIODRETURNRANKING_TYPE(self):
        #基金经理单年度回报排名的简单平均在各自类型中的排名
        data=Data()
        fund_manager_1y_periodreturnranking=data.fund_manager_1y_periodreturnranking
        #fund_manager_1y_periodreturnranking=self.fund_manager_1y_periodreturnranking
        fund_manager_1y_periodreturnranking_type=fund_manager_1y_periodreturnranking.groupby('manager_ID').rank(ascending=True)
        #fund_manager_1y_periodreturnranking_type.to_excel(os.path.join(basic_path, 'fund_manager_1y_periodreturnranking_type.xlsx'))
        return fund_manager_1y_periodreturnranking_type
    
    
    @lazyproperty
    def FUND_MANAGER_STYLE_STYLECOEFFICIENT(self):
        #基金经理在管基金风格系数的简单平均
        #data=Data()
        #style_stylecoefficient=data.style_stylecoefficient
        style_stylecoefficient=self.style_stylecoefficient
        style_stylecoefficient.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,style_stylecoefficient,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_style_stylecoefficient = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_style_stylecoefficient=tool3.adjust_months(fund_manager_style_stylecoefficient,'H')
        fund_manager_style_stylecoefficient = tool3.append_df(fund_manager_style_stylecoefficient)
        
        
        fund_manager_style_stylecoefficient = CALFUNC.del_dat_early_than(fund_manager_style_stylecoefficient, START_YEAR)
        #fund_manager_style_stylecoefficient.to_excel(os.path.join(basic_path, 'fund_manager_style_stylecoefficient.xlsx'))
        return fund_manager_style_stylecoefficient



    @lazyproperty
    def FUND_MANAGER_STYLE_AVERAGEPOSITIONTIME(self):
        #基金经理在管基金平均持仓时间的简单平均
        #data=Data()
        #style_averagepositiontime=data.style_averagepositiontime
        style_averagepositiontime=self.style_averagepositiontime
        style_averagepositiontime.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,style_averagepositiontime,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_style_averagepositiontime = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_style_averagepositiontime=tool3.adjust_months(fund_manager_style_averagepositiontime,'Y')
        fund_manager_style_averagepositiontime = tool3.append_df(fund_manager_style_averagepositiontime)
        
        
        fund_manager_style_averagepositiontime = CALFUNC.del_dat_early_than(fund_manager_style_averagepositiontime, START_YEAR)
        #fund_manager_style_averagepositiontime.to_excel(os.path.join(basic_path, 'fund_manager_style_averagepositiontime.xlsx'))
        return fund_manager_style_averagepositiontime
    
    
    
    
    @lazyproperty
    def FUND_MANAGER_PRT_HKSTOCKTONAV(self):
        #基金经理在管基金港股投资市值占基金资产净值比的简单平均
        data=Data()
        prt_hkstocktonav=data.prt_hkstocktonav
        #prt_hkstocktonav=self.prt_hkstocktonav
        prt_hkstocktonav.fillna(value=0, inplace=True)
        prt_hkstocktonav.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,prt_hkstocktonav,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_prt_hkstocktonav = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_prt_hkstocktonav=tool3.adjust_months(fund_manager_prt_hkstocktonav,'Q')
        fund_manager_prt_hkstocktonav = tool3.append_df(fund_manager_prt_hkstocktonav)
        
        
        fund_manager_prt_hkstocktonav = CALFUNC.del_dat_early_than(fund_manager_prt_hkstocktonav, START_YEAR)
        #fund_manager_prt_hkstocktonav.to_excel(os.path.join(basic_path, 'fund_manager_prt_hkstocktonav.xlsx'))
        return fund_manager_prt_hkstocktonav



    @lazyproperty
    def FUND_MANAGER_PRT_STOCKTONAV(self):
        #基金经理在管基金股票市值占基金资产净值比的简单平均
        #data=Data()
        #prt_stocktonav=data.prt_stocktonav
        prt_stocktonav=self.prt_stocktonav
        prt_stocktonav.fillna(value=0, inplace=True)
        prt_stocktonav.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,prt_stocktonav,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_prt_stocktonav = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_prt_stocktonav=tool3.adjust_months(fund_manager_prt_stocktonav,'Q')
        fund_manager_prt_stocktonav = tool3.append_df(fund_manager_prt_stocktonav)
        
        fund_manager_prt_stocktonav = CALFUNC.del_dat_early_than(fund_manager_prt_stocktonav, START_YEAR)
        #fund_manager_prt_stocktonav.to_excel(os.path.join(basic_path, 'fund_manager_prt_stocktonav.xlsx'))
        return fund_manager_prt_stocktonav




    @lazyproperty
    def FUND_MANAGER_PRT_FUNDNOOFSECURITIES(self):
        #基金经理在管基金重仓证券持有基金数的简单平均
        '''
        此指标的原数据仅有数据浏览器手动可以提取目前
        '''
        data=Data()
        prt_fundnoofsecurities=data.prt_fundnoofsecurities
        #prt_fundnoofsecurities=self.prt_fundnoofsecurities
        prt_fundnoofsecurities.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,prt_fundnoofsecurities,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_prt_fundnoofsecurities = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_prt_fundnoofsecurities=tool3.adjust_months(fund_manager_prt_fundnoofsecurities,'Q')
        fund_manager_prt_fundnoofsecurities = tool3.append_df(fund_manager_prt_fundnoofsecurities)
        
        fund_manager_prt_fundnoofsecurities = CALFUNC.del_dat_early_than(fund_manager_prt_fundnoofsecurities, START_YEAR)
        #fund_manager_prt_fundnoofsecurities.to_excel(os.path.join(basic_path, 'fund_manager_prt_fundnoofsecurities.xlsx'))
        return fund_manager_prt_fundnoofsecurities



    @lazyproperty
    def FUND_MANAGER_PRT_CORPORATEBONDTOBOND(self):
        #基金经理在管基金企业发行债券市值占债券投资市值比的简单平均
        data=Data()
        prt_corporatebondtobond=data.prt_corporatebondtobond
        #prt_corporatebondtobond=self.prt_corporatebondtobond
        prt_corporatebondtobond.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,prt_corporatebondtobond,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_prt_corporatebondtobond = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_prt_corporatebondtobond=tool3.adjust_months(fund_manager_prt_corporatebondtobond,'Q')
        fund_manager_prt_corporatebondtobond = tool3.append_df(fund_manager_prt_corporatebondtobond)
        
        
        fund_manager_prt_corporatebondtobond = CALFUNC.del_dat_early_than(fund_manager_prt_corporatebondtobond, START_YEAR)
        #fund_manager_prt_corporatebondtobond.to_excel(os.path.join(basic_path, 'fund_manager_prt_corporatebondtobond.xlsx'))
        return fund_manager_prt_corporatebondtobond



    @lazyproperty
    def FUND_MANAGER_RATING_SHANGHAIOVERALL3Y(self):
        #基金经理在管基金是否获得评级机构评级（是为1，否为0）结果的简单平均
        #data=Data()
        #rating_shanghaioverall3y=data.rating_shanghaioverall3y
        rating_shanghaioverall3y=self.rating_shanghaioverall3y
        rating_shanghaioverall3y[pd.notnull(rating_shanghaioverall3y)]=1
        rating_shanghaioverall3y[pd.isnull(rating_shanghaioverall3y)]=0
        rating_shanghaioverall3y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,rating_shanghaioverall3y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_rating_shanghaioverall3y = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        fund_manager_rating_shanghaioverall3y = CALFUNC.del_dat_early_than(fund_manager_rating_shanghaioverall3y, START_YEAR)
        #fund_manager_rating_shanghaioverall3y.to_excel(os.path.join(basic_path, 'fund_manager_rating_shanghaioverall3y.xlsx'))
        return fund_manager_rating_shanghaioverall3y


    @lazyproperty
    def FUND_MANAGER_PRT_TOPSECTOSEC(self):
        #基金经理在管基金前10名重仓证券市值合计占证券投资市值比的简单平均
        '''
        此指标的原数据仅有数据浏览器手动可以提取目前
        '''
        data=Data()
        prt_topsectosec=data.prt_topsectosec
        #prt_topsectosec=self.prt_topsectosec
        prt_topsectosec.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,prt_topsectosec,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_prt_topsectosec = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_prt_topsectosec=tool3.adjust_months(fund_manager_prt_topsectosec,'Q')
        fund_manager_prt_topsectosec = tool3.append_df(fund_manager_prt_topsectosec)
        
        
        fund_manager_prt_topsectosec = CALFUNC.del_dat_early_than(fund_manager_prt_topsectosec, START_YEAR)
        #fund_manager_prt_topsectosec.to_excel(os.path.join(basic_path, 'fund_manager_prt_topsectosec.xlsx'))
        return fund_manager_prt_topsectosec



    @lazyproperty
    def FUND_MANAGER_HOLDER_MNGEMP_HOLDINGPCT(self):
        #基金经理在管基金管理人员工持有比例的简单平均
        data=Data()
        holder_mngemp_holdingpct=data.holder_mngemp_holdingpct
        #holder_mngemp_holdingpct=self.holder_mngemp_holdingpct
        holder_mngemp_holdingpct.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,holder_mngemp_holdingpct,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_holder_mngemp_holdingpct = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_holder_mngemp_holdingpct=tool3.adjust_months(fund_manager_holder_mngemp_holdingpct,'H')
        fund_manager_holder_mngemp_holdingpct = tool3.append_df(fund_manager_holder_mngemp_holdingpct)
        
        
        fund_manager_holder_mngemp_holdingpct = CALFUNC.del_dat_early_than(fund_manager_holder_mngemp_holdingpct, START_YEAR)
        #fund_manager_holder_mngemp_holdingpct.to_excel(os.path.join(basic_path, 'fund_manager_holder_mngemp_holdingpct.xlsx'))
        return fund_manager_holder_mngemp_holdingpct



    @lazyproperty
    def FUND_MANAGER_FUND_CORP_TEAMSTABILITY(self):
        #基金公司团队稳定性
        data=Data()
        fund_corp_teamstability=data.fund_corp_teamstability
        #fund_corp_teamstability=self.fund_corp_teamstability
        fund_corp_teamstability.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,fund_corp_teamstability,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_fund_corp_teamstability = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_fund_corp_teamstability=tool3.adjust_months(fund_manager_fund_corp_teamstability,'Y')
        fund_manager_fund_corp_teamstability = tool3.append_df(fund_manager_fund_corp_teamstability)
        
        
        fund_manager_fund_corp_teamstability = CALFUNC.del_dat_early_than(fund_manager_fund_corp_teamstability, START_YEAR)
        #fund_manager_fund_corp_teamstability.to_excel(os.path.join(basic_path, 'fund_manager_fund_corp_teamstability.xlsx'))
        return fund_manager_fund_corp_teamstability



    @lazyproperty
    def FUND_MANAGER_FUND_CORP_FIVESTARFUNDSPROP(self):
        #基金经理所在基金公司五星基金占比
        #data=Data()
        #fund_corp_fivestarfundsprop=data.fund_corp_fivestarfundsprop
        fund_corp_fivestarfundsprop=self.fund_corp_fivestarfundsprop
        fund_corp_fivestarfundsprop.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,fund_corp_fivestarfundsprop,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_fund_corp_fivestarfundsprop = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        fund_manager_fund_corp_fivestarfundsprop = CALFUNC.del_dat_early_than(fund_manager_fund_corp_fivestarfundsprop, START_YEAR)
        #fund_manager_fund_corp_fivestarfundsprop.to_excel(os.path.join(basic_path, 'fund_manager_fund_corp_fivestarfundsprop.xlsx'))
        return fund_manager_fund_corp_fivestarfundsprop
    
    
    
    
    @lazyproperty
    def FUND_MANAGER_FUND_AVERAGEWORKINGYEARS(self):
        #基金经理在管基金基金经理平均年限的简单平均
        fund_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司基本资料.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
        fund_collection.rename(columns={'证券代码':'wind_code', '基金经理平均年限\r':'fund_averageworkingyears', '证券简称':'sec_name', '任职基金获奖记录\r':'fund_manager_awardrecord', '基金经理成熟度':'fund_corp_fundmanagermaturity'}, inplace = True)
        fund_collection=fund_collection[:-2]
        fund_collection.set_index('wind_code',inplace=True)
        tds=tool3.get_trade_days()
        fund_averageworkingyears=fund_collection['fund_averageworkingyears']
        fund_averageworkingyears=pd.DataFrame(np.array([fund_averageworkingyears]).repeat(len(tds), axis=0)).T
        fund_averageworkingyears.index.name='wind_code'
        fund_averageworkingyears.index=fund_collection.index
        fund_averageworkingyears.columns=tds
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,fund_averageworkingyears,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_fund_averageworkingyears = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        fund_manager_fund_averageworkingyears = CALFUNC.del_dat_early_than(fund_manager_fund_averageworkingyears, START_YEAR)
        #fund_manager_fund_averageworkingyears.to_excel(os.path.join(basic_path, 'fund_manager_fund_averageworkingyears.xlsx'))
        return fund_manager_fund_averageworkingyears





    @lazyproperty
    def FUND_MANAGER_FUND_MANAGER_AWARDRECORD(self):
        #基金经理在管基金获奖支数（同支基金多次获奖按一次计算）的总和（暂不支持历史）
        fund_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司基本资料.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
        fund_collection.rename(columns={'证券代码':'wind_code', '基金经理平均年限\r':'fund_averageworkingyears', '证券简称':'sec_name', '任职基金获奖记录\r':'fund_manager_awardrecord', '基金经理成熟度':'fund_corp_fundmanagermaturity'}, inplace = True)
        fund_collection=fund_collection[:-2]
        fund_collection.set_index('wind_code',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_awardrecord=fund_collection['fund_manager_awardrecord']
        fund_manager_awardrecord=pd.DataFrame(np.array([fund_manager_awardrecord]).repeat(len(tds), axis=0)).T
        fund_manager_awardrecord.index.name='wind_code'
        fund_manager_awardrecord.index=fund_collection.index
        fund_manager_awardrecord.columns=tds
        fund_manager_awardrecord[pd.notnull(fund_manager_awardrecord)]=1
        fund_manager_awardrecord[pd.isnull(fund_manager_awardrecord)]=0
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,fund_manager_awardrecord,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_fund_manager_awardrecord = df_temp.groupby(['manager_ID', 'firstinvesttype']).sum()
        fund_manager_fund_manager_awardrecord = CALFUNC.del_dat_early_than(fund_manager_fund_manager_awardrecord, START_YEAR)
        #fund_manager_fund_manager_awardrecord.to_excel(os.path.join(basic_path, 'fund_manager_fund_manager_awardrecord.xlsx'))
        return fund_manager_fund_manager_awardrecord



    @lazyproperty
    def FUND_MANAGER_RISK_ANNUTRACKERROR(self):
        #基金经理在管基金跟踪误差（年化）的简单平均
        data=Data()
        risk_annutrackerror=data.risk_annutrackerror
        #risk_annutrackerror=self.risk_annutrackerror
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,risk_annutrackerror,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_risk_annutrackerror = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_risk_annutrackerror=tool3.adjust_months(fund_manager_risk_annutrackerror,'Y')
        fund_manager_risk_annutrackerror = tool3.append_df(fund_manager_risk_annutrackerror)
        
        
        fund_manager_risk_annutrackerror = CALFUNC.del_dat_early_than(fund_manager_risk_annutrackerror, START_YEAR)
        #fund_manager_risk_annutrackerror.to_excel(os.path.join(basic_path, 'fund_manager_risk_annutrackerror.xlsx'))
        return fund_manager_risk_annutrackerror



    @lazyproperty
    def FUND_MANAGER_FUND_CORP_FUNDMANAGERMATURITY(self):
        #基金公司基金经理成熟度（青涩为0，成熟为1，稳重为2，老练为3）（暂不支持历史）
        fund_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司基本资料.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
        fund_collection.rename(columns={'证券代码':'wind_code', '基金经理平均年限\r':'fund_averageworkingyears', '证券简称':'sec_name', '任职基金获奖记录\r':'fund_manager_awardrecord', '基金经理成熟度':'fund_corp_fundmanagermaturity'}, inplace = True)
        fund_collection=fund_collection[:-2]
        fund_collection.set_index('wind_code',inplace=True)
        tds=tool3.get_trade_days()
        fund_corp_fundmanagermaturity=fund_collection['fund_corp_fundmanagermaturity']
        fundmanagermaturity_dict={'老练':3,'稳重':2,'成熟':1,'青涩':0}
        fund_corp_fundmanagermaturity=fund_corp_fundmanagermaturity.map(fundmanagermaturity_dict)
        fund_corp_fundmanagermaturity=pd.DataFrame(np.array([fund_corp_fundmanagermaturity]).repeat(len(tds), axis=0)).T
        fund_corp_fundmanagermaturity.index.name='wind_code'
        fund_corp_fundmanagermaturity.index=fund_collection.index
        fund_corp_fundmanagermaturity.columns=tds
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,fund_corp_fundmanagermaturity,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_fund_corp_fundmanagermaturity = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        fund_manager_fund_corp_fundmanagermaturity = CALFUNC.del_dat_early_than(fund_manager_fund_corp_fundmanagermaturity, START_YEAR)
        #fund_manager_fund_corp_fundmanagermaturity.to_excel(os.path.join(basic_path, 'fund_manager_fund_corp_fundmanagermaturity.xlsx'))
        return fund_manager_fund_corp_fundmanagermaturity




    @lazyproperty
    def FUND_MANAGER_PRT_FUNDCOTOTALNETASSETS(self):
        #基金经理在管基金基金管理人资产净值合计
        data=Data()
        prt_fundcototalnetassets=data.prt_fundcototalnetassets
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,prt_fundcototalnetassets,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_prt_fundcototalnetassets = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        fund_manager_prt_fundcototalnetassets=tool3.adjust_months(fund_manager_prt_fundcototalnetassets,'Q')
        fund_manager_prt_fundcototalnetassets = tool3.append_df(fund_manager_prt_fundcototalnetassets)
        
        fund_manager_prt_fundcototalnetassets = CALFUNC.del_dat_early_than(fund_manager_prt_fundcototalnetassets, START_YEAR)
        fund_manager_prt_fundcototalnetassets.to_excel(os.path.join(basic_path, 'fund_manager_prt_fundcototalnetassets.xlsx'))
        return fund_manager_prt_fundcototalnetassets




# =============================================================================
#     @lazyproperty
#     def FUND_MANAGER_PRT_FUNDCOTOTALNETASSETS(self):
#         #基金公司大股东持股比例（暂不支持历史）
#         #data=Data()
#         #fund_manager_collection=data.fund_manager_collection
#         fund_manager_collection=self.fund_manager_collection
#         fund_manager_collection.reset_index(inplace=True)
#         fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
#         fund_corp_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
#         fund_corp_collection.rename(columns={'证券代码':'fundco_ID', '证券简称':'fundco', '大股东持股比例\r':'holder_pct', '注册资本\r':'regcapital','董事会人数\r':'employee_board','基金经理人均管理产品数\r':'corp_productnopermanager'}, inplace = True)
#         fund_corp_collection=fund_corp_collection[:-2]
#         df_temp=pd.merge(fund_manager_collection,fund_corp_collection,left_on='fundco',right_on='fundco',how='left')
#         df_temp.set_index('manager_ID',inplace=True)
#         
#         
#         tds=tool3.get_trade_days()
#         fund_manager_prt_fundcototalnetassets=df_temp['holder_pct']
#         fund_manager_prt_fundcototalnetassets=pd.DataFrame(np.array([fund_manager_prt_fundcototalnetassets]).repeat(len(tds), axis=0)).T
#         fund_manager_prt_fundcototalnetassets.index.name='manager_ID'
#         fund_manager_prt_fundcototalnetassets.columns=tds
#         #fund_manager_prt_fundcototalnetassets.to_excel(os.path.join(basic_path, 'fund_manager_prt_fundcototalnetassets.xlsx'))
#         return fund_manager_prt_fundcototalnetassets
# =============================================================================




    @lazyproperty
    def FUND_MANAGER_HOLDER_PCT(self):
        #基金公司大股东持股比例（暂不支持历史）
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_corp_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
        fund_corp_collection.rename(columns={'证券代码':'fundco_ID', '证券简称':'fundco', '大股东持股比例\r':'holder_pct', '注册资本\r':'regcapital','董事会人数\r':'employee_board','基金经理人均管理产品数\r':'corp_productnopermanager'}, inplace = True)
        fund_corp_collection=fund_corp_collection[:-2]
        df_temp=pd.merge(fund_manager_collection,fund_corp_collection,left_on='fundco',right_on='fundco',how='left')
        df_temp.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_holder_pct=df_temp['holder_pct']
        fund_manager_holder_pct=pd.DataFrame(np.array([fund_manager_holder_pct]).repeat(len(tds), axis=0)).T
        fund_manager_holder_pct.index.name='manager_ID'
        fund_manager_holder_pct.columns=tds
        #fund_manager_holder_pct.to_excel(os.path.join(basic_path, 'fund_manager_holder_pct.xlsx'))
        return fund_manager_holder_pct




    @lazyproperty
    def FUND_MANAGER_REGCAPITAL(self):
        #基金公司注册资本（暂不支持历史）
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_corp_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
        fund_corp_collection.rename(columns={'证券代码':'fundco_ID', '证券简称':'fundco', '大股东持股比例\r':'holder_pct', '注册资本\r':'regcapital','董事会人数\r':'employee_board','基金经理人均管理产品数\r':'corp_productnopermanager'}, inplace = True)
        fund_corp_collection=fund_corp_collection[:-2]
        df_temp=pd.merge(fund_manager_collection,fund_corp_collection,left_on='fundco',right_on='fundco',how='left')
        df_temp.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_regcapital=df_temp['regcapital']
        fund_manager_regcapital=pd.DataFrame(np.array([fund_manager_regcapital]).repeat(len(tds), axis=0)).T
        fund_manager_regcapital.index.name='manager_ID'
        fund_manager_regcapital.columns=tds
        #fund_manager_regcapital.to_excel(os.path.join(basic_path, 'fund_manager_regcapital.xlsx'))
        return fund_manager_regcapital



# =============================================================================
#     @lazyproperty
#     def FUND_MANAGER_EMPLOYEE_BOARD(self):
#         #基金公司董事会人数（暂不支持历史）
#         #data=Data()
#         #fund_manager_collection=data.fund_manager_collection
#         fund_manager_collection=self.fund_manager_collection
#         fund_manager_collection.reset_index(inplace=True)
#         fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
#         fund_corp_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
#         fund_corp_collection.rename(columns={'证券代码':'fundco_ID', '证券简称':'fundco', '大股东持股比例\r':'holder_pct', '注册资本\r':'regcapital','董事会人数\r':'employee_board','基金经理人均管理产品数\r':'corp_productnopermanager'}, inplace = True)
#         fund_corp_collection=fund_corp_collection[:-2]
#         df_temp=pd.merge(fund_manager_collection,fund_corp_collection,left_on='fundco',right_on='fundco',how='left')
#         df_temp.set_index('manager_ID',inplace=True)
#         tds=tool3.get_trade_days()
#         fund_manager_employee_board=df_temp['employee_board']
#         fund_manager_employee_board=pd.DataFrame(np.array([fund_manager_employee_board]).repeat(len(tds), axis=0)).T
#         fund_manager_employee_board.index.name='manager_ID'
#         fund_manager_employee_board.columns=tds
#         #fund_manager_employee_board.to_excel(os.path.join(basic_path, 'fund_manager_employee_board.xlsx'))
#         return fund_manager_employee_board
# =============================================================================
    
    
    
    
    @lazyproperty
    def FUND_MANAGER_EMPLOYEE_BOARD(self):
        #基金公司董事会人数（暂不支持历史）
        #data=Data()
        #fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_corp_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
        fund_corp_collection.rename(columns={'证券代码':'fundco_ID', '证券简称':'fundco', '大股东持股比例\r':'holder_pct', '注册资本\r':'regcapital','董事会人数\r':'employee_board','基金经理人均管理产品数\r':'corp_productnopermanager'}, inplace = True)
        fund_corp_collection=fund_corp_collection[:-2]
        df_temp=pd.merge(fund_manager_collection,fund_corp_collection,left_on='fundco',right_on='fundco',how='left')
        df_temp.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_employee_board=df_temp['employee_board']
        fund_manager_employee_board=pd.DataFrame(np.array([fund_manager_employee_board]).repeat(len(tds), axis=0)).T
        fund_manager_employee_board.index.name='manager_ID'
        fund_manager_employee_board.columns=tds
        #fund_manager_employee_board.to_excel(os.path.join(basic_path, 'fund_manager_employee_board.xlsx'))
        return fund_manager_employee_board




    @lazyproperty
    def FUND_MANAGER_CORP_PRODUCTNOPERMANAGER(self):
        #基金公司人均管理产品数（暂不支持历史）
        data=Data()
        fund_manager_collection=data.fund_manager_collection
        fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_corp_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
        fund_corp_collection.rename(columns={'证券代码':'fundco_ID', '证券简称':'fundco', '大股东持股比例\r':'holder_pct', '注册资本\r':'regcapital','董事会人数\r':'employee_board','基金经理人均管理产品数\r':'corp_productnopermanager'}, inplace = True)
        fund_corp_collection=fund_corp_collection[:-2]
        df_temp=pd.merge(fund_manager_collection,fund_corp_collection,left_on='fundco',right_on='fundco',how='left')
        df_temp.set_index('manager_ID',inplace=True)
        tds=tool3.get_trade_days()
        fund_manager_corp_productnopermanager=df_temp['corp_productnopermanager']
        fund_manager_corp_productnopermanager=pd.DataFrame(np.array([fund_manager_corp_productnopermanager]).repeat(len(tds), axis=0)).T
        fund_manager_corp_productnopermanager.index.name='manager_ID'
        fund_manager_corp_productnopermanager.columns=tds
        #fund_manager_corp_productnopermanager.to_excel(os.path.join(basic_path, 'fund_manager_corp_productnopermanager.xlsx'))
        return fund_manager_corp_productnopermanager



    @lazyproperty
    def FUND_MANAGER_FUND_LIQUIDATION(self):
        #基金公司年度清算基金的总和
        '''
        清算基金数据目前只可以从风控板块手动提出
        '''
        fund_liquidation=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金清算(明细).xlsx",header=0,index_col=None,usecols=None,squeeze=False)
        fund_liquidation=fund_liquidation[:-2]
        fund_liquidation['年份']=fund_liquidation['基金到期日'].apply(lambda x:str(x)[0:4])
        fund_liquidation_table=pd.pivot_table(fund_liquidation,index='基金公司',columns='年份',values='代码',aggfunc='count')
        ed=datetime.today()
        tdc = [col for col in fund_liquidation_table.columns if int(col) <ed.year]
        tdc_list=list(range(int(tdc[0]),int(tdc[-1])+1))
        tdc_list_to_str=[str(col) for col in tdc_list]
        fund_liquidation_table=fund_liquidation_table.reindex(columns=tdc_list_to_str)
        tdc_yearend=[col+'-12-31' for col in tdc_list_to_str]
        fund_liquidation_table.columns=tdc_yearend
        fund_liquidation_table.reset_index(inplace=True)
        data=Data()
        fund_manager_collection=data.fund_manager_collection
        #fund_manager_collection=self.fund_manager_collection
        fund_manager_collection.reset_index(inplace=True)
        fund_manager_collection.drop_duplicates(subset='manager_ID',inplace=True)
        fund_corp_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金公司.xlsx",header=0,index_col=None,usecols=None,squeeze=False)
        fund_corp_collection.rename(columns={'证券代码':'fundco_ID', '证券简称':'fundco', '大股东持股比例\r':'holder_pct', '注册资本\r':'regcapital','董事会人数\r':'employee_board','基金经理人均管理产品数\r':'corp_productnopermanager'}, inplace = True)
        fund_corp_collection=fund_corp_collection[:-2]
        df_temp=pd.merge(fund_manager_collection,fund_liquidation_table,left_on='fundco',right_on='基金公司',how='left')
        df_temp.set_index('manager_ID',inplace=True)
        fund_manager_fund_liquidation=df_temp.reindex(columns=tdc_yearend)
        
        fund_manager_fund_liquidation=tool3.adjust_months(fund_manager_fund_liquidation,'Y')
        fund_manager_fund_liquidation = tool3.append_df(fund_manager_fund_liquidation)
        #fund_manager_fund_liquidation.to_excel(os.path.join(basic_path, 'fund_manager_fund_liquidation.xlsx'))
        return fund_manager_fund_liquidation



    @lazyproperty
    def FUND_MANAGER_EXCESSRETURN(self):
        #基金经理指数相对上证基金指数的超额收益率
        data=Data()
        pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
        pct_chg_of_index_price_daily=data.pct_chg_of_index_price_daily
        pct_chg_of_fund_manager_index.reset_index(inplace=True)
        pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        df_temp=pct_chg_of_fund_manager_index.T.sub(pct_chg_of_index_price_daily.loc['上证基金指数'], axis=0)
        fund_manager_excessreturn=df_temp.T
        #fund_manager_excessreturn.to_excel(os.path.join(basic_path, 'fund_manager_excessreturn.xlsx'))
        return fund_manager_excessreturn



    @lazyproperty
    def FUND_MANAGER_STDDEV(self):
        #基金经理指数过去一年的标准差
        data=Data()
        pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index.reset_index(inplace=True)
        pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        def stddev(df, window=10):
            """
            Wrapper function to estimate rolling standard deviation.
            :param df: a pandas DataFrame.
            :param window: the rolling window.
            :return: a pandas DataFrame with the time-series stddev over the past 'window' days.
            """
            return df.rolling(window,min_periods=1).std()
        fund_manager_stddev=stddev(pct_chg_of_fund_manager_index.T,252)
        fund_manager_stddev=fund_manager_stddev.T
        #fund_manager_stddev.to_excel(os.path.join(basic_path, 'fund_manager_stddev.xlsx'))
        return fund_manager_stddev





    @lazyproperty
    def FUND_MANAGER_SHARPE(self):
        #基金经理指数近一年的夏普比率
        #data=Data()
        #pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index=self.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index.reset_index(inplace=True)
        pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        #fund_manager_stddev=data.fund_manager_stddev
        fund_manager_stddev=self.fund_manager_stddev
        fund_manager_stddev.reset_index(inplace=True)
        fund_manager_stddev.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_sharpe=pct_chg_of_fund_manager_index/fund_manager_stddev
        #fund_manager_sharpe.to_excel(os.path.join(basic_path, 'fund_manager_sharpe.xlsx'))
        return fund_manager_sharpe




# =============================================================================
#     @lazyproperty
#     def FUND_MANAGER_BETA(self):
#         #基金经理所承担的市场风险
#         #data=Data()
#         #pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
#         pct_chg_of_fund_manager_index=self.pct_chg_of_fund_manager_index
#         pct_chg_of_fund_manager_index.reset_index(inplace=True)
#         pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
#         #pct_chg_of_index_price_daily=data.pct_chg_of_index_price_daily
#         pct_chg_of_index_price_daily=self.pct_chg_of_index_price_daily
#         df_temp=pct_chg_of_fund_manager_index.T.div(pct_chg_of_index_price_daily.loc['上证基金指数'], axis=0)
#         fund_manager_beta=df_temp.T
#         #fund_manager_beta.to_excel(os.path.join(basic_path, 'fund_manager_beta.xlsx'))
#         return fund_manager_beta
# =============================================================================




    @lazyproperty
    def FUND_MANAGER_CAPM(self):
        #alpha:基金经理根据系统风险调整的超额收益率
        #beta:基金经理相对于市场的系统风险
        #R squared:基金经理的确定系数
        data=Data()
        pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
        #pct_chg_of_fund_manager_index=self.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index.reset_index(inplace=True)
        pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        pct_chg_of_index_price_daily=data.pct_chg_of_index_price_daily
        x=pct_chg_of_index_price_daily.loc['上证基金指数']
        y=pct_chg_of_fund_manager_index.T
        filled_y = y.interpolate()#对y中的缺失值进行线性插值
        def regression(model_frame):
            try:
                reg = LinearRegression().fit(np.array(model_frame.iloc[:,0]).reshape(-1,1),np.array(model_frame.iloc[:,-1]).reshape(-1,1))  # 拟合调参
                r2=reg.score(np.array(model_frame.iloc[:,0]).reshape(-1,1), np.array(model_frame.iloc[:,-1]).reshape(-1,1))  # 拟合程度，1.0最高
                alpha=reg.intercept_[0]
                beta=reg.coef_[0][0]
                return r2,alpha,beta
            except Exception as e:
                r2=np.nan
                alpha=np.nan
                beta=np.nan
                return r2,alpha,beta
        r2=filled_y.copy()
        alpha=filled_y.copy()
        beta=filled_y.copy()
# =============================================================================
#         for col in filled_y.columns.values:
#             t=filled_y[col]
#             model_frame=pd.concat([x,t],axis=1)
#             model_frame.dropna(axis=0, how='any',inplace=True)
#             r2[col],alpha[col],beta[col]=regression(model_frame)
# =============================================================================
        for col in filled_y.columns.values:
            t=filled_y[col]
            model_frame=pd.DataFrame()
            model_frame=pd.concat([x,t],axis=1)
            model_frame.dropna(axis=0, how='any',inplace=True)
            try:
                results = rolling_ols(x=model_frame.iloc[:,0], y=model_frame.iloc[:,-1], window=min(len(model_frame.iloc[:,-1])-1,252))
                r2[col]=results.ss_reg.div(results.ss_tot)
                alpha[col]=results.alpha
                beta[col]=results.beta
            except Exception as e:
                r2[col]=np.nan
                alpha[col]=np.nan
                beta[col]=np.nan
        alpha=alpha.T
        beta=beta.T
        r2=r2.T


# =============================================================================
#         fund_manager_alpha=data.fund_manager_alpha
#         fund_manager_alpha.reset_index(inplace=True)
#         fund_manager_alpha.set_index(['manager_ID','firstinvesttype'],inplace=True)
#         fund_manager_beta=data.fund_manager_beta
#         fund_manager_beta.reset_index(inplace=True)
#         fund_manager_beta.set_index(['manager_ID','firstinvesttype'],inplace=True)
#         fund_manager_r2=data.fund_manager_r2
#         fund_manager_r2.reset_index(inplace=True)
#         fund_manager_r2.set_index(['manager_ID','firstinvesttype'],inplace=True)
# =============================================================================

        alpha = CALFUNC.del_dat_early_than(alpha, START_YEAR)
        beta = CALFUNC.del_dat_early_than(beta, START_YEAR)
        r2 = CALFUNC.del_dat_early_than(r2, START_YEAR)
        
        alpha=tool3.cleaning(alpha)
        beta=tool3.cleaning(beta)
        r2=tool3.cleaning(r2)
        
        
        #fund_manager_alpha.to_excel(os.path.join(basic_path, 'fund_manager_alpha.xlsx'))
        #fund_manager_beta.to_excel(os.path.join(basic_path, 'fund_manager_beta.xlsx'))
        #fund_manager_r2.to_excel(os.path.join(basic_path, 'fund_manager_r2.xlsx'))
        
        res_dict = {
            'fund_manager_alpha': alpha,
            'fund_manager_beta': beta,
            'fund_manager_r2': r2
            }
        
        return res_dict





    @lazyproperty
    def FUND_MANAGER_TREYNOR(self):
        #基金经理指数近一年的特雷诺比率
        data=Data()
        pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
        #pct_chg_of_fund_manager_index=self.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index.reset_index(inplace=True)
        pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_beta=data.fund_manager_beta
        #fund_manager_beta=self.fund_manager_beta
        fund_manager_beta.reset_index(inplace=True)
        fund_manager_beta.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_treynor=pct_chg_of_fund_manager_index.div(fund_manager_beta, axis=1)
        fund_manager_treynor.to_excel(os.path.join(basic_path, 'fund_manager_treynor.xlsx'))
        return fund_manager_treynor




    @lazyproperty
    def FUND_MANAGER_DOWN_STDDEV(self):
        #基金经理指数近一年的下行风险
        data=Data()
        pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index=self.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index.reset_index(inplace=True)
        pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        #df_temp=pct_chg_of_fund_manager_index.T
        def down_stddev(df, window=10):
            """
            Wrapper function to estimate rolling downside deviation.
            :param df: a pandas DataFrame.
            :param window: the rolling window.
            :return: a pandas DataFrame with the time-series stddev over the past 'window' days.
            """
            def calculate_down_stddev(t):
                t[t>=0] = 0
                t[t<0] = t**2
                return np.sqrt(t.mean())
            return df.rolling(window,min_periods=1).apply(calculate_down_stddev)
        fund_manager_down_stddev=down_stddev(pct_chg_of_fund_manager_index.T,252)
        fund_manager_down_stddev=fund_manager_down_stddev.T
        #fund_manager_down_stddev.to_excel(os.path.join(basic_path, 'fund_manager_down_stddev.xlsx'))
        return fund_manager_down_stddev




    @lazyproperty
    def FUND_MANAGER_SORTINO(self):
        #基金经理指数近一年的索提诺比率
        #data=Data()
        #pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index=self.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index.reset_index(inplace=True)
        pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        #fund_manager_down_stddev=data.fund_manager_down_stddev
        fund_manager_down_stddev=self.fund_manager_down_stddev
        fund_manager_down_stddev.reset_index(inplace=True)
        fund_manager_down_stddev.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_sortino=pct_chg_of_fund_manager_index.div(fund_manager_down_stddev,axis=1)
        #fund_manager_sortino.to_excel(os.path.join(basic_path, 'fund_manager_sortino.xlsx'))
        return fund_manager_sortino



    @lazyproperty
    def FUND_MANAGER_TE(self):
        #基金经理指数近一年的跟踪误差
        #data=Data()
        #fund_manager_excessreturn=data.fund_manager_excessreturn
        fund_manager_excessreturn=self.fund_manager_excessreturn
        fund_manager_excessreturn.reset_index(inplace=True)
        fund_manager_excessreturn.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_te=((fund_manager_excessreturn.T).rolling(window=252,min_periods=1).std()).T
        #fund_manager_te.to_excel(os.path.join(basic_path, 'fund_manager_te.xlsx'))
        return fund_manager_te




    @lazyproperty
    def FUND_MANAGER_IR(self):
        #基金经理指数近一年相对市场基金指数的信息比率
        data=Data()
        fund_manager_excessreturn=data.fund_manager_excessreturn
        fund_manager_excessreturn=self.fund_manager_excessreturn
        fund_manager_excessreturn.reset_index(inplace=True)
        fund_manager_excessreturn.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_te=data.fund_manager_te
        fund_manager_te=self.fund_manager_te
        fund_manager_te.reset_index(inplace=True)
        fund_manager_te.set_index(['manager_ID','firstinvesttype'],inplace=True)
        fund_manager_ir=fund_manager_excessreturn.div(fund_manager_te,axis=1)
        fund_manager_ir.to_excel(os.path.join(basic_path, 'fund_manager_ir.xlsx'))
        return fund_manager_ir






# =============================================================================
#     @lazyproperty
#     def FUND_MANAGER_MAXDRAWDOWN_3M(self):
#         #基金经理指数过去一个季度的最大回撤    day=63
#         data=Data()
#         fund_manager_index=data.fund_manager_index
#         #fund_manager_index=self.fund_manager_index
#         fund_manager_index.drop(["manager"],axis=1,inplace=True) 
#         fund_manager_index.reset_index(inplace=True)
#         fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
#         def maxdrawdown(data):
#             index_j = np.argmax(np.fmax.accumulate(data) - data)  # 结束位置
#             index_i = np.argmax(data[:index_j])  # 开始位置
#             d = data[index_j] - data[index_i]  # 最大回撤
#             return d/data[index_i]
#         df_temp=fund_manager_index.T
#         df_temp.dropna(axis=1, how='all',inplace=True)
#         fund_manager_maxdrawdown_3m=df_temp.copy()
#         for col in df_temp.columns.values:
#             t=df_temp[col]
#             t.dropna(inplace=True)
#             try:
#                 tt=t.rolling(window=min(63,len(t)-1)).apply(maxdrawdown)
#                 fund_manager_maxdrawdown_3m[col]=tt
#             except Exception as e:
#                 fund_manager_maxdrawdown_3m[col]=np.nan
#         fund_manager_maxdrawdown_3m=fund_manager_maxdrawdown_3m.T
#         #fund_manager_maxdrawdown_3m.to_excel(os.path.join(basic_path, 'fund_manager_maxdrawdown_3m.xlsx'))
#         return fund_manager_maxdrawdown_3m
# =============================================================================



    @lazyproperty
    def FUND_MANAGER_MAXDRAWDOWN_3M(self):
        #基金经理指数过去一个季度的最大回撤    day=63
        #data=Data()
        #fund_manager_index=data.fund_manager_index
        fund_manager_index=self.fund_manager_index
        fund_manager_index.drop(["manager"],axis=1,inplace=True) 
        fund_manager_index.reset_index(inplace=True)
        fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        def maxdrawdown(data):
            index_j = np.argmax(np.fmax.accumulate(data) - data)  # 结束位置
            index_i = np.argmax(data[:index_j])  # 开始位置
            d = data[index_j] - data[index_i]  # 最大回撤
            return d/data[index_i]
        df_temp=fund_manager_index.T
        df_temp.dropna(axis=1, how='all',inplace=True)
        fund_manager_maxdrawdown_3m=df_temp.copy()
        for col in df_temp.columns.values:
            t=df_temp[col]
            t.dropna(inplace=True)
            try:
                tt=t.rolling(window=min(63,len(t)-1)).apply(maxdrawdown)
                fund_manager_maxdrawdown_3m[col]=tt
            except Exception as e:
                fund_manager_maxdrawdown_3m[col]=np.nan
        fund_manager_maxdrawdown_3m=fund_manager_maxdrawdown_3m.T
        #fund_manager_maxdrawdown_3m.to_excel(os.path.join(basic_path, 'fund_manager_maxdrawdown_3m.xlsx'))
        return fund_manager_maxdrawdown_3m





    @lazyproperty
    def FUND_MANAGER_MAXDRAWDOWN_1Y(self):
        #基金经理指数过去一年的最大回撤    day=252
        data=Data()
        fund_manager_index=data.fund_manager_index
        #fund_manager_index=self.fund_manager_index
        fund_manager_index.drop(["manager"],axis=1,inplace=True) 
        fund_manager_index.reset_index(inplace=True)
        fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        def maxdrawdown(data):
            index_j = np.argmax(np.fmax.accumulate(data) - data)  # 结束位置
            index_i = np.argmax(data[:index_j])  # 开始位置
            d = data[index_j] - data[index_i]  # 最大回撤
            return d/data[index_i]
        df_temp=fund_manager_index.T
        df_temp.dropna(axis=1, how='all',inplace=True)
        fund_manager_maxdrawdown_1y=df_temp.copy()
        for col in df_temp.columns.values:
            t=df_temp[col]
            t.dropna(inplace=True)
            try:
                tt=t.rolling(window=min(252,len(t)-1)).apply(maxdrawdown)
                fund_manager_maxdrawdown_1y[col]=tt
            except Exception as e:
                fund_manager_maxdrawdown_1y[col]=np.nan
        fund_manager_maxdrawdown_1y=fund_manager_maxdrawdown_1y.T
        #fund_manager_maxdrawdown_1y.to_excel(os.path.join(basic_path, 'fund_manager_maxdrawdown_1y.xlsx'))
        return fund_manager_maxdrawdown_1y





    @lazyproperty
    def FUND_MANAGER_DRAWDOWN_1Y(self):
        #基金经理指数的回撤（相较于近一年高点）
        data=Data()
        fund_manager_index=data.fund_manager_index
        #fund_manager_index=self.fund_manager_index
        fund_manager_index.drop(["manager"],axis=1,inplace=True) 
        fund_manager_index.reset_index(inplace=True)
        fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        def drawdown(data):
            d = data[-1] - max(data)  # 相较于近一年高点回撤
            return d/max(data)
        df_temp=fund_manager_index.T
        df_temp.dropna(axis=1, how='all',inplace=True)
        fund_manager_drawdown_1y=df_temp.copy()
        for col in df_temp.columns.values:
            t=df_temp[col]
            t.dropna(inplace=True)
            try:
                tt=t.rolling(window=min(252,len(t)-1),min_periods=2).apply(drawdown)
                fund_manager_drawdown_1y[col]=tt
            except Exception as e:
                fund_manager_drawdown_1y[col]=np.nan
        fund_manager_drawdown_1y=fund_manager_drawdown_1y.T
        #fund_manager_drawdown_1y.to_excel(os.path.join(basic_path, 'fund_manager_drawdown_1y.xlsx'))
        return fund_manager_drawdown_1y





    @lazyproperty
    def FUND_MANAGER_MAXDRAWDOWN_DURATION_1Y(self):
        #基金经理指数过去一年的最大回撤久期(天)    day=252
        data=Data()
        fund_manager_index=data.fund_manager_index
        #fund_manager_index=self.fund_manager_index
        fund_manager_index.drop(["manager"],axis=1,inplace=True) 
        fund_manager_index.reset_index(inplace=True)
        fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        def maxdrawdown_duration(data):
            index_j =data.index.values[np.argmax(np.fmax.accumulate(data) - data)]   # 结束位置
            index_i =data.index.values[np.argmax(data[:index_j])]   # 开始位置
            try:
                index_d= [x for x in np.where(data>=data[index_i])[0] if x>index_j][0]
            except Exception as e:
                index_d=data.index.values[-1]
            return (index_d-index_i).astype('timedelta64[D]')/np.timedelta64(1, 'D')
        df_temp=fund_manager_index.T
        df_temp.dropna(axis=1, how='all',inplace=True)
        fund_manager_maxdrawdown_duration_1y=df_temp.copy()
        for col in df_temp.columns.values:
            t=df_temp[col]
            t.dropna(inplace=True)
            try:
                tt=t.rolling(window=min(252,len(t)-1)).apply(maxdrawdown_duration)
                fund_manager_maxdrawdown_duration_1y[col]=tt
            except Exception as e:
                fund_manager_maxdrawdown_duration_1y[col]=np.nan
        fund_manager_maxdrawdown_duration_1y=fund_manager_maxdrawdown_duration_1y.T
        #fund_manager_maxdrawdown_duration_1y.to_excel(os.path.join(basic_path, 'fund_manager_maxdrawdown_duration_1y.xlsx'))
        return fund_manager_maxdrawdown_duration_1y



    @lazyproperty
    def pct_chg_of_fund_manager_index_m(self):
        #data=Data()
        #pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index=self.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index.reset_index(inplace=True)
        pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        
        
        tds = trade_days()
        
        # 得到月末日期列表
        months_end = []
        for i in range(1, len(tds)):
            if tds[i].month != tds[i - 1].month:
                months_end.append(tds[i - 1])
            elif i == len(tds) - 1:
                months_end.append(tds[i])
        try:
            months_end = [me for me in months_end if me.year >= d_df.columns[0].year]
        except Exception as e:
            print('debug')
        
        # 赵到对应的月末日期列表，可能年月同日不同的情况
        new_col = []
        for col in pct_chg_of_fund_manager_index.columns:
            for me in months_end:
                if col.year == me.year and col.month == me.month:
                    new_col.append(me)
        new_col = sorted(list(set(new_col)))
        
        pct_chg_of_fund_manager_index.columns = pd.to_datetime(pct_chg_of_fund_manager_index.columns)
        pct_chg_of_fund_manager_index_m=pct_chg_of_fund_manager_index.groupby([pct_chg_of_fund_manager_index.columns.year,pct_chg_of_fund_manager_index.columns.month],axis=1).sum()
        pct_chg_of_fund_manager_index_m.columns=new_col
        pct_chg_of_fund_manager_index_m[pct_chg_of_fund_manager_index_m==0]=np.nan
        #pct_chg_of_fund_manager_index_m.to_excel(os.path.join(basic_path, 'pct_chg_of_fund_manager_index_m.xlsx'))
        return pct_chg_of_fund_manager_index_m



    @lazyproperty
    def pct_chg_of_fund_manager_index_m(self):
        #data=Data()
        #pct_chg_of_fund_manager_index=data.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index=self.pct_chg_of_fund_manager_index
        pct_chg_of_fund_manager_index.reset_index(inplace=True)
        pct_chg_of_fund_manager_index.set_index(['manager_ID','firstinvesttype'],inplace=True)
        
        
        tds = trade_days()
        
        # 得到月末日期列表
        months_end = []
        for i in range(1, len(tds)):
            if tds[i].month != tds[i - 1].month:
                months_end.append(tds[i - 1])
            elif i == len(tds) - 1:
                months_end.append(tds[i])
        try:
            months_end = [me for me in months_end if me.year >= d_df.columns[0].year]
        except Exception as e:
            print('debug')
        
        # 赵到对应的月末日期列表，可能年月同日不同的情况
        new_col = []
        for col in pct_chg_of_fund_manager_index.columns:
            for me in months_end:
                if col.year == me.year and col.month == me.month:
                    new_col.append(me)
        new_col = sorted(list(set(new_col)))
        
        pct_chg_of_fund_manager_index.columns = pd.to_datetime(pct_chg_of_fund_manager_index.columns)
        pct_chg_of_fund_manager_index_m=pct_chg_of_fund_manager_index.groupby([pct_chg_of_fund_manager_index.columns.year,pct_chg_of_fund_manager_index.columns.month],axis=1).sum()
        pct_chg_of_fund_manager_index_m.columns=new_col
        pct_chg_of_fund_manager_index_m[pct_chg_of_fund_manager_index_m==0]=np.nan
        #pct_chg_of_fund_manager_index_m.to_excel(os.path.join(basic_path, 'pct_chg_of_fund_manager_index_m.xlsx'))
        return pct_chg_of_fund_manager_index_m



    @lazyproperty
    def pct_chg_of_fund_manager_index_n3m(self):
        #data=Data()
        #pct_chg_of_fund_manager_index_m=data.pct_chg_of_fund_manager_index_m
        pct_chg_of_fund_manager_index_m=self.pct_chg_of_fund_manager_index_m
        pct_chg_of_fund_manager_index_m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_m.set_index(['manager_ID','firstinvesttype'],inplace=True)
        pct_chg_of_fund_manager_index_n3m=pct_chg_of_fund_manager_index_m.shift(periods=-1, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-2, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-3, axis=1)
        #pct_chg_of_fund_manager_index_n3m.to_excel(os.path.join(basic_path, 'pct_chg_of_fund_manager_index_n3m.xlsx'))
        return pct_chg_of_fund_manager_index_n3m





    @lazyproperty
    def pct_chg_of_fund_manager_index_n6m(self):
        #data=Data()
        #pct_chg_of_fund_manager_index_m=data.pct_chg_of_fund_manager_index_m
        pct_chg_of_fund_manager_index_m=self.pct_chg_of_fund_manager_index_m
        pct_chg_of_fund_manager_index_m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_m.set_index(['manager_ID','firstinvesttype'],inplace=True)
        pct_chg_of_fund_manager_index_n6m=pct_chg_of_fund_manager_index_m.shift(periods=-1, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-2, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-3, axis=1)\
            +pct_chg_of_fund_manager_index_m.shift(periods=-4, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-5, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-6, axis=1)
        #pct_chg_of_fund_manager_index_n6m.to_excel(os.path.join(basic_path, 'pct_chg_of_fund_manager_index_n6m.xlsx'))
        return pct_chg_of_fund_manager_index_n6m




    @lazyproperty
    def pct_chg_of_fund_manager_index_n1y(self):
        #data=Data()
        #pct_chg_of_fund_manager_index_m=data.pct_chg_of_fund_manager_index_m
        pct_chg_of_fund_manager_index_m=self.pct_chg_of_fund_manager_index_m
        pct_chg_of_fund_manager_index_m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_m.set_index(['manager_ID','firstinvesttype'],inplace=True)
        pct_chg_of_fund_manager_index_n1y=pct_chg_of_fund_manager_index_m.shift(periods=-1, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-2, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-3, axis=1)\
            +pct_chg_of_fund_manager_index_m.shift(periods=-4, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-5, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-6, axis=1)\
                +pct_chg_of_fund_manager_index_m.shift(periods=-7, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-8, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-9, axis=1)\
                    +pct_chg_of_fund_manager_index_m.shift(periods=-10, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-11, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-12, axis=1)
        #pct_chg_of_fund_manager_index_n1y.to_excel(os.path.join(basic_path, 'pct_chg_of_fund_manager_index_n1y.xlsx'))
        return pct_chg_of_fund_manager_index_n1y





    def pct_chg_of_fund_manager_index_n2y(self):
        #data=Data()
        #pct_chg_of_fund_manager_index_m=data.pct_chg_of_fund_manager_index_m
        pct_chg_of_fund_manager_index_m=self.pct_chg_of_fund_manager_index_m
        pct_chg_of_fund_manager_index_m.reset_index(inplace=True)
        pct_chg_of_fund_manager_index_m.set_index(['manager_ID','firstinvesttype'],inplace=True)
        pct_chg_of_fund_manager_index_n2y=pct_chg_of_fund_manager_index_m.shift(periods=-1, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-2, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-3, axis=1)\
            +pct_chg_of_fund_manager_index_m.shift(periods=-4, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-5, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-6, axis=1)\
                +pct_chg_of_fund_manager_index_m.shift(periods=-7, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-8, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-9, axis=1)\
                    +pct_chg_of_fund_manager_index_m.shift(periods=-10, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-11, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-12, axis=1)\
                        +pct_chg_of_fund_manager_index_m.shift(periods=-13, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-14, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-15, axis=1)\
                            +pct_chg_of_fund_manager_index_m.shift(periods=-16, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-17, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-18, axis=1)\
                                +pct_chg_of_fund_manager_index_m.shift(periods=-19, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-20, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-21, axis=1)\
                                    +pct_chg_of_fund_manager_index_m.shift(periods=-22, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-23, axis=1)+pct_chg_of_fund_manager_index_m.shift(periods=-24, axis=1)
        #pct_chg_of_fund_manager_index_n2y.to_excel(os.path.join(basic_path, 'pct_chg_of_fund_manager_index_n2y.xlsx'))
        return pct_chg_of_fund_manager_index_n2y






    @lazyproperty
    def FUND_MANAGER_1Y_excess_return(self):
        #基金经理在管基金近一年区间净值超越基准收益率的简单平均
        data=Data()
        NAV_over_bench_return_per_1y=data.NAV_over_bench_return_per_1y
        #return_y=self.return_y
        NAV_over_bench_return_per_1y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,NAV_over_bench_return_per_1y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_1y_excess_return = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        #fund_manager_1y_excess_return=tool3.adjust_months(fund_manager_1y_excess_return,'Y')
        fund_manager_1y_excess_return = tool3.append_df(fund_manager_1y_excess_return)
        
        fund_manager_1y_excess_return = CALFUNC.del_dat_early_than(fund_manager_1y_excess_return, START_YEAR)
        #fund_manager_1y_excess_return.to_excel(os.path.join(basic_path, 'fund_manager_1y_excess_return.xlsx'))
        return fund_manager_1y_excess_return






    @lazyproperty
    def FUND_MANAGER_1Y_excess_return(self):
        #基金经理在管基金近一年区间净值超越基准收益率的简单平均
        data=Data()
        NAV_over_bench_return_per_1y=data.NAV_over_bench_return_per_1y
        #return_y=self.return_y
        NAV_over_bench_return_per_1y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,NAV_over_bench_return_per_1y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_1y_excess_return = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        #fund_manager_1y_excess_return=tool3.adjust_months(fund_manager_1y_excess_return,'Y')
        fund_manager_1y_excess_return = tool3.append_df(fund_manager_1y_excess_return)
        
        fund_manager_1y_excess_return = CALFUNC.del_dat_early_than(fund_manager_1y_excess_return, START_YEAR)
        #fund_manager_1y_excess_return.to_excel(os.path.join(basic_path, 'fund_manager_1y_excess_return.xlsx'))
        return fund_manager_1y_excess_return







    @lazyproperty
    def FUND_MANAGER_3Y_excess_return(self):
        #基金经理在管基金近三年区间净值超越基准收益率的简单平均
        data=Data()
        NAV_over_bench_return_per_3y=data.NAV_over_bench_return_per_1y
        #NAV_over_bench_return_per_3y=self.NAV_over_bench_return_per_3y
        NAV_over_bench_return_per_3y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,NAV_over_bench_return_per_3y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_3y_excess_return = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        #fund_manager_1y_excess_return=tool3.adjust_months(fund_manager_1y_excess_return,'Y')
        fund_manager_3y_excess_return = tool3.append_df(fund_manager_3y_excess_return)
        
        fund_manager_3y_excess_return = CALFUNC.del_dat_early_than(fund_manager_3y_excess_return, START_YEAR)
        #fund_manager_3y_excess_return.to_excel(os.path.join(basic_path, 'fund_manager_3y_excess_return.xlsx'))
        return fund_manager_3y_excess_return







    @lazyproperty
    def FUND_MANAGER_3Y_excess_return(self):
        #基金经理在管基金近三年区间净值超越基准收益率的简单平均
        data=Data()
        NAV_over_bench_return_per_3y=data.NAV_over_bench_return_per_3y
        #NAV_over_bench_return_per_3y=self.NAV_over_bench_return_per_3y
        NAV_over_bench_return_per_3y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,NAV_over_bench_return_per_3y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_3y_excess_return = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        #fund_manager_1y_excess_return=tool3.adjust_months(fund_manager_1y_excess_return,'Y')
        fund_manager_3y_excess_return = tool3.append_df(fund_manager_3y_excess_return)
        
        fund_manager_3y_excess_return = CALFUNC.del_dat_early_than(fund_manager_3y_excess_return, START_YEAR)
        #fund_manager_3y_excess_return.to_excel(os.path.join(basic_path, 'fund_manager_3y_excess_return.xlsx'))
        return fund_manager_3y_excess_return







    @lazyproperty
    def FUND_MANAGER_5Y_excess_return(self):
        #基金经理在管基金近五年区间净值超越基准收益率的简单平均
        data=Data()
        NAV_over_bench_return_per_5y=data.NAV_over_bench_return_per_5y
        #NAV_over_bench_return_per_5y=self.NAV_over_bench_return_per_5y
        NAV_over_bench_return_per_5y.reset_index(inplace=True)
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,NAV_over_bench_return_per_5y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_5y_excess_return = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        #fund_manager_1y_excess_return=tool3.adjust_months(fund_manager_1y_excess_return,'Y')
        fund_manager_5y_excess_return = tool3.append_df(fund_manager_5y_excess_return)
        
        fund_manager_5y_excess_return = CALFUNC.del_dat_early_than(fund_manager_5y_excess_return, START_YEAR)
        #fund_manager_5y_excess_return.to_excel(os.path.join(basic_path, 'fund_manager_5y_excess_return.xlsx'))
        return fund_manager_5y_excess_return




    @lazyproperty
    def FUND_MANAGER_RISK_AVGTRACKDEVIATION_BENCHMARK_1Y(self):
        #基金经理在管基金近一年区间跟踪偏离度均值（业绩基准）的简单平均
        data=Data()
        risk_avgtrackdeviation_benchmark_1y=data.risk_avgtrackdeviation_benchmark_1y
        #risk_avgtrackdeviation_benchmark_1y=self.risk_avgtrackdeviation_benchmark_1y
        fund_manager_detail=pd.read_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'),index_col=0)
        def as_datetime(x):
            if pd.isnull(x):
                return x
            else:
                x_date=datetime.strptime(str(x),'%Y-%m-%d')
                return x_date
        fund_manager_detail['start_date']=fund_manager_detail['start_date'].apply(as_datetime)
        fund_manager_detail['start_date_vaild']=fund_manager_detail['start_date_vaild'].apply(as_datetime)
        fund_manager_detail['end_date']=fund_manager_detail['end_date'].apply(as_datetime)
        df_temp=pd.merge(fund_manager_detail,risk_avgtrackdeviation_benchmark_1y,left_on='wind_code',right_on='wind_code',how='inner')
        df_temp=df_temp.apply(tool3.cut_data,axis=1)
        fund_manager_risk_avgtrackdeviation_benchmark_1y = df_temp.groupby(['manager_ID', 'firstinvesttype']).mean()
        
        #fund_manager_risk_avgtrackdeviation_benchmark_1y=tool3.adjust_months(fund_manager_risk_annutrackerror,'Y')
        fund_manager_risk_avgtrackdeviation_benchmark_1y = tool3.append_df(fund_manager_risk_avgtrackdeviation_benchmark_1y)
        
        
        fund_manager_risk_avgtrackdeviation_benchmark_1y = CALFUNC.del_dat_early_than(fund_manager_risk_avgtrackdeviation_benchmark_1y, START_YEAR)
        #fund_manager_risk_avgtrackdeviation_benchmark_1y.to_excel(os.path.join(basic_path, 'fund_manager_risk_avgtrackdeviation_benchmark_1y.xlsx'))
        return fund_manager_risk_avgtrackdeviation_benchmark_1y





def fund_manager_table():
    fund_manager_collection=pd.read_excel(date_dair+"\\fund\\download_from_wind\\基金经理大全.xlsx",header=1,index_col=None,usecols=None,squeeze=False)
    fund_manager_collection.rename(columns={'Unnamed: 0':'fund_manager', '性别':'gender', '出生年份':'birthyear', '年龄':'age', '学历':'education', '专业':'major', '毕业院校':'school', '国籍':'nationality', '简介':'resume',
       '基金经理年限':'managerworkingyears', '任职基金数':'fundno', '任职基金公司数':'fundcono', '几何平均年化收益率(%)':'geometricannualizedyield', '算术平均年化收益率(%)':'arithmeticannualizedyield',
       '超越基准几何平均年化收益率(%)':'geometricavgannualyieldoverbench', '超越基准算术平均年化收益率(%)':'arithmeticavgyieldoverbench', '基金公司':'fundco', '历任基金公司':'previousfundco'}, inplace = True)
    fund_manager_collection=fund_manager_collection[:-2]
    #版本二
    fund_manager_collection['manager_ID']=pd.DataFrame(np.array(range(0,len(fund_manager_collection)+1)))
    fund_manager_collection=fund_manager_collection.drop('previousfundco', axis=1).join(fund_manager_collection['previousfundco'].str.split(',',expand=True).stack().reset_index(level=1, drop=True).rename('previousfundco'))
    #版本一
    #fund_manager_collection=fund_manager_collection.reset_index()
    #fund_manager_collection.rename(columns={'index':'manager_ID'}, inplace = True)
    #engine = create_engine('mysql+pymysql://root:'+token_mysql+'@localhost:3306/fund',encoding='utf8') 
    #pd.io.sql.to_sql(fund_manager_collection, 'fund_manager_collection', con=engine, schema='fund', if_exists='replace' )
    fund_manager_collection.to_excel(os.path.join(basic_path, 'fund_manager_collection.xlsx'))
    
    
    
def fund_manager_basic_info2():
    #w.wss("000001.OF", "fund_fundmanageroftradedate,fund_corp_fundmanagementcompany,name_official,fund_manager_startdate,fund_manager_onthepostdays,fund_manager_enddate,fund_manager_gender,fund_manager_education,fund_manager_resume,fund_manager_age,fund_manager_managerworkingyears,NAV_periodicannualizedreturn","tradeDate=20201017;order=1;topNum=1")
    #w.start()
    #w.wss("320014.OF",  "fund_fullname,fund_mgrcomp,fund_setupdate,fund_predfundmanager,fund_fundmanager", usedf=True)[1]
    '''
    调用前先在wind中按照基金经理任职信息模板存储数据，保存为xlsx格式替换原数据！！！
    '''
    fund_manager_initial_info=pd.read_excel(date_dair+"\\fund\\download_from_wind\\fund_manager_initial_info.xlsx",header=0,index_col=0,usecols=None,squeeze=False)
    fund_manager_initial_info = fund_manager_initial_info.dropna(axis=0, how='all')
    fund_manager_initial_info=fund_manager_initial_info[fund_manager_initial_info['是否初始基金']=='是']
    fund_manager_initial_info=fund_manager_initial_info.drop('基金经理(历任)', axis=1).join(fund_manager_initial_info["基金经理(历任)"].str.split('\n',expand=True).stack().reset_index(level=1, drop=True).rename("基金经理"))
    lst=list(fund_manager_initial_info['基金经理'])
    lst_replaced=[str(sec).replace('至今','-*') for sec in lst]
    fund_manager_initial_info['基金经理']=lst_replaced
    lst2=list(fund_manager_initial_info['基金经理'])
    lst2_replaced=[re.split('[()-]',str(sec)) for sec in lst2]
    t=pd.DataFrame(lst2_replaced)
    t.index=fund_manager_initial_info.index
    fund_manager_initial_info=fund_manager_initial_info.drop(['基金经理','是否初始基金'], axis=1)
    fund_manager_initial_info['基金经理']=t[0]
    fund_manager_initial_info['任职日期']=t[1]
    fund_manager_initial_info['离职日期']=t[2]
    lst3=list(fund_manager_initial_info['基金成立日'])
    lst3_replaced=[]
    for day in lst3:
        if pd.isnull(day):
            lst3_replaced.append(np.nan)
        else:
            lst3_replaced.append(day.strftime("%Y-%m-%d"))
    fund_manager_initial_info['基金成立日']=lst3_replaced
    lst4=[]
    for x,y in zip(fund_manager_initial_info['基金成立日'],fund_manager_initial_info['任职日期']):
        if x==y:
            if pd.isnull(x):
                lst4.append(np.nan)
            else:                
                lst4.append((datetime.strptime(y,'%Y%m%d')+relativedelta(months=6)).strftime("%Y-%m-%d"))
        else:
            if pd.isnull(x):
                lst4.append(np.nan)
            else:
                lst4.append((datetime.strptime(y,'%Y%m%d')+relativedelta(months=3)).strftime("%Y-%m-%d"))
    fund_manager_initial_info['有效任职日期']=lst4
    lst5=list(fund_manager_initial_info['任职日期'])
    lst5_replaced=[]
    for day in lst5:
        if pd.isnull(day):
            lst5_replaced.append(np.nan)
        else:
            lst5_replaced.append(day[0:4]+'-'+day[4:6]+'-'+day[6:8])
    fund_manager_initial_info['任职日期']=lst5_replaced    
    lst6=list(fund_manager_initial_info['离职日期'])
    lst6_replaced=[]
    for day in lst6:
        if pd.isnull(day):
            lst6_replaced.append(np.nan)
        elif day=='*':
            lst6_replaced.append(np.nan)
        else:
            lst6_replaced.append(day[0:4]+'-'+day[4:6]+'-'+day[6:8])
    fund_manager_initial_info['离职日期']=lst6_replaced         
    fund_manager_basic_info=fund_manager_initial_info.reindex(columns=['证券简称','投资类型(一级分类)', '投资类型(二级分类)','基金管理人中文名称','基金经理','任职日期','有效任职日期','离职日期'])
    fund_manager_basic_info=fund_manager_basic_info.reset_index()
    fund_manager_basic_info.rename(columns={'证券代码':'wind_code','证券简称':'sec_name','投资类型(一级分类)':'firstinvesttype', '投资类型(二级分类)':'investtype','基金管理人中文名称':'fundco','基金经理':'fund_manager','任职日期':'start_date','有效任职日期':'start_date_vaild','离职日期':'end_date'}, inplace = True)
    #engine = create_engine('mysql+pymysql://root:'+token_mysql+'@localhost:3306/fund',encoding='utf8') 
    #pd.io.sql.to_sql(fund_manager_basic_info, 'fund_manager_detail', con=engine, schema='fund', if_exists='replace' )
    fund_manager_basic_info['manager_ID']=pd.DataFrame(np.zeros(len(fund_manager_basic_info)))
    fund_manager_basic_info['manager_ID']=np.nan
    fund_manager_collection=pd.read_excel(date_dair+"\\fund\\factor_data\\fund_manager_collection.xlsx",header=0,index_col=0,usecols=None,squeeze=False)
    fund_manager_collection.reset_index(drop=True, inplace=True)
    for i in range(len(fund_manager_basic_info)):
        for j in range(len(fund_manager_collection)):
            if fund_manager_basic_info.loc[i,'fund_manager']==fund_manager_collection.loc[j,'fund_manager']:
                if fund_manager_basic_info.loc[i,'fundco']==fund_manager_collection.loc[j,'previousfundco']:
                    fund_manager_basic_info.loc[i,'manager_ID']=fund_manager_collection.loc[j,'manager_ID']
                else:
                    continue
    fund_manager_basic_info.to_excel(os.path.join(basic_path, 'fund_manager_detail.xlsx'))




# =============================================================================
#草稿部分
# k1=fund_manager_collection['fund_manager']
# k2=fund_manager_collection['previousfundco']
# v=fund_manager_collection['manager_ID']
# 
# manager_dict={}
# for i in range(len(fund_manager_collection)):
#     
# manager_dict = {fund_manager_collection.loc[i,'fund_manager']:{fund_manager_collection.loc[i,'previousfundco']:fund_manager_collection.loc[i,'manager_ID']}}
# 
# manager_dict=dict(zip(k1,dict(zip(k2,v))))
# 
# 
# for k1,k2,v in zip(k1,dict(zip(k2,v))):
#     
# manager_dict={
#       '宋德舜':{'诺安基金管理有限公司':0}
#       }
# =============================================================================










def compute_factor(status):
    fc = Factor_Compute(status)

    factor_names = [k for k in Factor_Compute.__dict__.keys() if k.split('_')[0]!='']
    for f in factor_names:
        print(f)
        try:
            res = eval('fc.' + f)
            if isinstance(res, dict):
                for k, v in res.items():
                    fc.save(v, k,save_path=basic_path)
            if isinstance(res, pd.DataFrame):
                fc.save(res, f,save_path=basic_path)
            if (res is not None):
                continue
        except Exception as e:
            print('debug')


if __name__ == "__main__":
    #fund_manager_table()#调用compute_factor前必须先运行此行
    #fund_manager_basic_info2()#调用compute_factor前必须先运行此行
    #compute_factor('all')

    # 测试某个因子
    f='FUND_MANAGER_MAXDRAWDOWN_3M'# 这里可以且需要改因子名
    def saving(f):
        fc = Factor_Compute('all')  # 这里可以选择状态是从头算（all）还是只更新最后一列（update）
        factor_names = [f]
        for f in factor_names:
            print(f)
            if f == 'compute_pct_chg_nm':
                res = fc.compute_pct_chg_nm
                fc.save(res, 'pct_chg_nm'.upper())
            else:
                try:
                    res = eval('fc.' + f)  # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
                    if isinstance(res, dict):
                        # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                        # isinstance() 与 type() 区别：
                        # type() 不会认为子类是一种父类类型，不考虑继承关系。
                        # isinstance() 会认为子类是一种父类类型，考虑继承关系。
                        # 如果要判断两个类型是否相同推荐使用 isinstance()。
                        for k, v in res.items():
                            fc.save(v, k.upper())
                    if isinstance(res, pd.DataFrame):
                        fc.save(res, f.upper())
                    # if (res is not None):  # 返回None，表示无需更新
                    #     continue
                except Exception as e:
                    print('debug')
    saving(f)
    


