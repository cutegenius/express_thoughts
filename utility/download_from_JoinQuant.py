import pandas as pd
import numpy as np
from collections import defaultdict
from utility.tool0 import Data
from utility.constant import date_dair
import os
from jqdatasdk import *
from WindPy import *
from datetime import *
from sqlalchemy import create_engine
import pymysql
#注：聚宽上说是限制只能取三千行的股票实际上每次都是限制能取出来五千行的股票，所以在代码中offset是5000而不是3000
token_path = r'D:\文档\OneDrive\earning money\入职后资料\token_joinquant.txt'
if os.path.exists(token_path):
    f = open(token_path)
    token = f.read()
    f.close()

token_path2 = r'D:\文档\OneDrive\earning money\入职后资料\token_mysql.txt'
if os.path.exists(token_path2):
    f = open(token_path2)
    token_mysql = f.read()
    f.close()

try:
    os.makedirs(date_dair + './fund' + './download_from_JoinQuant')
    basic_path = os.path.join(date_dair, 'fund', 'download_from_JoinQuant')
except Exception as e:
    basic_path = os.path.join(date_dair, 'fund', 'download_from_JoinQuant')


def trade_days(freq='d', interface='wind'):
    st = '20030102'
    ed = datetime.today()
    # 使用tushare数据接口
    if interface == 'tushare':
        days = pro.trade_cal(exchange='SHFE', start_date=st, end_date=ed.strftime("%Y%m%d"))
        days = days[days['is_open'] == 1]

        if freq == 'd':
            res = [datetime.strptime(i, "%Y%m%d") for i in days['cal_date']]
        elif freq == 'w':

            days['group_basic'] = None
            days = days.set_index('cal_date')
            days.index = pd.to_datetime(days.index)
            for i in days.index:
                days.loc[i, 'group_basic'] = i.strftime("%Y-%W")

            res = []
            grouped = days.groupby('group_basic')
            for i, v in grouped:
                res.append(v.index[-1])
    # 使用Wind数据接口
    elif interface == 'wind':
        w.start()
        _, days = w.tdays(st, ed, usedf=True)
        days.columns = ['cal_date']

        if freq == 'd':
            res = [i for i in days['cal_date']]
        elif freq == 'w':

            days['group_basic'] = None
            days = days.set_index('cal_date')
            days.index = pd.to_datetime(days.index)
            for i in days.index:
                days.loc[i, 'group_basic'] = i.strftime("%Y-%W")

            res = []
            grouped = days.groupby('group_basic')
            for i, v in grouped:
                res.append(v.index[-1])

    return res


def get_all_fund():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_fund=get_all_securities(['fund'])
    all_fund.to_csv(os.path.join(basic_path, 'all_fund.csv'), encoding='gbk')    

def get_all_open_fund():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_open_fund=get_all_securities(['open_fund'])
    all_open_fund.to_csv(os.path.join(basic_path, 'all_open_fund.csv'), encoding='gbk')
    
def get_all_etf():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_etf=get_all_securities(['etf'])
    all_etf.to_csv(os.path.join(basic_path, 'all_etf.csv'), encoding='gbk')  

def get_all_lof():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_lof=get_all_securities(['lof'])
    all_lof.to_csv(os.path.join(basic_path, 'all_lof.csv'), encoding='gbk')

def get_all_fja():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_fja=get_all_securities(['fja'])
    all_fja.to_csv(os.path.join(basic_path, 'all_fja.csv'), encoding='gbk')      
    
def get_all_fjb():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_fjb=get_all_securities(['fjb'])
    all_fjb.to_csv(os.path.join(basic_path, 'all_fjb.csv'), encoding='gbk')     

def get_all_bond_fund():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_bond_fund=get_all_securities(['bond_fund'])
    all_bond_fund.to_csv(os.path.join(basic_path, 'all_bond_fund.csv'), encoding='gbk')   

def get_all_stock_fund():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_stock_fund=get_all_securities(['stock_fund'])
    all_stock_fund.to_csv(os.path.join(basic_path, 'all_stock_fund.csv'), encoding='gbk')   

def get_all_QDII_fund():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_QDII_fund=get_all_securities(['QDII_fund'])
    all_QDII_fund.to_csv(os.path.join(basic_path, 'all_QDII_fund.csv'), encoding='gbk')   

def get_all_money_market_fund():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_money_market_fund=get_all_securities(['money_market_fund'])
    all_money_market_fund.to_csv(os.path.join(basic_path, 'all_money_market_fund.csv'), encoding='gbk')   

def get_all_mixture_fund():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    all_mixture_fund=get_all_securities(['mixture_fund'])
    all_mixture_fund.to_csv(os.path.join(basic_path, 'all_mixture_fund.csv'), encoding='gbk')  
    
def get_all_fund_info():
    w.start()
    data=Data()
    all_fund = data.all_fund
    all_open_fund = data.all_open_fund
    all_etf = data.all_etf
    all_lof = data.all_lof
    all_fja = data.all_fja
    all_fjb = data.all_fjb
    all_bond_fund = data.all_bond_fund
    all_stock_fund = data.all_stock_fund
    all_QDII_fund = data.all_QDII_fund
    all_money_market_fund = data.all_money_market_fund
    all_mixture_fund = data.all_mixture_fund
    all_fund_info=pd.concat([all_fund,all_open_fund,all_etf,all_lof,all_fja,all_fjb,all_bond_fund,all_stock_fund,all_QDII_fund,all_money_market_fund,all_mixture_fund])
    all_fund_info = all_fund_info.groupby(all_fund_info.index).first()
    date=datetime.today().strftime("%Y-%m-%d")
    all_wind_fund=w.wset("sectorconstituent","date="+date+";sectorid=1000008492000000").Data
    lst_wind=all_wind_fund[1]
    lst_JQ=list(all_fund_info.index.values)
    replace_dict={'XSHG':'SH','XSHE':'SZ'}
    replaced_lst_JQ=[]
    for sec in lst_JQ:
        for k,v in replace_dict.items():
            sec=sec.replace(k,v)
        replaced_lst_JQ.append(sec)
    #all_fund_info.reset_index(inplace=True,drop=True)
    all_fund_info.set_index(pd.Series(replaced_lst_JQ),inplace=True)  
    samestocks=list(set(lst_wind).intersection(set(replaced_lst_JQ)))
    all_fund_info=all_fund_info.reindex(samestocks)
    all_fund_info.to_csv(os.path.join(basic_path, 'all_fund_info.csv'), encoding='gbk')
    w.stop()

# 暂时采取全部使用wind接口提取基金指数等指数，后续如果有需要可以继续开发
# def get_all_index_info():
#     auth('15620560122',token)
#     if not is_auth():
#         print('JQData can  not be connect.')
#         return None
#     all_index=get_all_securities(['index'])
#     lst_JQ=list(all_index.index.values)
#     replace_dict={'XSHG':'SH','XSHE':'SZ'}
#     replaced_lst_JQ=[]
#     for sec in lst_JQ:
#         for k,v in replace_dict.items():
#             sec=sec.replace(k,v)
#         replaced_lst_JQ.append(sec)
    
def get_shares_info():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    tds=trade_days()
    st_date = datetime(2009, 1, 1)
    tds_turncate= [n for n in tds if n > st_date ]
    fund_share_daily=pd.DataFrame()
    for date in tds_turncate:
        df=finance.run_query(query(finance.FUND_SHARE_DAILY).filter(finance.FUND_SHARE_DAILY.date==date.strftime("%Y-%m-%d")))
        df=df.set_index('code')
        fund_share_daily = pd.concat([fund_share_daily, pd.DataFrame({date.strftime("%Y-%m-%d"): df['shares']})], axis=1)
    lst_JQ=list(fund_share_daily.index.values)
    replace_dict={'XSHG':'SH','XSHE':'SZ'}
    replaced_lst_JQ=[]
    for sec in lst_JQ:
        for k,v in replace_dict.items():
            sec=sec.replace(k,v)
        replaced_lst_JQ.append(sec)
    fund_share_daily.set_index(pd.Series(replaced_lst_JQ),inplace=True)
    fund_share_daily.to_csv(os.path.join(basic_path, 'fund_share_daily.csv'), encoding='gbk')
    
def get_main_info():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    q=query(finance.FUND_MAIN_INFO)
    df=finance.run_query(q)
    q2=query(finance.FUND_MAIN_INFO).offset(5000)
    df2=finance.run_query(q2)
    q3=query(finance.FUND_MAIN_INFO).offset(10000)
    df3=finance.run_query(q3)
    #目前查询三次之后可以把所有的基金取全，之后如有需要，可以重写函数
    fund_main_info=pd.concat([df,df2,df3])
    fund_main_info.set_index('id',inplace=True)
    fund_main_info.set_index('main_code',inplace=True)
    #多余步骤fund_main_info['operate_mode_id'] = fund_main_info['operate_mode_id'].map({401001:'开放式基金',401002:"封闭式基金",401003:"QDII",401004:"FOF",401005:"ETF",401006:"LOF"})
    index=fund_main_info.index.values.tolist()
    str_index=[]
    for ind in index:
        str_ind=ind+'.OF'
        str_index.append(str_ind)            
    fund_main_info.index=str_index
    fund_main_info.to_csv(os.path.join(basic_path, 'fund_main_info.csv'), encoding='gbk')



def get_fund_net_value():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    tds=trade_days()
    st_date = datetime(2009, 1, 1)
    tds_turncate= [n for n in tds if n > st_date ]
    net_value=pd.DataFrame()
    sum_value=pd.DataFrame()
    factor=pd.DataFrame()
    acc_factor=pd.DataFrame()
    refactor_net_value=pd.DataFrame()
    for date in tds_turncate:
        q=query(finance.FUND_NET_VALUE).filter(finance.FUND_NET_VALUE.day==date.strftime("%Y-%m-%d"))
        df=finance.run_query(q)
        q2=query(finance.FUND_NET_VALUE).filter(finance.FUND_NET_VALUE.day==date.strftime("%Y-%m-%d")).offset(5000)
        df2=finance.run_query(q2)
        q3=query(finance.FUND_NET_VALUE).filter(finance.FUND_NET_VALUE.day==date.strftime("%Y-%m-%d")).offset(10000)
        df3=finance.run_query(q3)    
        #目前查询三次之后可以把所有的基金取全，之后如有需要，可以重写函数
        fund_net_value_info=pd.concat([df,df2,df3]) 
        fund_net_value_info.set_index('id',inplace=True)
        fund_net_value_info.set_index('code',inplace=True)
        index=fund_net_value_info.index.values.tolist()
        str_index=[]
        for ind in index:
            str_ind=ind+'.OF'
            str_index.append(str_ind)            
        fund_net_value_info.index=str_index
        net_value=pd.concat([net_value, pd.DataFrame({date.strftime("%Y-%m-%d"): fund_net_value_info['net_value']})], axis=1)
        sum_value=pd.concat([sum_value, pd.DataFrame({date.strftime("%Y-%m-%d"): fund_net_value_info['sum_value']})], axis=1)
        factor=pd.concat([factor, pd.DataFrame({date.strftime("%Y-%m-%d"): fund_net_value_info['factor']})], axis=1)
        acc_factor=pd.concat([acc_factor, pd.DataFrame({date.strftime("%Y-%m-%d"): fund_net_value_info['acc_factor']})], axis=1)
        refactor_net_value=pd.concat([refactor_net_value, pd.DataFrame({date.strftime("%Y-%m-%d"): fund_net_value_info['refactor_net_value']})], axis=1)
    net_value.to_csv(os.path.join(basic_path, 'net_value.csv'), encoding='gbk')
    sum_value.to_csv(os.path.join(basic_path, 'sum_value.csv'), encoding='gbk')
    factor.to_csv(os.path.join(basic_path, 'factor.csv'), encoding='gbk') 
    acc_factor.to_csv(os.path.join(basic_path, 'acc_factor.csv'), encoding='gbk')
    refactor_net_value.to_csv(os.path.join(basic_path, 'refactor_net_value.csv'), encoding='gbk')
    

def get_main_info():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    q=query(finance.FUND_MAIN_INFO)
    df=finance.run_query(q)
    q2=query(finance.FUND_MAIN_INFO).offset(5000)
    df2=finance.run_query(q2)
    q3=query(finance.FUND_MAIN_INFO).offset(10000)
    df3=finance.run_query(q3)
    #目前查询三次之后可以把所有的基金取全，之后如有需要，可以重写函数
    fund_main_info=pd.concat([df,df2,df3])
    fund_main_info.set_index('id',inplace=True)
    fund_main_info.set_index('main_code',inplace=True)
    #多余步骤fund_main_info['operate_mode_id'] = fund_main_info['operate_mode_id'].map({401001:'开放式基金',401002:"封闭式基金",401003:"QDII",401004:"FOF",401005:"ETF",401006:"LOF"})
    index=fund_main_info.index.values.tolist()
    str_index=[]
    for ind in index:
        str_ind=ind+'.OF'
        str_index.append(str_ind)
    fund_main_info.index=str_index
    fund_main_info.to_csv(os.path.join(basic_path, 'fund_main_info.csv'), encoding='gbk')
    
def get_fund_portfolio_stock():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    i=0
    length=5000
    fund_portfolio_stock=pd.DataFrame()
    while length==5000:
        df=finance.run_query(query(finance.FUND_PORTFOLIO_STOCK).offset(5000*i))
        fund_portfolio_stock=pd.concat([fund_portfolio_stock,df])
        length=len(df)
        i=i+1
    #fund_portfolio_stock.to_csv(os.path.join(basic_path, 'fund_portfolio_stock.txt'), sep='\t',encoding='gbk')
    engine = create_engine('mysql+pymysql://root:'+token_mysql+'@localhost:3306/fund',encoding='utf8') 
    pd.io.sql.to_sql(fund_portfolio_stock, 'fund_portfolio_stock', con=engine, schema='fund', if_exists='replace' )
    

def get_fund_portfolio():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    period_ends=[]
    for year in range(2009,2021):
        for i in range(3, 13, 3):
            if i==3:
                period_ends.append(str(year)+'-03-31')
            if i==6:
                period_ends.append(str(year)+'-06-30')
            if i==9:
                period_ends.append(str(year)+'-09-30')
            if i==12:
                period_ends.append(str(year)+'-12-31')
    equity_value=pd.DataFrame()
    equity_rate=pd.DataFrame()
    stock_value=pd.DataFrame()
    stock_rate=pd.DataFrame()
    fixed_income_value=pd.DataFrame()
    fixed_income_rate=pd.DataFrame()
    precious_metal_value=pd.DataFrame()
    precious_metal_rate=pd.DataFrame()
    derivative_value=pd.DataFrame()
    derivative_rate=pd.DataFrame()
    buying_back_value=pd.DataFrame()
    buying_back_rate=pd.DataFrame()
    deposit_value=pd.DataFrame()
    deposit_rate=pd.DataFrame()
    others_value=pd.DataFrame()
    others_rate=pd.DataFrame()
    total_asset=pd.DataFrame()
    for date in period_ends:
        q=query(finance.FUND_PORTFOLIO).filter(finance.FUND_PORTFOLIO.period_end==date)
        df=finance.run_query(q)
        q2=query(finance.FUND_PORTFOLIO).filter(finance.FUND_PORTFOLIO.period_end==date).offset(5000)
        df2=finance.run_query(q2)
        q3=query(finance.FUND_PORTFOLIO).filter(finance.FUND_PORTFOLIO.period_end==date).offset(10000)
        df3=finance.run_query(q3)    
        #目前查询三次之后可以把所有的基金取全，之后如有需要，可以重写函数
        fund_portfolio_info=pd.concat([df,df2,df3]) 
        fund_portfolio_info.set_index('id',inplace=True)
        fund_portfolio_info.set_index('code',inplace=True)
        index=fund_portfolio_info.index.values.tolist()
        str_index=[]
        for ind in index:
            str_ind=ind+'.OF'
            str_index.append(str_ind)            
        fund_portfolio_info.index=str_index
        fund_portfolio_info = fund_portfolio_info.groupby(fund_portfolio_info.index).first()
        equity_value=pd.concat([equity_value, pd.DataFrame({date:fund_portfolio_info['equity_value']})], axis=1)
        equity_rate=pd.concat([equity_rate, pd.DataFrame({date:fund_portfolio_info['equity_rate']})], axis=1)
        stock_value=pd.concat([stock_value, pd.DataFrame({date:fund_portfolio_info['stock_value']})], axis=1)
        stock_rate=pd.concat([stock_rate, pd.DataFrame({date:fund_portfolio_info['stock_rate']})], axis=1)
        fixed_income_value=pd.concat([fixed_income_value, pd.DataFrame({date:fund_portfolio_info['fixed_income_value']})], axis=1)
        fixed_income_rate=pd.concat([fixed_income_rate, pd.DataFrame({date:fund_portfolio_info['fixed_income_rate']})], axis=1)
        precious_metal_value=pd.concat([precious_metal_value, pd.DataFrame({date:fund_portfolio_info['precious_metal_value']})], axis=1)
        precious_metal_rate=pd.concat([precious_metal_rate, pd.DataFrame({date:fund_portfolio_info['precious_metal_rate']})], axis=1)
        derivative_value=pd.concat([derivative_value, pd.DataFrame({date:fund_portfolio_info['derivative_value']})], axis=1)
        derivative_rate=pd.concat([derivative_rate, pd.DataFrame({date:fund_portfolio_info['derivative_rate']})], axis=1)
        buying_back_value=pd.concat([buying_back_value, pd.DataFrame({date:fund_portfolio_info['buying_back_value']})], axis=1)
        buying_back_rate=pd.concat([buying_back_rate, pd.DataFrame({date:fund_portfolio_info['buying_back_rate']})], axis=1)
        deposit_value=pd.concat([deposit_value, pd.DataFrame({date:fund_portfolio_info['deposit_value']})], axis=1)
        deposit_rate=pd.concat([deposit_rate, pd.DataFrame({date:fund_portfolio_info['deposit_rate']})], axis=1)
        others_value=pd.concat([others_value, pd.DataFrame({date:fund_portfolio_info['others_value']})], axis=1)
        others_rate=pd.concat([others_rate, pd.DataFrame({date:fund_portfolio_info['others_rate']})], axis=1)
        total_asset=pd.concat([total_asset, pd.DataFrame({date:fund_portfolio_info['total_asset']})], axis=1)
    equity_value.to_csv(os.path.join(basic_path, 'equity_value.csv'), encoding='gbk')
    equity_rate.to_csv(os.path.join(basic_path, 'equity_rate.csv'), encoding='gbk')
    stock_value.to_csv(os.path.join(basic_path, 'stock_value.csv'), encoding='gbk')
    stock_rate.to_csv(os.path.join(basic_path, 'stock_rate.csv'), encoding='gbk')
    fixed_income_value.to_csv(os.path.join(basic_path, 'fixed_income_value.csv'), encoding='gbk')
    fixed_income_rate.to_csv(os.path.join(basic_path, 'fixed_income_rate.csv'), encoding='gbk')
    precious_metal_value.to_csv(os.path.join(basic_path, 'precious_metal_value.csv'), encoding='gbk')
    precious_metal_rate.to_csv(os.path.join(basic_path, 'precious_metal_rate.csv'), encoding='gbk')
    derivative_value.to_csv(os.path.join(basic_path, 'derivative_value.csv'), encoding='gbk')
    derivative_rate.to_csv(os.path.join(basic_path, 'derivative_rate.csv'), encoding='gbk')
    deposit_value.to_csv(os.path.join(basic_path, 'deposit_value.csv'), encoding='gbk')
    deposit_rate.to_csv(os.path.join(basic_path, 'deposit_rate.csv'), encoding='gbk')
    others_value.to_csv(os.path.join(basic_path, 'others_value.csv'), encoding='gbk')
    others_rate.to_csv(os.path.join(basic_path, 'others_rate.csv'), encoding='gbk')
    total_asset.to_csv(os.path.join(basic_path, 'total_asset.csv'), encoding='gbk')    
    buying_back_value.to_csv(os.path.join(basic_path, 'buying_back_value.csv'), encoding='gbk')
    buying_back_rate.to_csv(os.path.join(basic_path, 'buying_back_rate.csv'), encoding='gbk')
    

def get_fund_portfolio_bond():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    i=0
    length=5000
    fund_portfolio_bond=pd.DataFrame()
    while length==5000:
        df=finance.run_query(query(finance.FUND_PORTFOLIO_BOND).offset(5000*i))
        fund_portfolio_bond=pd.concat([fund_portfolio_bond,df])
        length=len(df)
        i=i+1
    #fund_portfolio_stock.to_csv(os.path.join(basic_path, 'fund_portfolio_stock.txt'), sep='\t',encoding='gbk')
    engine = create_engine('mysql+pymysql://root:'+token_mysql+'@localhost:3306/fund',encoding='utf8') 
    pd.io.sql.to_sql(fund_portfolio_bond, 'fund_portfolio_bond', con=engine, schema='fund', if_exists='replace' )


def get_fund_dividend():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None
    i=0
    length=5000
    fund_dividend=pd.DataFrame()
    while length==5000:
        df=finance.run_query(query(finance.FUND_DIVIDEND).offset(5000*i))
        fund_dividend=pd.concat([fund_dividend,df])
        length=len(df)
        i=i+1
    #fund_portfolio_stock.to_csv(os.path.join(basic_path, 'fund_portfolio_stock.txt'), sep='\t',encoding='gbk')
    engine = create_engine('mysql+pymysql://root:'+token_mysql+'@localhost:3306/fund',encoding='utf8') 
    pd.io.sql.to_sql(fund_dividend, 'fund_dividend', con=engine, schema='fund', if_exists='replace' )
 
    
def get_fund_fin_indicator():
    auth('15620560122',token)
    if not is_auth():
        print('JQData can  not be connect.')
        return None    
    period_ends=[]
    for year in range(2009,2021):
        for i in range(3, 13, 3):
            if i==3:
                period_ends.append(str(year)+'-03-31')
            if i==6:
                period_ends.append(str(year)+'-06-30')
            if i==9:
                period_ends.append(str(year)+'-09-30')
            if i==12:
                period_ends.append(str(year)+'-12-31')
    #季报年报均披露的财务数据按照季报数据存储，仅有年报、半年报才披露的财务数据按照年报半年报数据存储
    profit=pd.DataFrame()
    adjust_profit=pd.DataFrame()
    avg_profit=pd.DataFrame()
    avg_roe=pd.DataFrame()
    profit_available=pd.DataFrame()
    profit_avaialbe_per_share=pd.DataFrame()
    total_tna=pd.DataFrame()
    nav=pd.DataFrame()
    adjust_nav=pd.DataFrame()
    nav_growth=pd.DataFrame()
    acc_nav_growth=pd.DataFrame()
    adjust_nav_growth=pd.DataFrame()
    total_asset=pd.DataFrame()
    for date in period_ends:
        if date[5:7] in ('06','12'):            
            q=query(finance.FUND_FIN_INDICATOR).filter(finance.FUND_FIN_INDICATOR.period_end==date,finance.FUND_FIN_INDICATOR.report_type.like('%%年度'))
            df=finance.run_query(q)
            q2=query(finance.FUND_FIN_INDICATOR).filter(finance.FUND_FIN_INDICATOR.period_end==date,finance.FUND_FIN_INDICATOR.report_type.like('%%年度')).offset(5000)
            df2=finance.run_query(q2)
            q3=query(finance.FUND_FIN_INDICATOR).filter(finance.FUND_FIN_INDICATOR.period_end==date,finance.FUND_FIN_INDICATOR.report_type.like('%%年度')).offset(10000)
            df3=finance.run_query(q3)    
            #目前查询三次之后可以把所有的基金取全，之后如有需要，可以重写函数
            fund_fin_indicator_yearly=pd.concat([df,df2,df3]) 
            fund_fin_indicator_yearly.set_index('id',inplace=True)
            fund_fin_indicator_yearly.set_index('code',inplace=True)
            index=fund_fin_indicator_yearly.index.values.tolist()
            str_index=[]
            for ind in index:
                str_ind=ind+'.OF'
                str_index.append(str_ind)            
            fund_fin_indicator_yearly.index=str_index
            fund_fin_indicator_yearly = fund_fin_indicator_yearly.groupby(fund_fin_indicator_yearly.index).first()
            avg_roe=pd.concat([avg_roe, pd.DataFrame({date:fund_fin_indicator_yearly['avg_roe']})], axis=1)
            profit_available=pd.concat([profit_available, pd.DataFrame({date:fund_fin_indicator_yearly['profit_available']})], axis=1)
            profit_avaialbe_per_share=pd.concat([profit_avaialbe_per_share, pd.DataFrame({date:fund_fin_indicator_yearly['profit_avaialbe_per_share']})], axis=1)
            adjust_nav=pd.concat([adjust_nav, pd.DataFrame({date:fund_fin_indicator_yearly['adjust_nav']})], axis=1)
            nav_growth=pd.concat([nav_growth, pd.DataFrame({date:fund_fin_indicator_yearly['nav_growth']})], axis=1)
            acc_nav_growth=pd.concat([acc_nav_growth, pd.DataFrame({date:fund_fin_indicator_yearly['acc_nav_growth']})], axis=1)
            adjust_nav_growth=pd.concat([adjust_nav_growth, pd.DataFrame({date:fund_fin_indicator_yearly['adjust_nav_growth']})], axis=1)
            total_asset=pd.concat([total_asset, pd.DataFrame({date:fund_fin_indicator_yearly['total_asset']})], axis=1)
        if date[5:7] in ('03','06','09','12'):
        #注：此处一定要用if而不能用elif，用elif或者else均会吧06，12月份排除在外
            
            q=query(finance.FUND_FIN_INDICATOR).filter(finance.FUND_FIN_INDICATOR.period_end==date,finance.FUND_FIN_INDICATOR.report_type.like('%%季度'))
            df=finance.run_query(q)
            q2=query(finance.FUND_FIN_INDICATOR).filter(finance.FUND_FIN_INDICATOR.period_end==date,finance.FUND_FIN_INDICATOR.report_type.like('%%季度')).offset(5000)
            df2=finance.run_query(q2)
            q3=query(finance.FUND_FIN_INDICATOR).filter(finance.FUND_FIN_INDICATOR.period_end==date,finance.FUND_FIN_INDICATOR.report_type.like('%%季度')).offset(10000)
            df3=finance.run_query(q3)    
            #目前查询三次之后可以把所有的基金取全，之后如有需要，可以重写函数
            fund_fin_indicator_seasonal=pd.concat([df,df2,df3]) 
            fund_fin_indicator_seasonal.set_index('id',inplace=True)
            fund_fin_indicator_seasonal.set_index('code',inplace=True)
            index=fund_fin_indicator_seasonal.index.values.tolist()
            str_index=[]
            for ind in index:
                str_ind=ind+'.OF'
                str_index.append(str_ind)            
            fund_fin_indicator_seasonal.index=str_index
            fund_fin_indicator_seasonal = fund_fin_indicator_seasonal.groupby(fund_fin_indicator_seasonal.index).first()
            profit=pd.concat([profit, pd.DataFrame({date:fund_fin_indicator_seasonal['profit']})], axis=1)
            adjust_profit=pd.concat([adjust_profit, pd.DataFrame({date:fund_fin_indicator_seasonal['adjust_profit']})], axis=1)
            avg_profit=pd.concat([avg_profit, pd.DataFrame({date:fund_fin_indicator_seasonal['avg_profit']})], axis=1)
            total_tna=pd.concat([total_tna, pd.DataFrame({date:fund_fin_indicator_seasonal['total_tna']})], axis=1)
            nav=pd.concat([nav, pd.DataFrame({date:fund_fin_indicator_seasonal['nav']})], axis=1)       
    avg_roe.to_csv(os.path.join(basic_path, 'avg_roe.csv'), encoding='gbk')
    profit_available.to_csv(os.path.join(basic_path, 'profit_available.csv'), encoding='gbk')
    profit_avaialbe_per_share.to_csv(os.path.join(basic_path, 'profit_avaialbe_per_share.csv'), encoding='gbk')
    adjust_nav.to_csv(os.path.join(basic_path, 'adjust_nav.csv'), encoding='gbk')
    adjust_nav_growth.to_csv(os.path.join(basic_path, 'adjust_nav_growth.csv'), encoding='gbk')
    nav_growth.to_csv(os.path.join(basic_path, 'nav_growth.csv'), encoding='gbk')
    acc_nav_growth.to_csv(os.path.join(basic_path, 'acc_nav_growth.csv'), encoding='gbk')
    total_asset.to_csv(os.path.join(basic_path, 'total_asset.csv'), encoding='gbk')
    profit.to_csv(os.path.join(basic_path, 'profit.csv'), encoding='gbk')
    adjust_profit.to_csv(os.path.join(basic_path, 'adjust_profit.csv'), encoding='gbk')
    avg_profit.to_csv(os.path.join(basic_path, 'avg_profit.csv'), encoding='gbk')
    total_tna.to_csv(os.path.join(basic_path, 'total_tna.csv'), encoding='gbk')
    nav.to_csv(os.path.join(basic_path, 'nav.csv'), encoding='gbk')

    
