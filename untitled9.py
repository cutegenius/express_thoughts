# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:55:37 2021

@author: 37212
"""


def efficient_test_typestock(start_date=datetime(year=2009, month=1, day=1),end_date=datetime(year=2020, month=9, day=30)):
    data=Data()
    start_date=datetime(year=2009, month=1, day=1)
    end_date=end_date=datetime(year=2020, month=9, day=30)
    IC=dict()
    dict_len=len(fund_manager_factor_dict2)
    keys=list(fund_manager_factor_dict2.keys())
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
                    fund_manager_factor_dict2[key1]=factor1
                else:
                    factor1.columns=pd.DatetimeIndex(factor1.columns)
                    fund_manager_factor_dict2[key1]=factor1
            except Exception as e:
                fund_manager_factor_dict2[key1]=np.nan
        else:
            factor1=tool3.cleaning(factor1)
            if key1 in d_freq_factor_list:
                factor1=tool3.d_freq_to_m_freq(factor1)
                factor1.columns=pd.DatetimeIndex(factor1.columns)
                fund_manager_factor_dict2[key1]=factor1
            else:
                factor1.columns=pd.DatetimeIndex(factor1.columns)
                fund_manager_factor_dict2[key1]=factor1
                fund_manager_factor_dict2[key1]=factor1
    
    for month_end in month_ends:
        IC_dataframe=pd.DataFrame()
        n3m=pct_chg_of_fund_manager_index_n3m.loc['股票型基金',month_end]
        n6m=pct_chg_of_fund_manager_index_n6m.loc['股票型基金',month_end]
        n1y=pct_chg_of_fund_manager_index_n1y.loc['股票型基金',month_end]
        n2y=pct_chg_of_fund_manager_index_n2y.loc['股票型基金',month_end]
        for i in range(dict_len):
            key=keys[i]
            try:
                t=fund_manager_factor_dict2[key][month_end]
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

    
    
    
    IC_mean.to_excel(os.path.join(plot_path, 'IC_mean_stock2.xlsx'))
    IC_std.to_excel(os.path.join(plot_path, 'IC_std_stock2.xlsx'))
    IR.to_excel(os.path.join(plot_path, 'IR_stock2.xlsx'))
    IC_win.to_excel(os.path.join(plot_path, 'IC_win_stock2.xlsx'))