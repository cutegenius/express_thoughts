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


    #todo 目前这个因子的原始数据从wind中手动提取更新后期有必要时应该接口导入
    @lazyproperty
    def fund_manager_basic_info(self):
        #w.wss("000001.OF", "fund_fundmanageroftradedate,fund_corp_fundmanagementcompany,name_official,fund_manager_startdate,fund_manager_onthepostdays,fund_manager_enddate,fund_manager_gender,fund_manager_education,fund_manager_resume,fund_manager_age,fund_manager_managerworkingyears,NAV_periodicannualizedreturn","tradeDate=20201017;order=1;topNum=1")
        #w.start()
        #w.wss("320014.OF",  "fund_fullname,fund_mgrcomp,fund_setupdate,fund_predfundmanager,fund_fundmanager", usedf=True)[1]
        '''
        调用前先在wind中按照基金经理任职信息模板存储数据，保存为csv格式替换原数据！！！
        '''
        fund_manager_initial_info=self.fund_manager_initial_info
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
                lst3_replaced.append(day.strftime("%Y%m%d"))
        fund_manager_initial_info['基金成立日']=lst3_replaced
        lst4=[]
        for x,y in zip(fund_manager_initial_info['基金成立日'],fund_manager_initial_info['任职日期']):
            if x==y:
                if pd.isnull(x):
                    lst4.append(np.nan)
                else:                
                    lst4.append((datetime.strptime(str(x),'%Y%m%d')+relativedelta(months=6)).strftime("%Y%m%d"))
            else:
                if pd.isnull(x):
                    lst4.append(np.nan)
                else:
                    lst4.append((datetime.strptime(str(x),'%Y%m%d')+relativedelta(months=3)).strftime("%Y%m%d"))
        fund_manager_initial_info['有效任职日期']=lst4
        fund_manager_basic_info=fund_manager_initial_info.reindex(columns=['证券简称','基金管理人中文名称','基金经理','任职日期','有效任职日期','离职日期'])

        return fund_manager_basic_info


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
    compute_factor('all')


