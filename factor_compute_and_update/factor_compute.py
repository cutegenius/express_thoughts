from copy import deepcopy
import sys
import pandas as pd
import numpy as np
import os
from itertools import chain
from functools import reduce
import statsmodels.api as sm
from datetime import datetime, timedelta
# from pyfinance.ols import PandasRollingOLS as rolling_ols
from pyfinance.utils import rolling_windows
from utility.tool0 import Data, scaler
from utility.factor_data_preprocess import add_to_panels, align
from utility.relate_to_tushare import generate_months_ends
from utility.tool1 import CALFUNC, _calculate_su_simple, parallelcal,  lazyproperty, time_decorator, \
    get_signal_season_value, get_fill_vals, linear_interpolate, get_season_mean_value
from utility.constant import data_dair, root_dair
from utility.tool3 import adjust_months, add_to_panels, append_df

START_YEAR = 2009


class Factor_Compute(CALFUNC):

    def __init__(self, status):
        super().__init__()
        # status = 'update' 表示仅对已有的因子值进行更新， ='all' 表示全部重新计算
        self._mes = generate_months_ends()
        self._status = status

    def _get_update_month(self, fn):
        factor_m = eval('self.' + fn)
        # factor_m = self.RETURN_12M
        last_dt = pd.to_datetime(factor_m.columns[-1])
        to_update_month_list = [i for i in self._mes if i > last_dt]
        if len(to_update_month_list) == 0:
            print('没有更新必要')
            return None
            # sys.exit()
        else:
            return to_update_month_list

    @lazyproperty
    def compute_pct_chg_nm(self):

        pct = self.changepct_daily
        pct = 1 + pct/100

        mes1 = [m for m in self._mes if m in pct.columns]
        pct_chg = pd.DataFrame()
        for m in mes1:
            cols = [c for c in pct.columns if c.year == m.year and c.month == m.month]
            tmp_df = pct[cols]
            tmp_cum = tmp_df.cumprod(axis=1)
            res_df_t = tmp_cum[[tmp_cum.columns[-1]]] - 1
            pct_chg = pd.concat([pct_chg, res_df_t], axis=1)

        pct_chg = pct_chg * 100
        pct_chg_nm = pct_chg.shift(-1, axis=1)
        pct_chg_nm = CALFUNC.del_dat_early_than(pct_chg_nm, START_YEAR)

        return pct_chg_nm

    @lazyproperty
    def is_open1(self):
        open = self.openprice_daily
        high = self.highprice_daily
        low = self.lowprice_daily

        if self._status == 'all':
            # 不是停牌的
            is_open = ~pd.isna(open)
            # 不是开盘涨跌停的
            tmp1 = open == high
            tmp2 = high == low
            tmp = ~(tmp1 & tmp2)

            is_open = tmp & is_open

            is_open = CALFUNC.d_freq_to_m_freq(is_open, shift=True)
            is_open = CALFUNC.del_dat_early_than(is_open, START_YEAR)
        elif self._status == 'update':
            factor = self.IS_OPEN
            # 先删除过去计算的bug
            to_del = [c for c in factor.columns if c not in self._mes]
            factor.drop(to_del, axis=1, inplace=True)

            latest_dt = factor.columns[-1]
            # 删除无用的日频数据
            saved_cols = [i for i in open.columns if i > latest_dt]
            open = open[saved_cols]
            high = high[saved_cols]
            low = low[saved_cols]

            is_open = ~pd.isna(open)
            # 不是开盘涨跌停的
            tmp1 = open == high
            tmp2 = high == low
            tmp = ~(tmp1 & tmp2)

            is_open = tmp & is_open
            is_open = CALFUNC.d_freq_to_m_freq(is_open, shift=True)
            is_open = pd.concat([factor, is_open], axis=1)

        return is_open

    @lazyproperty
    def liquidity_barra(self):

        totalmv = self.totalmv_daily                 # 流通市值（万元）
        turnovervalue = self.turnovervalue_daily     # 成交额（万元）

        totalmv, turnovervalue = align(totalmv, turnovervalue)

        share_turnover = turnovervalue / totalmv

        new_mes = [m for m in self._mes if m in share_turnover.columns]

        def t_fun(tmp_df, freq=1):
            tmp_ar = tmp_df.values
            sentinel = -1e10
            res = np.log(np.nansum(tmp_ar, axis=1) / freq)
            res = np.where(np.isinf(res), sentinel, res)
            res_df = pd.DataFrame(data=res, index=tmp_df.index, columns=[tmp_df.columns[-1]])
            return res_df

        stom = pd.DataFrame()
        stoq = pd.DataFrame()
        stoa = pd.DataFrame()

        for m in new_mes:
            loc = np.where(share_turnover.columns == m)[0][0]
            if loc > 12 * 21:
                res_df1 = t_fun(share_turnover.iloc[:, loc+1 - 21:loc+1], 1)
                res_df3 = t_fun(share_turnover.iloc[:, loc+1 - 3*21:loc+1], 3)
                res_df12 = t_fun(share_turnover.iloc[:, loc+1 - 12*21:loc+1], 12)

                stom = pd.concat([stom, res_df1], axis=1)
                stoq = pd.concat([stoq, res_df3], axis=1)
                stoa = pd.concat([stoa, res_df12], axis=1)

        stom = CALFUNC.del_dat_early_than(stom, START_YEAR)
        stoq = CALFUNC.del_dat_early_than(stoq, START_YEAR)
        stoa = CALFUNC.del_dat_early_than(stoa, START_YEAR)

        res_dict = {"STOM_BARRA": stom,
                    "STOQ_BARRA": stoq,
                    "STOA_BARRA": stoa,
                    }

        return res_dict

    @lazyproperty
    def West_netprofit_yoy(self):
        west = self.west_netprofit_yoy
        west.fillna(0, inplace=True)
        return west

    @lazyproperty
    def beta(self):

        y = self.changepct_daily.T / 100
        index_p = self.index_price_daily
        index_p = index_p.loc['HS300', :]
        index_r = index_p/index_p.shift(1) - 1
        index_r = index_r.dropna()

        new_index = [i for i in y.index if i in index_r.index]
        y = y.loc[new_index, :]

        new_mes = [m for m in self._mes if m in y.index and np.where(y.index == m)[0][0] > 504]
        b, alpha, sigma = self._rolling_regress(y, index_r, window=504, half_life=252, target_date=new_mes)

        return b

    # 流通市值
    @lazyproperty
    def Mkt_cap_float(self):
        negotiablemv = self.negotiablemv_daily
        negotiablemv = CALFUNC.d_freq_to_m_freq(negotiablemv)
        res = CALFUNC.del_dat_early_than(negotiablemv, START_YEAR)
        return res

    # 规模因子
    @lazyproperty
    def LNCAP_Barra(self):
        lncap = np.log(self.negotiablemv_daily * 10000)
        lncap = CALFUNC.d_freq_to_m_freq(lncap)
        lncap = CALFUNC.del_dat_early_than(lncap, START_YEAR)
        return lncap

    @lazyproperty
    def MIDCAP_barra(self):
        lncap = np.log(self.negotiablemv_daily * 10000)
        lncap = CALFUNC.d_freq_to_m_freq(lncap)
        y = lncap ** 3
        X = lncap
        y = y.T
        X = X.T

        resid = pd.DataFrame()
        for code in y.columns:
            y_ = y[[code]]
            x_ = X[[code]]
            x_['const'] = 1
            dat = pd.concat([x_, y_], axis=1)
            dat = dat.dropna(how='any', axis=0)
            X_, y_ = dat.iloc[:, :-1], dat.iloc[:, -1:]

            if len(y_) > 0:
                model = sm.WLS(y_, X_)
                result = model.fit()

                params_ = result.params
                resid_ = y_ - pd.DataFrame(np.dot(X_, params_), index=y_.index,
                                           columns=[code])
            else:
                resid_ = pd.DataFrame([np.nan] * len(y), index=y.index, columns=[code])

            resid = pd.concat([resid, resid_], axis=1)

        resid = resid.T
        resid = CALFUNC.del_dat_early_than(resid, START_YEAR)

        return resid

    @lazyproperty
    def Std_nm(self):
        # n分别为1、3、6、12，每个月为21个交易日
        pct = self.changepct_daily/100

        new_mes = [m for m in self._mes if m in pct.columns]

        std_1m = pd.DataFrame()
        std_3m = pd.DataFrame()
        std_6m = pd.DataFrame()
        std_12m = pd.DataFrame()

        for m in new_mes:
            loc = np.where(pct.columns == m)[0][0]
            if loc > 12 * 21:
                res_df1 = pct.iloc[:, loc + 1 - 21:loc + 1].std(axis=1)        # 对DF使用std，会自动处理nan
                res_df3 = pct.iloc[:, loc + 1 - 3 * 21:loc + 1].std(axis=1)
                res_df6 = pct.iloc[:, loc + 1 - 3 * 21:loc + 1].std(axis=1)
                res_df12 = pct.iloc[:, loc + 1 - 12 * 21:loc + 1].std(axis=1)

                std_1m = pd.concat([std_1m, pd.DataFrame({m: res_df1})], axis=1)
                std_3m = pd.concat([std_3m, pd.DataFrame({m: res_df3})], axis=1)
                std_6m = pd.concat([std_6m, pd.DataFrame({m: res_df6})], axis=1)
                std_12m = pd.concat([std_12m, pd.DataFrame({m: res_df12})], axis=1)

        std_1m = CALFUNC.del_dat_early_than(std_1m, START_YEAR)
        std_3m = CALFUNC.del_dat_early_than(std_3m, START_YEAR)
        std_6m = CALFUNC.del_dat_early_than(std_6m, START_YEAR)
        std_12m = CALFUNC.del_dat_early_than(std_12m, START_YEAR)

        res_dict = {"Std_1m": std_1m,
                    "Std_3m": std_3m,
                    "Std_6m": std_6m,
                    "Std_12m": std_12m,
                    }

        return res_dict

    # 高管薪酬前三的合
    @lazyproperty
    def mgmt_ben_top3m(self):
        mgmt_ben_top3m = self.stmnote_mgmt_ben_top3m
        mgmt_ben_top3m = np.log(mgmt_ben_top3m)
        mgmt_ben_top3m = adjust_months(mgmt_ben_top3m, orig='Y')
        mgmt_ben_top3m = append_df(mgmt_ben_top3m)

        return mgmt_ben_top3m

    # 盈利预测机构家数
    @lazyproperty
    def Est_instnum(self):
        print(self.__dict__)
        est_instnum = self.est_instnum
        # 该数据缺失较多，对Nan的直接用0替代
        est_instnum = est_instnum.fillna(0)
        # 取根号
        est_instnum = est_instnum ** 0.5
        est_instnum = append_df(est_instnum)
        return est_instnum

    @lazyproperty
    def reverse_nm(self):
        '''
        1）在每个月底，对于股票s，回溯其过去N个交易日的数据（为方便处理， N取偶数）；
        2）对于股票s，逐日计算平均单笔成交金额D（D = 当日成交金额 / 当日成交笔数），将N个交易日按D值从大到小排序，前N/2
          个交易日称为高D组，后N/2个交易日称为低D组；
        3）对于股票s，将高D组交易日的涨跌幅加总[1]，得到因子M_high；将低D组交易日的涨跌幅加总，得到因子M_low；
        4） 对于所有股票，分别按照上述流程计算因子值。
        '''
        # n为20、60、180
        deals = self.turnoverdeals
        turnovervalue = self.turnovervalue_daily  # 成交额（万元）
        turnovervalue, deals = align(turnovervalue, deals)

        value_per_deal = turnovervalue/deals
        pct = self.changepct_daily / 100

        value_per_deal, pct = align(value_per_deal, pct)

        def _cal_M_reverse(series, pct_chg=None):
            code = series.name
            series = series.dropna()
            if len(series) == 0:
                return None

            series = series.sort_values()
            if len(series) % 2 == 1:
                low_vals = series.iloc[:len(series) // 2 + 1]
            else:
                low_vals = series.iloc[:len(series) // 2]
            high_vals = series.iloc[len(series) // 2:]
            m_high = (pct_chg.loc[code, high_vals.index] + 1).cumprod().iloc[-1] - 1
            m_low = (pct_chg.loc[code, low_vals.index] + 1).cumprod().iloc[-1] - 1
            res = m_high - m_low

            return res

        if self._status == 'update':
            new_mes = self._get_update_month('M_REVERSE_20')
            # 若返回None，表示没有更新必要，因子计算函数同样返回None
            if not new_mes:
                return None

            reverse_20 = self.M_REVERSE_20
            reverse_60 = self.M_REVERSE_60
            reverse_180 = self.M_REVERSE_180

        elif self._status == 'all':
            new_mes = [m for m in self._mes if m in value_per_deal.columns]
            reverse_20 = pd.DataFrame()
            reverse_60 = pd.DataFrame()
            reverse_180 = pd.DataFrame()

        for m in new_mes:
            print(m)
            loc = np.where(value_per_deal.columns == m)[0][0]
            if loc > 180:
                tmp_20 = value_per_deal.iloc[:, loc + 1 - 20:loc + 1].apply(_cal_M_reverse, axis=1, args=(pct,))
                tmp_60 = value_per_deal.iloc[:, loc + 1 - 60:loc + 1].apply(_cal_M_reverse, axis=1, args=(pct,))
                tmp_180 = value_per_deal.iloc[:, loc + 1 - 180:loc + 1].apply(_cal_M_reverse, axis=1, args=(pct,))

                reverse_20 = pd.concat([reverse_20, pd.DataFrame({m: tmp_20})], axis=1)
                reverse_60 = pd.concat([reverse_60, pd.DataFrame({m: tmp_60})], axis=1)
                reverse_180 = pd.concat([reverse_180, pd.DataFrame({m: tmp_180})], axis=1)

        reverse_20 = CALFUNC.del_dat_early_than(reverse_20, START_YEAR)
        reverse_60 = CALFUNC.del_dat_early_than(reverse_60, START_YEAR)
        reverse_180 = CALFUNC.del_dat_early_than(reverse_180, START_YEAR)

        res_dict = {"M_reverse_20": reverse_20,
                    "M_reverse_60": reverse_60,
                    "M_reverse_180": reverse_180,
                    }

        return res_dict

    @lazyproperty
    # # 估值因子
    def ep(self):
        pe_daily = self.pe_daily
        pe = CALFUNC.d_freq_to_m_freq(pe_daily)
        ep = 1/pe
        res = CALFUNC.del_dat_early_than(ep, START_YEAR)

        return res

    @lazyproperty
    def bp(self):
        pb_daily = self.pb_daily
        pb = CALFUNC.d_freq_to_m_freq(pb_daily)
        bp = 1 / pb
        res = CALFUNC.del_dat_early_than(bp, START_YEAR)

        return res

    @lazyproperty
    def assetturnover_q(self):
        totalassets = self.totalassets
        revenue = self.operatingrevenue
        # 得到单季度 净利润
        sig_season_revenue = get_signal_season_value(revenue)
        # 得到季度平均总资产
        s_mean_totalassets = get_season_mean_value(totalassets)

        turnover_q = (sig_season_revenue / s_mean_totalassets) * 100
        turnover_q = adjust_months(turnover_q)
        turnover_q = append_df(turnover_q)
        turnover_q = CALFUNC.del_dat_early_than(turnover_q, START_YEAR)

        return turnover_q

    @lazyproperty
    def totalassetturnover(self):

        totalassettrate = self.totalassettrate
        tmp0 = adjust_months(totalassettrate)
        tmp1 = append_df(tmp0)
        res = CALFUNC.del_dat_early_than(tmp1, START_YEAR)

        return res

    @lazyproperty
    # 单季度毛利率
    def grossprofitmargin_q(self):
        '''
        计算公示：（营业收入 - 营业成本） / 营业收入 * 100 %
        计算单季度指标，应该先对 营业收入 和 营业成本 分别计算单季度指标，再计算
        '''
        revenue = self.operatingrevenue    # 营业收入
        cost = self.operatingcost       # 营业成本
        # 财务指标常规处理，移动月份，改月末日期
        revenue_q = get_signal_season_value(revenue)
        cost_q = get_signal_season_value(cost)
        gross_q = (revenue_q - cost_q) / revenue_q
        # 调整为公告日期
        tmp = adjust_months(gross_q)
        # 用来扩展月度数据
        tmp = append_df(tmp)
        res = CALFUNC.del_dat_early_than(tmp, START_YEAR)
        return res

    @lazyproperty
    # 毛利率ttm
    def grossprofitmargin_ttm(self):

        gir = self.grossincomeratiottm
        gir = adjust_months(gir)
        # 用来扩展月度数据
        gir = append_df(gir)
        res = CALFUNC.del_dat_early_than(gir, START_YEAR)
        return res

    @lazyproperty
    def peg(self):
        # PEG = PE / 过去12个月的EPS增长率
        pe_daily = self.pe_daily
        basicepsyoy = self.basicepsyoy
        basicepsyoy = adjust_months(basicepsyoy)
        epsyoy = append_df(basicepsyoy, target_feq='D', fill_type='preceding')

        pe_daily = CALFUNC.del_dat_early_than(pe_daily, START_YEAR)
        epsyoy = CALFUNC.del_dat_early_than(epsyoy, START_YEAR)

        [pe_daily, epsyoy] = align(pe_daily, epsyoy)

        [h, l] = pe_daily.shape
        pe_ar = pe_daily.values
        eps_ar = epsyoy.values

        res = np.zeros([h, l])
        for i in range(0, h):
            for j in range(0, l):
                if pd.isna(eps_ar[i, j]) or eps_ar[i, j] == 0:
                    res[i, j] = np.nan
                else:
                    res[i, j] = pe_ar[i, j] / eps_ar[i, j]

        res_df = pd.DataFrame(data=res, index=pe_daily.index, columns=pe_daily.columns)

        return res_df

    @lazyproperty
    # 毛利率季度改善
    def grossprofitmargin_diff(self):
        revenue = self.operatingrevenue  # 营业收入
        cost = self.operatingcost  # 营业成本
        # 财务指标常规处理，移动月份，改月末日期
        revenue_q = get_signal_season_value(revenue)
        cost_q = get_signal_season_value(cost)
        gross_q = (revenue_q - cost_q) / revenue_q

        gir_d = CALFUNC.generate_diff(gross_q)
        gir_d = adjust_months(gir_d)
        # 用来扩展月度数据
        gir_d = append_df(gir_d)
        res = CALFUNC.del_dat_early_than(gir_d, START_YEAR)
        return res

    # Mom
    @lazyproperty
    def return_n_m(self):
        close = self.closeprice_daily
        adj = self.adjfactor

        close, adj = align(close, adj)
        c_p = close*adj
        c_v = c_p.values
        hh, ll =c_v.shape

        # 1个月、3个月、6个月、12个月
        m1 = np.zeros(c_v.shape)
        m3 = np.zeros(c_v.shape)
        m6 = np.zeros(c_v.shape)
        m12 = np.zeros(c_v.shape)
        for i in range(21, ll):
            m1[:, i] = c_v[:, i]/c_v[:, i-21]
        for i in range(21*3, ll):
            m3[:, i] = c_v[:, i]/c_v[:, i-21*3]
        for i in range(21*6, ll):
            m6[:, i] = c_v[:, i]/c_v[:, i-21*6]
        for i in range(21*12, ll):
            m12[:, i] = c_v[:, i]/c_v[:, i-21*12]

        m1_df = pd.DataFrame(data=m1, index=c_p.index, columns=c_p.columns)
        m3_df = pd.DataFrame(data=m3, index=c_p.index, columns=c_p.columns)
        m6_df = pd.DataFrame(data=m6, index=c_p.index, columns=c_p.columns)
        m12_df = pd.DataFrame(data=m12, index=c_p.index, columns=c_p.columns)

        m1_df_m = CALFUNC.d_freq_to_m_freq(m1_df)
        m3_df_m = CALFUNC.d_freq_to_m_freq(m3_df)
        m6_df_m = CALFUNC.d_freq_to_m_freq(m6_df)
        m12_df_m = CALFUNC.d_freq_to_m_freq(m12_df)

        m1_df_m1 = CALFUNC.del_dat_early_than(m1_df_m, START_YEAR)
        m3_df_m1 = CALFUNC.del_dat_early_than(m3_df_m, START_YEAR)
        m6_df_m1 = CALFUNC.del_dat_early_than(m6_df_m, START_YEAR)
        m12_df_m1 = CALFUNC.del_dat_early_than(m12_df_m, START_YEAR)

        res_dict = {'RETURN_1M': m1_df_m1 - 1,
                    'RETURN_3M': m3_df_m1 - 1,
                    'RETURN_6M': m6_df_m1 - 1,
                    'RETURN_12M': m12_df_m1 - 1,
                    }

        return res_dict

    @lazyproperty
    # 盈余动量
    def SUE(self):
        # 使用原始的财务数据
        eps = self.basiceps
        # 得到单季度的数据。
        sig_season_va = get_signal_season_value(eps)
        cols = pd.DataFrame([i for i in sig_season_va.columns])

        sue = pd.DataFrame()
        rolling_cols = rolling_windows(cols, 6)
        for roll in rolling_cols:
            res = _calculate_su_simple(sig_season_va[roll])
            res = pd.DataFrame(res.values, index=res.index, columns=[roll[-1]])
            sue = pd.concat([sue, res], axis=1)

        sue.dropna(how='all', axis=0, inplace=True)

        sue = adjust_months(sue)
        sue = append_df(sue)

        return sue

    @lazyproperty
    # 营收动量
    def REVSU(self):
        netprofit = self.totaloperatingrevenueps
        # 得到单季度的数据。
        sig_season_va = get_signal_season_value(netprofit)
        cols = pd.DataFrame([i for i in sig_season_va.columns])

        revsu = pd.DataFrame()
        rolling_cols = rolling_windows(cols, 6)
        for roll in rolling_cols:
            res = _calculate_su_simple(sig_season_va[roll])
            res = pd.DataFrame(res.values, index=res.index, columns=[roll[-1]])
            revsu = pd.concat([revsu, res], axis=1)

        revsu.dropna(how='all', axis=0, inplace=True)

        revsu = adjust_months(revsu)
        revsu = append_df(revsu)

        return revsu

    # 盈利
    @lazyproperty
    def ROA_ttm(self):
        roa_ttm = self.roattm
        roa_ttm = adjust_months(roa_ttm)
        roa_ttm = append_df(roa_ttm)
        roa_ttm = CALFUNC.del_dat_early_than(roa_ttm, START_YEAR)
        return roa_ttm

    # todo
    @lazyproperty
    def ROA_q(self):
        totalassets = self.totalassets
        netprofit = self.netprofit
        # 得到单季度 净利润
        sig_season_netprofit = get_signal_season_value(netprofit)
        # 得到季度平均总资产
        s_mean_totalassets = get_season_mean_value(totalassets)

        roa_q = (sig_season_netprofit/s_mean_totalassets) * 100
        roa_q = adjust_months(roa_q)
        roa_q = append_df(roa_q)
        roa_q = CALFUNC.del_dat_early_than(roa_q, START_YEAR)
        return roa_q

    @lazyproperty
    def ROE_q(self):
        totalshareholderequity = self.totalshareholderequity
        netprofit = self.netprofit
        # 得到单季度 净利润
        sig_season_netprofit = get_signal_season_value(netprofit)
        # 得到季度平均总资产
        s_mean_equity = get_season_mean_value(totalshareholderequity)

        roe_q = (sig_season_netprofit / s_mean_equity) * 100
        roe_q = adjust_months(roe_q)
        roe_q = append_df(roe_q)
        roe_q = CALFUNC.del_dat_early_than(roe_q, START_YEAR)
        return roe_q

    @lazyproperty
    def Profitmargin_q(self):     # 单季度净利润率
        '''
        1.qfa_deductedprofit：单季度.扣除非经常损益后的净利润
        2.qfa_oper_rev： 单季度.营业收入
        :return:
        '''

        netprofit = self.netprofitcut               # 扣除非经常损益后的净利润
        operatingrevenue = self.operatingrevenue
        sig_season_netprofit = get_signal_season_value(netprofit)
        sig_season_operatingrevenue = get_signal_season_value(operatingrevenue)
        profitmargin_q = sig_season_netprofit/sig_season_operatingrevenue
        profitmargin_q = adjust_months(profitmargin_q)
        profitmargin_q = append_df(profitmargin_q)

        pq = CALFUNC.del_dat_early_than(profitmargin_q, START_YEAR)

        return pq

    # 成长
    @lazyproperty
    def Profit_G_q(self):     # qfa_yoyprofit：单季度.净利润同比增长率
        netprofit = self.netprofitcut               # 扣除非经常损益后的净利润
        sig_season_netprofit = get_signal_season_value(netprofit)
        p_g = CALFUNC.generate_yoygr(sig_season_netprofit)
        p_g = adjust_months(p_g)
        p_g = append_df(p_g)
        profit_g_q = CALFUNC.del_dat_early_than(p_g, START_YEAR)
        return profit_g_q

    @lazyproperty
    def ROE_G_q(self):        # 单季度.ROE同比增长率
        roe = self.roe
        sig_season_roe = get_signal_season_value(roe)
        roe_g = CALFUNC.generate_yoygr(sig_season_roe)
        roe_g = adjust_months(roe_g)
        roe_g = append_df(roe_g)
        roe_g_q = CALFUNC.del_dat_early_than(roe_g, START_YEAR)
        return roe_g_q

    @lazyproperty
    def Sales_G_q(self):      # qfa_yoysales：单季度.营业收入同比增长率
        operatingrevenue = self.operatingrevenue
        sig_season_operatingrevenue = get_signal_season_value(operatingrevenue)
        sales_g = CALFUNC.generate_yoygr(sig_season_operatingrevenue)
        sales_g = adjust_months(sales_g)
        sales_g = append_df(sales_g)
        sales_g = CALFUNC.del_dat_early_than(sales_g, START_YEAR)
        return sales_g

    @lazyproperty
    def Rps(self):
        data = Data()

        all_codes = data.stock_basic_inform
        all_codes = pd.to_datetime(all_codes['ipo_date'.upper()])

        close_daily = data.closeprice_daily
        adjfactor = data.adjfactor
        close_price = close_daily*adjfactor
        close_price.dropna(axis=1, how='all', inplace=True)

        # 剔除上市一年以内的情况，把上市二年以内的股票数据都设为nan
        for i, row in close_price.iterrows():
            if i not in all_codes.index:
                row[:] = np.nan
                continue

            d = all_codes[i]
            row[row.index[row.index < d + timedelta(200)]] = np.nan

        if self._status == 'all':
            ext_120 = close_price/close_price.shift(periods=120, axis=1)
            ext_120.dropna(how='all', axis=1, inplace=True)
            rps_120 = ext_120.apply(scaler, scaler_max=100, scaler_min=1)

            rps = rps_120
            rps.dropna(how='all', axis=1, inplace=True)
            res = rps.apply(scaler, scaler_max=100, scaler_min=1)

            res = CALFUNC.del_dat_early_than(res, START_YEAR)
        elif self._status == 'update':
            hased_rps = data.RPS
            to_update = [col for col in close_price.columns if col not in hased_rps.columns and col > hased_rps.columns[-1]]
            if len(to_update) == 0:
                print('RPS无需要更新的部分')
                return hased_rps

            st = to_update[0]
            st_loc = np.where(close_price.columns == st)[0][0]
            st_loc = st_loc - 121

            close_price_new = close_price.iloc[:, st_loc:]
            ext_120 = close_price_new / close_price_new.shift(periods=120, axis=1)
            ext_120.dropna(how='all', axis=1, inplace=True)
            rps_120 = ext_120.apply(scaler, scaler_max=100, scaler_min=1)
            rps_120.dropna(how='all', axis=1, inplace=True)
            res0 = rps_120.apply(scaler, scaler_max=100, scaler_min=1)

            hased_rps[res0.columns] = res0
            res = hased_rps

        return res

    # 研发支出占营业收入的比例，因研发支出数据是在2018年3季度以后才开始披露的，所以该数据是在2018年3季度以后才有
    @lazyproperty
    def RDtosales(self):
        data = Data()

        rd_exp = data.rd_exp
        revenue = data.operatingrevenue
        rd_exp = CALFUNC.del_dat_early_than(rd_exp, 2018)
        revenue = CALFUNC.del_dat_early_than(revenue, 2018)

        res = rd_exp/revenue
        res = adjust_months(res)
        res = append_df(res)

        to_del = res.columns[res.isna().sum() / len(res) > 0.9]
        res.drop(to_del, axis=1, inplace=True)

        return res

    @lazyproperty
    def Rps_by_industry(self):
        data = Data()
        rps = data.RPS
        industry = data.stock_basic_inform
        industry = industry['申万一级行业']
        industry.dropna(inplace=True)

        t_del = [i for i in industry.index if i not in rps.index]
        industry = industry.drop(t_del)
        t_del = [i for i in rps.index if i not in industry.index]
        rps = rps.drop(t_del, axis=0)

        if self._status == 'all':
            new_mes = list(rps.columns)
            res = pd.DataFrame()
        elif self._status == 'update':
            hased_rps_ind = data.RPS_BY_INDUSTRY
            new_mes = [col for col in rps.columns if col not in hased_rps_ind.columns]
            res = pd.DataFrame()

        for col in new_mes:
            rps_tmp = rps[col]
            tmp_pd = pd.DataFrame({'rps': rps_tmp, 'industry': industry})
            grouped = tmp_pd.groupby('industry')

            rps_section = pd.DataFrame()
            for i, v_df in grouped:
                se = scaler(v_df['rps'], 100, 1)
                dat = pd.DataFrame({col: se})
                rps_section = pd.concat([rps_section, dat], axis=0)

            res = pd.concat([res, rps_section], axis=1)

        if self._status == 'update':
            res = pd.concat([hased_rps_ind, res], axis=1)

        return res

    # # 小郭写的---------------------------------------------------------------------------
    # @lazyproperty
    # def holder_avgpctchange(self):
    #     holder = self.holder_num
    #     holder_avgpct = 1 / holder*1000
    #     holder_shift12 = holder.shift(12, axis=1)
    #     holder_shift12pct = 1 / holder_shift12*1000
    #     holder_avgpctchange = holder_avgpct/holder_shift12pct-1
    #     return holder_avgpctchange
    #
    # @lazyproperty
    # def MACD(self):
    #     close = self.closeprice_daily
    #
    #     def EMA(arr, period=21):
    #         df = pd.DataFrame(arr)
    #         df = df.T
    #         return (df.ewm(span=period, min_periods=period).mean()).T
    #
    #     def myMACD(close, fastperiod=10, slowperiod=30, signalperiod=15):
    #         ewma10 = EMA(close, fastperiod)
    #         ewma30 = EMA(close, slowperiod)
    #         dif = ewma10 - ewma30
    #         dea = EMA(dif, signalperiod)
    #         bar = (dif - dea) * 2  # 有些地方的bar = (dif-dea)*2，但是talib中MACD的计算是bar = (dif-dea)*1
    #         return dif, dea, bar
    #
    #     diff, dea, macd = myMACD(close)
    #     diff = CALFUNC.del_dat_early_than(diff, START_YEAR)
    #     dea = CALFUNC.del_dat_early_than(dea, START_YEAR)
    #     macd = CALFUNC.del_dat_early_than(macd, START_YEAR)
    #
    #     res_dict = {"DIF": diff, "DEA": dea, "MACD": macd}
    #
    #     return res_dict
    #
    # @lazyproperty
    # def RSI(self):
    #     '''
    #     RSI（Relative Strength Index），中文名称：相对强弱指标。在一段时间内，上涨幅度代表多方力量，下跌幅度代表空方力量，两种力量的对比决定了个股及大盘所处的状态：强势或弱势。周期取20日
    #     RSI = N日内上涨幅度累计 / N日内上涨及下跌幅度累计 * 100
    #     "RSI = SMA(MAX(CLOSE-LAST_CLOSE,0),N,1)/SMA(ABS(CLOSE-LAST_CLOSE),N,1)*100"
    #     '''
    #     import copy
    #     pct = self.changepct_daily.T
    #     pct.index = pd.to_datetime(pct.index)
    #     pct = pct/100
    #     pct_nonnegative = copy.deepcopy(pct)
    #     pct_nonnegative[pct_nonnegative < 0] = 0
    #     numerator = pct_nonnegative.rolling(20).sum()
    #     pct_positive = abs(pct)
    #     denominator = pct_positive.rolling(20).sum()
    #     RSI = (numerator/denominator)*100
    #     RSI = RSI.T
    #     RSI = CALFUNC.del_dat_early_than(RSI, START_YEAR)
    #     return RSI
    #
    # @lazyproperty
    # def PSY(self):
    #     '''
    #     PSY（Psychology Line），中文名称：心理线。它是研究某段时间的投资人趋向于买方或卖方的心理与事实，是考察市场中群体心理变化的依据。周期取20日
    #     PSY = N周期内上涨的周期数 / N * 100
    #     PSY = COUNT(CLOSE>LAST_CLOSE,N)/N*100
    #     '''
    #     import copy
    #     pct = self.changepct_daily.T
    #     pct.index = pd.to_datetime(pct.index)
    #     pct = pct / 100
    #     pct_positive = copy.deepcopy(pct)
    #     pct_positive[pct_positive != np.nan] = 0
    #     pct_positive[pct>0] = 1
    #     numerator = pct_positive.rolling(20).sum()
    #     PSY = (numerator / 20) * 100
    #     PSY = PSY.T
    #     PSY = CALFUNC.del_dat_early_than(PSY, START_YEAR)
    #     return PSY
    #
    # @lazyproperty
    # def BIAS(self):
    #     '''
    #     BIAS（乖离率）是用股价指数与移动平均线的比值关系，来描述股票价格与移动平均线之间的偏离程度。周期取20日
    #     BIAS(N) = （收盘价 - N周期移动平均价） / N周期移动平均价 * 100
    #     BIAS = (CLOSE-MA(CLOSE,N))/MA(CLOSE,N)*100
    #     '''
    #     import copy
    #     close = self.closeprice_daily.T
    #     close.index = pd.to_datetime(close.index)
    #     MA20 = close.rolling(20, min_periods=1).mean()
    #     numerator = close-MA20
    #     BIAS = numerator/MA20*100
    #     BIAS = BIAS.T
    #     BIAS = CALFUNC.del_dat_early_than(BIAS, START_YEAR)
    #     return BIAS
    #
    # @lazyproperty
    # def DASTD(self):
    #     '''
    #     日标准差
    #     STD={∑[wi（Xi-X）^2]}^(0.5),Xi为日收益率序列，X为该月每日平均涨跌幅，wi为半衰期权重
    #     日超额收益过去252个交易日的波动率，半衰期42天
    #     '''
    #     dastd = self._rolling(self.changepct / 100, window=252, half_life=42, func_name='std')
    #     dastd = CALFUNC.del_dat_early_than(dastd, START_YEAR)
    #     return dastd
    #
    # @lazyproperty
    # def CMRA(self):
    #     stock_ret = self._fillna(self.changepct / 100, 0)
    #     ret = np.log(1 + stock_ret)
    #     cmra = self._pandas_parallelcal(ret, self._cal_cmra, args=(12, 21), window=252, axis=0).T
    #     cmra = CALFUNC.del_dat_early_than(cmra, START_YEAR)
    #     return cmra
    #
    # @lazyproperty
    # def HALPHA_BARRA(self):
    #     if 'HALPHA_BARRA' in self.__dict__:
    #         return self.__dict__['HALPHA_BARRA']
    #     else:
    #         beta, alpha, hsigma = self._capm_regress(window=252, half_life=63)
    #         self.__dict__['BETA_BARRA'] = beta
    #         self.__dict__['HSIGMA_BARRA'] = hsigma
    #         return alpha
    #
    # @lazyproperty
    # def RSTR(self, version=6):
    #     benchmark_ret = self._get_benchmark_ret()
    #     stock_ret = self.changepct / 100
    #     benchmark_ret, stock_ret = self._align(benchmark_ret, stock_ret)
    #     benchmark_ret = benchmark_ret['000300.SH']
    #
    #     excess_ret = np.log((1 + stock_ret).divide((1 + benchmark_ret), axis=0))
    #     if version == 6:
    #         rstr = self._rolling(excess_ret, window=252, half_life=126, func_name='sum')
    #         rstr = rstr.rolling(window=11, min_periods=1).mean()
    #     elif version == 5:
    #         exp_wt = self._get_exp_weight(504 + 21, 126)[:504]
    #         rstr = self._rolling(excess_ret.shift(21), window=504, weights=exp_wt,
    #                              func_name='sum')
    #     rstr = CALFUNC.del_dat_early_than(rstr, START_YEAR)
    #     return rstr
    #
    # @lazyproperty
    # def MLEV(self, version=6):
    #     '''
    #     市场杠杆
    #     MLEV=(ME+PE+LD)/ME
    #     其中ME为上一财年普通股的市场价值，PE和LD分别是上一财年的优先股和长期负债
    #     '''
    #
    #
    #
    # @lazyproperty
    # def BLEV(self):
    #     longdebttoequity = self.longdebttoequity
    #     be = self.totalshareholderequity
    #     ld = be * longdebttoequity
    #     pe = self.pe_daily
    #     be, pe, ld = self._align(be, pe, ld)
    #     blev = (be + pe + ld) / be
    #     blev = blev.T
    #     blev = adjust_months(blev)
    #     blev = append_df(blev)
    #     return blev
    #
    # @lazyproperty
    # def DTOA(self):
    #     tl = self.totalassets  # 季频
    #     ta = self.totalliability  # 季频
    #     dtoa = tl / ta
    #     dtoa = dtoa.shift(4, axis=1)
    #     dtoa = adjust_months(dtoa)
    #     dtoa = append_df(dtoa)
    #     dtoa = CALFUNC.del_dat_early_than(dtoa, START_YEAR)
    #     return dtoa
    #
    # # 5 ** dtoa = td / ta; td -- long-term debt+current liabilities;td,ta ---- mrq
    #
    # @lazyproperty
    # def BTOP(self):
    #     '''
    #     账面市值比
    #     BP = sewithoutmi / total_mv, 即归属母公司股东的权益 / 总市值
    #     最近报告期的普通股账面价值除以当前市值 Net Asset PS每股净资产
    #     股票账面价值又称股票净值或每股净资产，是每股股票所代表的实际资产的价值。每股账面价值是以公司净资产减去优先股账面价值后，除以发行在外的普通股票的股数求得的。
    #     '''
    #     bv = self.netassetps  # 季频
    #     bv = adjust_months(bv)
    #     bv = append_df(bv)
    #     mkv = self.totalmv_daily  # 日频
    #     btop = bv / mkv
    #     btop = CALFUNC.del_dat_early_than(btop, START_YEAR)
    #     return btop
    #
    # @lazyproperty
    # def ETOP(self):
    #     '''
    #     盈利市值比
    #     EP = netprofit_ttm / total_mv
    #     过去12个月的盈利(TTM)除以当前市值
    #     '''
    #     e_ttm = self.netprofit_ttm
    #     mkv = self.totalmv_daily
    #     e = e_ttm.fillna(value=0)
    #     mk = mkv.fillna(np.inf)
    #     etop = e / mk
    #     etop = CALFUNC.del_dat_early_than(etop, START_YEAR)
    #     return etop
    #
    # @lazyproperty
    # def CETOP(self):
    #     operatecashflow = self.operatecashflow_ttm
    #     mkv = self.totalmv_daily
    #     o = operatecashflow.fillna(value=0)
    #     mk = mkv.fillna(np.inf)
    #     cetop = o / mk
    #     cetop = CALFUNC.del_dat_early_than(cetop, START_YEAR)
    #     return cetop
    #
    # @lazyproperty
    # def EGRO(self):
    #     import pandas as pd
    #     import statsmodels.api as sm
    #     eps = self.basiceps
    #     # 删除非12月份的数据
    #     eps.columns = pd.to_datetime(eps.columns)
    #     tdc = [col for col in eps.columns if col.month != 12]
    #     eps = eps.drop(tdc, axis=1)
    #     eps = eps.T
    #     x = pd.Series(range(1, 6))
    #     X = sm.add_constant(x)
    #     ut = pd.DataFrame()
    #
    #     def regress(y):
    #         y = pd.Series(y)
    #         X.set_index(y.index.values, drop=True, inplace=True)
    #         model = sm.OLS(y, X)
    #         results = model.fit()
    #         par = results.params[[1]]
    #         return par
    #
    #     for col in eps.columns:
    #         dependent = pd.Series(eps[col]).fillna(value=0)
    #         result_reg = dependent.rolling(5).apply(regress)
    #         result_mean = dependent.rolling(5).mean()
    #         ratio = result_reg / result_mean
    #         ut = pd.concat([ut, ratio], axis=1)
    #     ut = ut.T
    #     return ut
    #
    # # 5 --
    # @lazyproperty
    # def SGRO(self):
    #     import pandas as pd
    #     import statsmodels.api as sm
    #     ops = self.operatingrevenuepsttm
    #     # 删除非12月份的数据
    #     ops.columns = pd.to_datetime(ops.columns)
    #     tdc = [col for col in ops.columns if col.month != 12]
    #     ops = ops.drop(tdc, axis=1)
    #     ops = ops.T
    #     x = pd.Series(range(1, 6))
    #     X = sm.add_constant(x)
    #     ut = pd.DataFrame()
    #
    #     def regress(y):
    #         y = pd.Series(y)
    #         X.set_index(y.index.values, drop=True, inplace=True)
    #         model = sm.OLS(y, X)
    #         results = model.fit()
    #         par = results.params[[1]]
    #         return par
    #
    #     for col in ops.columns:
    #         dependent = pd.Series(ops[col]).fillna(value=0)
    #         result_reg = dependent.rolling(5).apply(regress)
    #         result_mean = dependent.rolling(5).mean()
    #         ratio = result_reg / result_mean
    #         ut = pd.concat([ut, ratio], axis=1)
    #     ut = ut.T
    #     return ut
    #
    # @lazyproperty
    # def turn_nm(self):
    #     # n分别为1、3、6、12，每个月为21个交易日
    #     # turnover_ratio	换手率	decimal(10,4)		单位：％
    #     turnover = self.turnoverrate_daily / 100
    #     new_mes = [m for m in self._mes if m in turnover.columns]
    #     turn_1m = pd.DataFrame()
    #     turn_3m = pd.DataFrame()
    #     turn_6m = pd.DataFrame()
    #     turn_12m = pd.DataFrame()
    #     for m in new_mes:
    #         loc = np.where(turnover.columns == m)[0][0]
    #         if loc > 12 * 21:
    #             # t = turnover.iloc[:, loc + 1 - 21:loc + 1]
    #             res_df1 = turnover.iloc[:, loc + 1 - 21:loc + 1].mean(axis=1)
    #             res_df3 = turnover.iloc[:, loc + 1 - 3 * 21:loc + 1].mean(axis=1)
    #             res_df6 = turnover.iloc[:, loc + 1 - 6 * 21:loc + 1].mean(axis=1)
    #             res_df12 = turnover.iloc[:, loc + 1 - 12 * 21:loc + 1].mean(axis=1)
    #
    #             turn_1m = pd.concat([turn_1m, pd.DataFrame({m: res_df1})], axis=1)
    #             turn_3m = pd.concat([turn_3m, pd.DataFrame({m: res_df3})], axis=1)
    #             turn_6m = pd.concat([turn_6m, pd.DataFrame({m: res_df6})], axis=1)
    #             turn_12m = pd.concat([turn_12m, pd.DataFrame({m: res_df12})], axis=1)
    #
    #     turn_1m = CALFUNC.del_dat_early_than(turn_1m, START_YEAR)
    #     turn_3m = CALFUNC.del_dat_early_than(turn_3m, START_YEAR)
    #     turn_6m = CALFUNC.del_dat_early_than(turn_6m, START_YEAR)
    #     turn_12m = CALFUNC.del_dat_early_than(turn_12m, START_YEAR)
    #
    #     res_dict = {"Turn_1m": turn_1m,
    #                 "Turn_3m": turn_3m,
    #                 "Turn_6m": turn_6m,
    #                 "Turn_12m": turn_12m,
    #                 }
    #
    #     return res_dict
    #
    # @lazyproperty
    # def bias_turn_nm(self):
    #     # n分别为1、3、6、12，每个月为21个交易日
    #     # turnover_ratio	换手率	decimal(10,4)		单位：％
    #     turnover = self.turnoverrate_daily / 100
    #     new_mes = [m for m in self._mes if m in turnover.columns]
    #     turn_1m = pd.DataFrame()
    #     turn_3m = pd.DataFrame()
    #     turn_6m = pd.DataFrame()
    #     turn_12m = pd.DataFrame()
    #     turn_24m = pd.DataFrame()
    #     for m in new_mes:
    #         loc = np.where(turnover.columns == m)[0][0]
    #         if loc > 12 * 21:
    #             # t = turnover.iloc[:, loc + 1 - 21:loc + 1]
    #             res_df1 = turnover.iloc[:, loc + 1 - 21:loc + 1].mean(axis=1)
    #             res_df3 = turnover.iloc[:, loc + 1 - 3 * 21:loc + 1].mean(axis=1)
    #             res_df6 = turnover.iloc[:, loc + 1 - 6 * 21:loc + 1].mean(axis=1)
    #             res_df12 = turnover.iloc[:, loc + 1 - 12 * 21:loc + 1].mean(axis=1)
    #             res_df24 = turnover.iloc[:, loc + 1 - 24 * 21:loc + 1].mean(axis=1)
    #
    #             turn_1m = pd.concat([turn_1m, pd.DataFrame({m: res_df1})], axis=1)
    #             turn_3m = pd.concat([turn_3m, pd.DataFrame({m: res_df3})], axis=1)
    #             turn_6m = pd.concat([turn_6m, pd.DataFrame({m: res_df6})], axis=1)
    #             turn_12m = pd.concat([turn_12m, pd.DataFrame({m: res_df12})], axis=1)
    #             turn_24m = pd.concat([turn_24m, pd.DataFrame({m: res_df24})], axis=1)
    #
    #     turn_1m = CALFUNC.del_dat_early_than(turn_1m, START_YEAR)
    #     turn_3m = CALFUNC.del_dat_early_than(turn_3m, START_YEAR)
    #     turn_6m = CALFUNC.del_dat_early_than(turn_6m, START_YEAR)
    #     turn_12m = CALFUNC.del_dat_early_than(turn_12m, START_YEAR)
    #     turn_24m = CALFUNC.del_dat_early_than(turn_12m, START_YEAR)
    #
    #     bias_turn_1m = turn_1m / turn_24m - 1
    #     bias_turn_3m = turn_3m / turn_24m - 1
    #     bias_turn_6m = turn_6m / turn_24m - 1
    #     bias_turn_12m = turn_12m / turn_24m - 1
    #
    #     res_dict = {"bias_turn_1m": bias_turn_1m,
    #                 "bias_turn_3m": bias_turn_3m,
    #                 "bias_turn_6m": bias_turn_6m,
    #                 "bias_turn_12m": bias_turn_12m,
    #                 }
    #
    #     return res_dict


def compute_factor(status):
    # 动量类因子
    fc = Factor_Compute(status)

    factor_names = [k for k in Factor_Compute.__dict__.keys() if not k.startswith('_')]
    for f in factor_names:
        print(f)
        if f == 'compute_pct_chg_nm':
            res = fc.compute_pct_chg_nm
            if isinstance(res, pd.DataFrame):
                fc.save(res, 'pct_chg_nm'.upper())
        else:
            try:
                tmp = eval('fc.' + f)
                if isinstance(tmp, pd.DataFrame):            # 返回None，表示无需更新
                    fc.save(tmp, f.upper())
                elif isinstance(tmp, dict):
                    for k, v in tmp.items():
                        fc.save(v, k.upper())
                elif not tmp:
                    continue
            except Exception as e:
                print('debug')


if __name__ == "__main__":
    # compute_factor('update')

    # 'all'  'update'
    # 测试某个因子
    fc = Factor_Compute('update')  # ('update')
    res = fc.beta
    fc.save(res, 'beta'.upper())
    panel_path = os.path.join(root_dair, '因子预处理模块', '因子')
    add_to_panels(res, panel_path, 'Beta', freq_in_dat='M')

    # reverse_20 = res["M_reverse_20"]
    # reverse_60 = res["M_reverse_60"]
    # reverse_180 = res["M_reverse_180"]
    #
    # fc.save(reverse_20, 'M_reverse_20'.upper())
    # fc.save(reverse_60, 'M_reverse_60'.upper())
    # fc.save(reverse_180, 'M_reverse_180'.upper())
    #
    # panel_path = os.path.join(root_dair, '因子预处理模块', '因子')
    # add_to_panels(reverse_20, panel_path, 'M_reverse_20', freq_in_dat='M')
    # add_to_panels(reverse_60, panel_path, 'M_reverse_60', freq_in_dat='M')
    # add_to_panels(reverse_180, panel_path, 'M_reverse_180', freq_in_dat='M')

    # res = fc.Rps
    # fc.save(res, 'Rps'.upper())
    # res = fc.Rps_by_industry
    # fc.save(res, 'Rps_by_industry'.upper())
    # panel_path = os.path.join(root_dair, '因子预处理模块', '因子')
    # add_to_panels(res, panel_path, 'Rps_by_industry', freq_in_dat='M')
    # grossprofitmargin_q
    # grossprofitmargin_q_diff


