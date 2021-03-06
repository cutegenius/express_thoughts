#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import statsmodels.api as sm
from datetime import datetime
import shutil
from functools import reduce
from sklearn.covariance import LedoitWolf
from utility.factor_data_preprocess import add_to_panels, align
from utility.tool0 import Data, add_stock_pool_txt
from utility.tool3 import adjust_months, append_df, wr_excel
from utility.relate_to_tushare import stocks_basis, generate_months_ends

from utility.stock_pool import financial_condition_pool, factor_condition_pool, concat_stock_pool, save_each_sec_name,\
    del_industry, keep_industry, get_scores, twice_sort, del_market, keep_market, from_stock_wei_2_industry_wei,\
    compute_icir
from utility.download_from_wind import section_stock_infor
from utility.analysis import BackTest, bool_2_ones
from utility.stock_pool import get_scores_with_wei, get_scores, month_return_compare_to_market_index
from utility.select_industry import my_factor_concat, history_factor_return,  forecast_factor_return, \
    copy_matrix, forecast_factor_return
from utility.index_enhance import linear_programming, concat_factors_panel, get_factor, get_est_stock_return
from utility.optimization import optimization_fun
from utility.constant import data_dair, root_dair, default_dict, factor_dict_to_concate  #,factor_dict_for_scores,
from utility.single_factor_test import get_datdf_in_panel, panel_to_matrix


class FormStockPool:

    def __init__(self, abs_financial_dict=None, opt_para_dict=None, factor_concate_dict=None,
                 factor_dict_for_scores=None, path_dict=None, use_risk_model=False,
                 alpha_factor=None, risk_factor=None, control_factor=None, select_mode='total_num',
                 twice_sort_dict={}, percent_n=0.1, freq='M', industry_dict=None, special_market_dict=None,
                 benchmark='ZZ500', method='regress', score_mode='all_same', update_ornot='renew',
                 start_date=None, end_date=None
                 ):
        """
        :param abs_financial_dict:
        :param factor_wei:
        :param opt_para_dict:
        :param factor_concate_dict:
        :param factor_dict_for_scores:
        :param path_dict:
        :param use_risk_model:
        :param alpha_factor:
        :param risk_factor:
        :param select_mode:
        :param first_max:
        :param twice_sort_dict:
        :param max_n:
        :param freq:
        :param industry_dict:
        :param special_market_dict: 板块设定，如果仅保留创业板的股票或剔除创业板的股票
        :param benchmark:     基准设定
        :param method:
        """

        self._data = Data()
        # 回测开始日期
        if start_date:
            self.start_date = start_date
        else:
            self.start_date = datetime(2009, 3, 1)
        # 回测结束日期
        if end_date:
            self.end_date = end_date
        else:
            self.end_date = datetime.today()
        # 选股方法： 打分法还是回归法
        self.rov_method = method
        # 是否应用风险模型进行权重优化
        self.use_risk_model = use_risk_model

        # 财务绝对要求条件
        self.abs_financial_dict = abs_financial_dict
        # 使用回归法预测股票收益时的所有因子
        if factor_concate_dict:
            self.factor_concate_dict = factor_concate_dict
        else:
            self.factor_concate_dict = factor_dict_to_concate
        # 使用打分法预测股票收益排名时的所有因子
        self.factor_dict_for_scores = factor_dict_for_scores

        # 因子合成地址
        if not path_dict:
            self.path_dict = {'save_path': os.path.join(root_dair, '多因子选股'),
                              'factor_panel_path': os.path.join(root_dair, '因子预处理模块', '因子（已预处理）'),
                              'ic_path': os.path.join(root_dair, '单因子检验', 'ic.csv'),
                              'old_matrix_path': os.path.join(root_dair, '单因子检验', '因子矩阵')}
        else:
            self.path_dict = path_dict

        # alpha因子, self.factor_dict中的key且不再alpha的名称列表里的，是风险因子
        self.alpha_factor = alpha_factor
        if not alpha_factor:
            self.alpha_factor = ['quality', 'growth']
        else:
            self.alpha_factor = alpha_factor

        self.risk_factor = risk_factor
        self.control_factor = control_factor
        self.opt_para_dict = opt_para_dict

        self.factors_panel = None

        # 总股票数量
        self.top_percent_n = percent_n
        # 单个行业的股票数量
        self.one_industry_num = None
        # 因子加权方式为等权
        self.wei_type = 'equal'
        # 测试频率
        self.freq = freq
        # 行业设定
        self.industry_dict = industry_dict
        # 特定板块设定
        self.special_market_dict = special_market_dict
        # 历史的因子表现
        self.factor_ret_history = None
        # 股票的特质收益率
        self.resid = None
        # 计算压缩矩阵时的窗口期
        self.shrinkaged_window = 12
        # 使用打分法时，对不同行业的股票，是否使用同样的大类因子打分，
        # 如果使用同样的大类因子，则为'all_same'，否则为'each_industry'
        self.score_mode = score_mode
        self.icir_window = 6
        # 历史的icir
        self.icir_dict = {}
        # 预期的icir
        self.icir_e_dict = {}
        # 不同截面的组合协方差矩阵
        self.E_dict = None
        # 月度价格变化
        self.pet_chg_m = self._data.PCT_CHG_NM

        self.final_stock_pool = None

        self.index_wei = None
        self.constituent_industry = None
        self.benchmark = benchmark
        if self.benchmark:
            self.set_benchmark()

        self.pool = None
        stock_basic = self._data.stock_basic_inform
        self.industry_map = pd.DataFrame({'first_industry': stock_basic['申万一级行业']})

        # 设置为'total_num'，表示从所有股票中选择排名靠前的， 'by_industry'表示分行业选择, twice_sort表示两步排序法
        self.select_type = select_mode
        self.twice_sort_dict = twice_sort_dict
        self.update_or_renew = update_ornot

    # 设定基准的成分股权重
    def set_benchmark(self):
        if self.benchmark == 'allA':  # 'HS300' 'ZZ500' 'SZ50'
            self.constituent_industry = None
        elif self.benchmark == 'HS300':
            tmp = self._data.hs300_wt
            tmp = tmp / 100
            self.index_wei = tmp
            self.constituent_industry = from_stock_wei_2_industry_wei(tmp)
        elif self.benchmark == 'ZZ500':
            tmp = self._data.zz500_wt
            tmp = tmp / 100
            self.index_wei = tmp
            self.constituent_industry = from_stock_wei_2_industry_wei(tmp)

    def get_stock_pool_by_removing(self):

        # 财务绝对要求条件的股票池构建，如roettm大于5%，eps同比增速大于5，sue位于所有股票的top5%。
        stock_pool = financial_condition_pool(self.abs_financial_dict, self.start_date, self.end_date)

        # 特定行业的股票
        if self.industry_dict:
            if self.industry_dict['handle_type'] == 'delete':
                stock_pool = del_industry(stock_pool, self.industry_dict['to_handle_indus'])
            elif self.industry_dict['handle_type'] == 'keep':
                stock_pool = keep_industry(stock_pool, self.industry_dict['to_handle_indus'])
        # 特定股票板块
        if self.special_market_dict:
            for k, v in self.special_market_dict.items():
                if k == 'delete':
                    stock_pool = del_market(stock_pool, v)
                elif k == 'keep':
                    print('没需求，未实现')
                    stock_pool = keep_market(stock_pool, v)

        # 选择给定时间段的列
        cols = [col for col in stock_pool.columns if col >= self.start_date]
        stock_pool = stock_pool[cols]
        stock_pool.sum()
        self.pool = stock_pool

    # 合成新的大类因子。无论是打分法还是回归法，都需要进行该操作。
    def concate_factor(self):
        concate_dict = self.factor_concate_dict
        path_dict = self.path_dict

        # ----------------------------------
        # 因子合成
        if self.update_or_renew == 'renew':
            # 删除过去合成的结果
            if os.path.exists(os.path.join(path_dict['save_path'], '新合成因子')):
                shutil.rmtree(os.path.join(path_dict['save_path'], '新合成因子'))
        print('开始进行因子合成处理.....')

        # 先对所有股票均使用相同的规则合成大类因子
        factors_dict = concate_dict['default']
        my_factor_concat(path_dict, factors_dict, self.update_or_renew, concat_type='equal_weight',
                         start_date=self.start_date)

        # 再对特殊处理的行业按照特定规则合成大类因子
        for ind, factor_dict_tmp in concate_dict.items():
            if ind == 'default':
                continue

            if ind == '计算机':
                print('debug')

            # 针对特殊行业定义路径，使用前期已经处理完毕的行业内因子截面
            tmp_path_dict = {
                'save_path': os.path.join(root_dair, '分行业研究', ind, '多因子选股'),
                'factor_panel_path': os.path.join(root_dair, '分行业研究', ind, '因子（已预处理）'),
                             }

            if self.update_or_renew == 'renew':
                if os.path.exists(os.path.join(tmp_path_dict['save_path'], '新合成因子')):
                    shutil.rmtree(os.path.join(tmp_path_dict['save_path'], '新合成因子'))

            # 针对特殊行业进行因子合成
            my_factor_concat(tmp_path_dict, factor_dict_tmp, self.update_or_renew,
                             concat_type='equal_weight', start_date=self.start_date)

            print('{}因子合成完毕！'.format(ind))

    # 因子正交
    def factor_to_symmetry(self, factor_path, save_path, symmetry_dict):
        after_sym_path = os.path.join(save_path, '正交后因子')
        after_sym_panel_spath = os.path.join(after_sym_path, '因子截面')

        if not os.path.exists(after_sym_path):
            os.mkdir(after_sym_path)
        if not os.path.exists(after_sym_panel_spath):
            os.mkdir(after_sym_panel_spath)

        if self.update_or_renew == 'update':
            fd_list = [pd.Timestamp(datetime.strptime(m.split('.')[0], "%Y-%m-%d")) for m in
                       os.listdir(factor_path)]
            hased_list = [pd.Timestamp(datetime.strptime(m.split('.')[0], "%Y-%m-%d")) for m in
                          os.listdir(after_sym_panel_spath)]

            to_compute = [m for m in fd_list if m not in hased_list and m > hased_list[-1]]

            if len(to_compute) == 0:
                print(f"正交后因子数据无需要更新，退出.")
                return 0
            else:
                dirlist = [m.to_pydatetime().strftime("%Y-%m-%d") + '.csv' for m in to_compute]
        else:
            dirlist = factor_path

        for f in os.listdir(dirlist):
            # f = os.listdir(factors_path)[0]
            # 依次打开每个月度数据
            dat = pd.read_csv(os.path.join(dirlist, f), encoding='gbk',
                              engine='python', index_col=[0])
            for key, values in symmetry_dict.items():
                for v in values:
                    # 正交化处理 线性回归 y=kx  sm.OLS(y, x)
                    model = sm.OLS(dat[v], dat[key], hasconst=None)
                    result = model.fit()
                    tmp_se = pd.Series(index=dat.index, data=result.resid)
                    dat[v] = tmp_se
            dat.to_csv(os.path.join(after_sym_panel_spath, f), encoding='gbk')

    def get_factor_corr(self, factors, path):
        panel_dict = get_datdf_in_panel(path)

        corrs = []
        for date in sorted(panel_dict.keys()):
            factor_panel = panel_dict[date]
            df_tmp = factor_panel[factors]
            corrs.append(df_tmp.corr())

        avg_corr = reduce(lambda df1, df2: df1 + df2, corrs) / len(corrs)
        return avg_corr

    def compute_icir(self, panel_path=None, save_path=None):
        # default的icir值
        if not panel_path:
            panel_path = os.path.join(self.path_dict['save_path'], '正交后因子', '因子截面')
        if not save_path:
            save_path = os.path.join(self.path_dict['save_path'], '正交后因子')

        factors = list(self.factor_concate_dict['default'].keys())
        icir_default = compute_icir(self.icir_window, panel_path, save_path, factors,
                                    update_or_renew=self.update_or_renew)
        if icir_default.empty:
            print('icir计算有bug')
        self.icir_dict.update({'default': icir_default})

        # 再对特殊处理的行业按照特定规则合成大类因子
        for ind, factor_dict in self.factor_concate_dict.items():
            if ind == 'default':
                continue

            if ind == '计算机':
                print('debug')

            print('开始计算{}的icir值'.format(ind))
            panel_path = os.path.join(root_dair, '分行业研究', ind, '多因子选股', '新合成因子', '因子截面')
            save_path = os.path.join(root_dair, '分行业研究', ind, '多因子选股')
            factors = factor_dict.keys()
            icir = compute_icir(self.icir_window, panel_path, save_path, factors,
                                update_or_renew=self.update_or_renew)
            self.icir_dict.update({ind: icir})
        print('ICIR计算完毕')

        # 从历史icir到计算中使用的icir，直接沿着日期向下shift一期。
        self.icir_e_dict = {}
        for key, value in self.icir_dict.items():
            value_shifted = value.shift(1, axis=0)
            value_shifted = value_shifted.dropna(how='all')
            self.icir_e_dict.update({key: value_shifted})

    def compute_factor_return(self):

        factors_dict = self.factor_concate_dict
        path_dict = self.path_dict
        factors, window, half_life = [key for key in factors_dict.keys()], 6, None
        factors.extend(['Pct_chg_nm'])

        fn = list(self.factor_concate_dict['default'].keys())
        fn.append('Pct_chg_nm')
        fn.append('Lncap_barra')
        p = os.path.join(path_dict['save_path'], '正交后因子', '因子截面')
        factors_panel = get_datdf_in_panel(p, fn)
        self.factors_panel = factors_panel
        factor_rets = pd.DataFrame()
        resid = pd.DataFrame()

        if self.update_or_renew == 'renew':
            factor_rets = pd.DataFrame()
            resid = pd.DataFrame()
        elif self.update_or_renew == 'update':
            # 读取已经计算过的因子收益和残差值
            factor_rets = pd.read_csv(os.path.join(root_dair, '临时', 'factor_rets.csv'), encoding='gbk')
            factor_rets.set_index(factor_rets.columns[0], inplace=True)
            factor_rets.index = pd.to_datetime(factor_rets.index)
            resid = pd.read_csv(os.path.join(root_dair, '临时', 'resid.csv'), encoding='gbk')
            resid.set_index(resid.columns[0], inplace=True)
            resid.index = pd.to_datetime(resid.index)
            # 选择出需要计算的月份，并更改factors_panel变量
            to_compute = [i for i in factor_rets.index if i not in list(factors_panel.keys())]
            if len(to_compute) == 0:
                print('因子收益无需更新。')
                return factor_rets, resid
            else:
                to_del = [i for i in list(factors_panel.keys()) if i not in to_compute]
                for d in to_del:
                    factors_panel.pop(d)

        new_factor_ret_history, new_resid = history_factor_return(factors, factors_panel, window, half_life)

        factor_rets = pd.concat([factor_rets, new_factor_ret_history], axis=0)
        resid = pd.concat([resid, new_resid], axis=0)

        # 保存
        factor_rets.to_csv(os.path.join(root_dair, '临时', 'factor_rets.csv'), encoding='gbk')
        resid.to_csv(os.path.join(root_dair, '临时', 'resid.csv'), encoding='gbk')
        # 最后一行的nan先不能删除，因为后面预测时要向后shift一行
        return factor_rets, resid

    def get_shrinkaged(self):
        '''
        E = fX*F_shrinkaged*fX.T + e
        :return:
        '''

        shrinkaged_dict = {}
        self.shrinkaged_window = 12

        # 因子名称
        # fn = [k for k in self.factor_concate_dict['default'].keys()]
        fn = self.risk_factor

        for l in range(self.shrinkaged_window, len(self.factor_ret_history)):

            factor_ret_tmp = self.factor_ret_history.loc[self.factor_ret_history.index[l-self.shrinkaged_window:l], fn]
            try:
                cov = LedoitWolf().fit(factor_ret_tmp)
                factor_cov_tmp = cov.covariance_  # 通过压缩矩阵算法得到因子收益协方差矩阵
            except Exception as e:
                print('压缩矩阵1')

            # 该期的因子截面数据
            dat_tmp_df = self.factors_panel[self.factor_ret_history.index[l]]
            f_expo = dat_tmp_df.loc[:, fn].dropna(axis=0, how='any')

            # f_expo为N*M，N为股票数量，M为因子个数； factor_cov_tmp为M*M，f_expo.T 为M*N，最后结果为N*N
            factor_cov = np.dot(np.dot(f_expo.values, factor_cov_tmp), f_expo.values.T)
            factor_cov_df = pd.DataFrame(data=factor_cov, index=f_expo.index, columns=f_expo.index)

            resid_tmp = self.resid.loc[self.factor_ret_history.index[l-self.shrinkaged_window:l], :]
            resid_tmp2 = resid_tmp.loc[:, f_expo.index].fillna(0)

            # 两部分相加，就是整个的股票组合的协方差矩阵，在马科维茨模型中就表示风险
            e_ef = factor_cov_df + resid_tmp2.cov()
            shrinkaged_dict.update({self.factor_ret_history.index[l]: e_ef})

        # wr_excel(r'D:\Database_Stock\临时\e.xlsx', shrinkaged_dict, w_or_r='w')

        return shrinkaged_dict

    def rov_of_all_stocks(self, method):
        # 通过打分法选股, stock_pool 为各个股票的打分
        if method == 'score':
            stock_pool = self.select_stocks_by_scores()
        # 通过回归法选股，stock_pool 为各个股票的预测收益率
        elif method == 'regress':
            stock_pool = self.select_stocks_by_regress()

        return stock_pool

    def select_stocks_by_regress(self):

        # 预测的因子收益
        est_facor_rets = forecast_factor_return(self.alpha_factor, self.factor_ret_history, window=12)
        # 全部股票收益预测
        est_stock_rets = get_est_stock_return(self.alpha_factor, self.factors_panel, est_facor_rets, 12, 6)
        print('计算预期收益率完成...')

        est_stock_rets.name = 'stock_return'

        if isinstance(self.pool, pd.DataFrame):
            # 使用财务数据过滤一下股票池
            self.pool
            ret = est_stock_rets
        else:
            ret = est_stock_rets

        return est_stock_rets

    def select_stocks_by_scores(self):
        new_stock_pool = pd.DataFrame()

        if self.update_or_renew == 'update_111':
            p = r'D:\Database_Stock\临时\stock_pool.csv'
            if os.path.exists(p):
                new_stock_pool = pd.read_csv(p, encoding='gbk')
                new_stock_pool.set_index(new_stock_pool.columns[0], inplace=True)
                new_stock_pool.columns = pd.to_datetime(new_stock_pool.columns)

            cols = [col for col in self.pool.columns if col not in new_stock_pool.columns and
                    col > new_stock_pool.columns[-1]]
            # 无需要更新的，返回
            if len(cols) == 0:
                return new_stock_pool
            else:
            # 若有需要更新的，则选出指定的列
                pool = self.pool[cols]
        else:
            new_stock_pool = pd.DataFrame()
            pool = self.pool

        for col, value in pool.iteritems():
            codes = list(value[value == True].index)

            # 对所有股票使用同样的打分大类因子
            all_codes_scores = self.scores_for_stocks(codes, col)
            # 对不同行业的股票使用不同的打分大类因子
            if isinstance(all_codes_scores, pd.DataFrame):
                scores = self.scores_for_special_industry(all_codes_scores, codes, col)
            else:
                scores = None

            if isinstance(scores, pd.DataFrame):
                new_stock_pool = pd.concat([new_stock_pool, scores], axis=1)
            elif not scores:
                continue

        new_stock_pool = new_stock_pool.sort_index()
        new_stock_pool.fillna(0, inplace=True)

        p = r'D:\Database_Stock\临时\stock_pool.csv'
        new_stock_pool.to_csv(p, encoding='gbk')

        return new_stock_pool

    # 在打分法模型中，对所有股票使用同样的大类因子。该种做法，主要用在需要保证收益预测模型可控的时候，
    # 测算一些其他功能时使用，比如测算风险模型时、测算对冲模型时。
    def scores_for_stocks(self, codes, dt):
        scores = pd.Series()
        tmp_root = os.path.join(root_dair, '多因子选股', '正交后因子', '因子截面')
        f_list = os.listdir(tmp_root)
        fn = dt.strftime("%Y-%m-%d") + '.csv'
        if fn not in f_list:
            print('未找到'+ fn + '该期数据')
            return None

        data = pd.read_csv(os.path.join(tmp_root, fn), engine='python', encoding='gbk')
        data = data.set_index('Code')
        new_index = [c for c in codes if c in data.index]
        data = data.loc[new_index, :]
        fn = self.factor_dict_for_scores['default']

        f_list = [f for f in fn if f in data.columns]
        if len(f_list) == 0:
            return None
        icir_dat = self.icir_e_dict['default']
        if dt not in list(icir_dat.index):
            print('该日期还未计算出icir')
            return None
        icir_se = icir_dat.loc[dt, :]
        if self.wei_type == 'equal':
            wei = icir_se[f_list]/abs(icir_se[f_list])
            wei = np.array(wei / np.nansum(np.abs(wei)))
        elif self.wei_type == 'icir':
            wei = icir_se[f_list]
            wei = np.array(wei / np.nansum(wei))

        val = data[f_list].values
        wei = np.array(wei / np.nansum(wei))
        scores1 = pd.Series(index=data.index, data=np.dot(val, wei))
        scores = pd.concat([scores, scores1])

        if len(scores) == 0:
            return None
        else:
            return pd.DataFrame({dt: scores})

    # 在打分法模型中，对不同的行业使用不同的大类因子，在做单行业测试，或者行业内新因子测算的使用用到。月报里也使用该模型。
    def scores_for_special_industry(self, original_scores, codes, dt):

        score_tmp = copy.deepcopy(original_scores)
        for ind, factor_dict in self.factor_dict_for_scores.items():
            if ind == 'default':
                continue

            if ind == '计算机':
                print('debug')

            tmp_root = os.path.join(root_dair, '分行业研究', ind, '多因子选股', '新合成因子', '因子截面')
            f_list = os.listdir(tmp_root)
            fn = dt.strftime("%Y-%m-%d") + '.csv'

            if fn not in f_list:
                # print('未找到'+ fn + '该期数据')
                return None
            data = pd.read_csv(os.path.join(tmp_root, fn), engine='python', encoding='gbk')
            data = data.set_index('Code')
            new_index = [c for c in codes if c in data.index]
            data = data.loc[new_index, :]

            factor_ns = self.factor_dict_for_scores[ind]

            f_list = [f for f in factor_ns if f in data.columns]
            # 分行业选择，如果行业没在key里面，则使用default。
            if len(f_list) == 0:
                continue

            icir_dat = self.icir_e_dict[ind]
            if dt not in list(icir_dat.index):
                continue
            icir_se = icir_dat.loc[dt, :]
            icir_se = icir_se.dropna()

            # sco = get_scores_with_wei(data, f_list, icir_se)
            val = data[f_list].values

            wei = icir_se[f_list] / abs(icir_se[f_list])
            wei = np.array(wei / np.nansum(np.abs(wei)))

            # wei = np.array(icir_se[f_list] / np.nansum(icir_se[f_list]))
            scores_tmp = pd.Series(index=data.index, data=np.dot(val, wei))

            # 如果设定了每个行业仅选择N个公司
            if self.one_industry_num:
                scores_sorted = scores_tmp.sort_values(ascending=False)
                tmp = list(scores_sorted.index[:self.one_industry_num])
                scores_sorted = scores_sorted[tmp]
                return scores_sorted

            codes_in_origi = [i for i in scores_tmp.index if i in score_tmp.index]
            score_tmp.loc[codes_in_origi, score_tmp.columns[0]] = scores_tmp[codes_in_origi]

        return score_tmp

    # 根据股票池得分，选择得分排名前N的股票
    def select_top_n(self, pool):
        res_df = pd.DataFrame()
        for col, se in pool.iteritems():
            tmp_sorted = se.sort_values(ascending=False)
            man_n = int(len(tmp_sorted) * self.top_percent_n)
            sel = list(tmp_sorted[:man_n].index)
            sco = pd.Series(index=sel, data=np.ones(len(sel)))
            sco = sco / sco.sum()
            res_df = pd.concat([res_df, pd.DataFrame({col: sco})], axis=1)

        res_df = res_df.fillna(0)
        return res_df

    # 风险模型（组合优化）
    def compose_optimization(self, stock_pool, is_enhance=True, lamda=10, turnover=None, te=None,
                             industry_expose_control=True, industry_max_expose=0,
                             in_benchmark=False, in_benchmark_wei=0.8,
                             limit_factor_panel=None, max_num=None, s_type=None):
        '''
        :param stock_pool:    每期的alpha排序或预测的股票收益
        :param is_enhance:    是否是指数增强
        :param lamda:         风险厌恶系数
        :param turnover:      是否有换手率约束
        :param te:            是否有跟踪误差约束
        :param industry_max_expose  行业风险敞口，如果该值为0，则表示行业中性
        :param size_neutral   是否做市值中性
        :param in_benchmark:  是否必须成份股内选股
        :param max_num:       最大股票数量要求
        :return: stock_wei:   每期的股票权重
        '''

        his_tamp = list(set(stock_pool.columns) & set(self.E_dict.keys()))
        his_tamp.sort()
        wei_df = pd.DataFrame()
        save_path = os.path.join(root_dair, '临时')
        pre_w = None
        # todo 添加一个读取已经优化好的权重数据的功能。
        for d in his_tamp:
            # if d < datetime(2019, 4, 27):
            #             #     continue
            if d == datetime(2014, 1, 30):
                print('debug')
            # d = pd.Timestamp(datetime(2014, 1, 30))

            loc = his_tamp.index(d)
            # 如果
            if loc != 0 and not isinstance(pre_w, pd.Series):
                wei_df = pd.read_csv(os.path.join(save_path, '股票权重.csv'), engine='python')
                wei_df.set_index(wei_df.columns[0], inplace=True)
                wei_df.columns = pd.to_datetime(wei_df.columns)
                pre_w = wei_df[his_tamp[loc-1]]

            print(d)
            r_tmp = stock_pool[d]
            e_tmp = self.E_dict[d]
            bench_wei = self.index_wei[d].dropna()
            f_n = self.control_factor.keys()
            limit_factor_df = self.factors_panel[d][f_n]
            in_benchmark = in_benchmark

            wei = optimization_fun(r_tmp, e_tmp, bench_wei, pre_w=pre_w, lam=lamda, turnover=turnover, te=te,
                                   industry_expose_control=industry_expose_control,
                                   industry_max_expose=industry_max_expose, control_factor_dict=self.control_factor,
                                   limit_factor_df=limit_factor_df, in_benchmark=in_benchmark,
                                   in_benchmark_wei=in_benchmark_wei, max_num=max_num, s_max_type=s_type)
            wei_df = pd.concat([wei_df, pd.DataFrame({d: wei})], axis=1)
            # wei_df.to_csv(os.path.join(save_path, '股票权重.csv'), encoding='gbk')
            # todo pre_w应该是经过一个月的价格变动后的权重，优化后的权重要变化的。
            pre_w = self.wei_after_a_month(wei, d)

        wei_df.fillna(0, inplace=True)
        print('全部优化完毕')
        return wei_df

    # 月初的权重，经过一个月的变动后月末的权重结果
    def wei_after_a_month(self, wei, d):
        pct = self.pet_chg_m[d]
        pct = 1 + pct / 100
        new_wei = wei * pct[wei.index]
        new_wei = new_wei / new_wei.sum()
        return new_wei

    def run_test(self):
        # 负向指标选股，剔除一些不好的股票
        self.get_stock_pool_by_removing()

        # 因子合成
        self.concate_factor()

        # if self.update_or_renew != 'update':
        #     path = os.path.join(root_dair, '多因子选股', '新合成因子', '因子截面')
        #     factors = list(default_dict.keys())
        #     corr1 = self.get_factor_corr(factors, path)
        #     print(corr1)
        #     corr1.to_csv('D:\Database_Stock\相关系数.csv', encoding='gbk')

        path = os.path.join(root_dair, '多因子选股', '新合成因子', '因子截面')
        save_path = os.path.join(root_dair, '多因子选股')
        symmetry_dict = {'quality': ['value'],
                         'volatility': ['liquidity', 'mom'],
                        }
        self.factor_to_symmetry(path, save_path, symmetry_dict)

        if self.update_or_renew != 'update':
            path = os.path.join(root_dair, '多因子选股', '正交后因子', '因子截面')
            factors = list(default_dict.keys())
            corr2 = self.get_factor_corr(factors, path)
            print(corr2)

        # 如果使用打分法，那么需要计算新合成因子的ICIR值，在确定权重部分会使用到
        if self.rov_method == 'score':
            self.compute_icir()

        # 在回归法预测股票收益 或 使用风险模型条件下，需要计算因子收益和股票残差。
        if self.rov_method == 'regress' or self.use_risk_model:
            # 历史的因子表现 以及 股票的特质收益率
            self.factor_ret_history, self.resid = self.compute_factor_return()
        # 在使用风险模型条件下需要计算协方差矩阵
        if self.use_risk_model:
            self.E_dict = self.get_shrinkaged()

        # 正向选股，打分法或者是回归法，返回股票的打分或者是回归得到的预测收益率，
        stock_pool = self.rov_of_all_stocks(self.rov_method)

        # 选打分拍前N的股票，并等权配置得到权重
        if self.rov_method == 'score' and self.use_risk_model == False and self.score_mode == 'all_same' and \
            self.select_type == 'total_num':
            stock_pool = self.select_top_n(stock_pool)

        if self.use_risk_model:
            # 根据打分或是预测的收益率，以及基准，得到股票的配置权重
            is_enhance = self.opt_para_dict['is_enhance']
            lamda = self.opt_para_dict['lamda']
            turnover = self.opt_para_dict['turnover']
            te = self.opt_para_dict['te']
            industry_expose_control = self.opt_para_dict['industry_expose_control']
            industry_max_expose = self.opt_para_dict['industry_max_expose']
            in_benchmark = self.opt_para_dict['in_benchmark']
            in_benchmark_wei = self.opt_para_dict['in_benchmark_wei']
            max_num = self.opt_para_dict['max_num']
            s_type = self.opt_para_dict['s_type']
            stock_pool = self.compose_optimization(stock_pool, lamda=lamda, is_enhance=is_enhance, turnover=turnover,
                                                   te=te, industry_expose_control=industry_expose_control,
                                                   industry_max_expose=industry_max_expose,
                                                   in_benchmark=in_benchmark,
                                                   in_benchmark_wei=in_benchmark_wei, max_num=max_num, s_type=s_type)

        # # 不用shift，因为是使用预测的因子收益或排序来选择的股票
        # pp = os.path.join(root_dair, '临时', '股票权重.csv')
        # stock_pool.to_csv(pp, encoding='gbk')
        # stock_pool = pd.read_csv(pp, engine='python', encoding='gbk')
        # stock_pool.set_index(stock_pool.columns[0], inplace=True)
        # stock_pool.columns = pd.to_datetime(stock_pool.columns)

        self.final_stock_pool = stock_pool
        ana, nv = self.backtest(stock_pool, bench=self.benchmark, hedging=False, plt=True)

        return ana, nv

    def latest_wei_and_stock(self):
        latest_df = self.final_stock_pool[[self.final_stock_pool.columns[-1]]]
        # latest_df.sum()
        latest_df = latest_df.where(latest_df > 0.0, None)
        latest_df = latest_df.dropna()

        data = Data()
        stock_basic = data.stock_basic_inform
        res = stock_basic.loc[latest_df.index, 'SEC_NAME']
        res_df = pd.DataFrame({'SEC_NAME': res})
        res_df.index.name = 'CODE'
        latest_df = pd.concat([latest_df, res_df], axis=1)
        return latest_df

    # 存储相关结果
    def save(self, pool, save_name='股票池每期结果.csv'):
        basic_save_path = r'D:\pythoncode\IndexEnhancement\股票池'
        save_each_sec_name(pool, save_name)

        # 把最新股票池结果存成txt, 并下载股票池基本信息
        to_save = list(pool.index[pool[pool.columns[-1]] == True])
        add_stock_pool_txt(to_save, '盈利成长选股策略_排名前50', renew=True)
        info = section_stock_infor(to_save)
        info.to_csv(os.path.join(basic_save_path, '股票基本信息.csv'), encoding='gbk')

    # 得到最新股票池
    def latest_pool(self, method, add_infor=False):
        self.get_stock_pool_by_removing()

        self.concate_factor()
        path = os.path.join(root_dair, '多因子选股', '新合成因子', '因子截面')
        save_path = os.path.join(root_dair, '多因子选股')
        symmetry_dict = {'quality': ['value'],
                         'volatility': ['liquidity', 'mom'],
                         }
        self.factor_to_symmetry(path, save_path, symmetry_dict)
        self.compute_icir()
        stock_pool = self.rov_of_all_stocks(method)
        print(stock_pool)
        # 选打分拍前N的股票，并等权配置得到权重
        if self.rov_method == 'score' and not self.use_risk_model and self.score_mode == 'all_same' and \
                self.select_type == 'total_num':
            stock_pool = self.select_top_n(stock_pool)

        if stock_pool.empty:
            return None

        tmp = stock_pool[stock_pool.columns[-1]]
        tmp = tmp[tmp.index[tmp != 0]]
        tmp.sort_values(inplace=True)
        s_list = list(tmp.index)
        # 添加股票名称
        data = Data()
        stock_basic = data.stock_basic_inform
        res = stock_basic.loc[s_list, 'SEC_NAME']
        res_df = pd.DataFrame({'SEC_NAME': res})
        res_df.index.name = 'CODE'

        # res_df.to_csv('D://库存表.csv', encoding='gbk')

        # 是否添加概念数据和调入股票池时间数据
        if add_infor:
            # 添加调入股票池日期数据
            res_df['跳入股票池日期'] = None
            for k, v in stock_pool.loc[s_list, :].iterrows():
                for i in range(len(v)-1, -1, -1):
                    if not v[i]:
                        res_df.loc[k, '跳入股票池日期'] = v.index[i+1]
                        break

            # 添加概念数据。
            concept = data.concept
            res_df = pd.concat([res_df, concept], axis=1, join='inner')

            for k, v in res_df['CONCEPT'].items():
                try:
                    res_df.loc[k, 'CONCEPT'] = v.replace('[', '').replace(']', '').replace('\'', '')
                except Exception:
                    pass

        return res_df, stock_pool

    def backtest(self, stock_pool, bench='WindA', hedging=False, plt=True):

        bt = BackTest(stock_pool, self.freq, adjust_freq=self.freq, fee_type='fee',
                      benchmark_str=bench, hedge_status=hedging)
        bt.run_bt()
        ana = bt.analysis()
        nv = bt.net_value
        if plt:
            bt.plt_pic()
        print(bt.analysis())
        return ana, nv

    # 得到特定月份的股票池
    def special_month_pool(self, special_date, stock_pool=None):
        if not isinstance(stock_pool, pd.DataFrame):
            self.get_stock_pool()
            stock_pool = self.main_of_select_stocks_by_scores()

        if stock_pool.empty:
            return None

        ff = None
        for c in stock_pool.columns:
            if c.year == special_date.year and c.month == special_date.month:
                ff = c
                break

        tmp = stock_pool[ff]
        s_list = list(tmp.index[tmp == True])
        return s_list


def growth_stock_pool(method='score', score_m='all_same', select_type='by_industry', risk_model=False,
                      bt_or_latest='bt', special_market_dict={}, update_ornot='update', para_d=None,
                      indus_d=None, bm='ZZ500', start_d=None, fd_for_scores=None, percent_n=0.1,
                      risk_factor=None):

    # 财务绝对要求条件的股票池构建，如roettm大于5%，eps同比增速大于5，sue位于所有股票的top5%。
    financial_dict = {'all': {'scope_0': ('roettm', -99, None),
                              # 'scope_1': ('basicepsyoy', 5, 500),
                              # 'scope_2': ('netprofitgrowrate', 5, 500),  # 净利润同比增长率
                              # 'scope_3': ('debtassetsratio', 1, 60),
                              # 'rise_0': ('netprofitgrowrate', 2)
                              }
                      }

    # indus_dict = {'to_handle_indus': ['有色金属', '钢铁', '采掘', '非银金融'],
    #               # keep 和 delete 两种模式， keep模型下，保留下来上个变量定义的子行业，delete模式下，删除上面的子行业
    #               'handle_type': 'delete',
    #               }

    # 仅对家用电器做一下，看看结果如何
    # indus_dict = {'keep': '家用电器'}
    select_m = select_type   # 'total_num'
    if not fd_for_scores:
        print('fd_for_scores 未定义')
        raise ValueError

    if not para_d:
        para_d = {'is_enhance': True,
                  'lamda': 10,
                  'turnover': 2,
                  'te': None,
                  'industry_max_expose': 0,
                  'industry_expose_control': True,
                  'size_neutral': False,
                  'in_benchmark': False,
                  'in_benchmark_wei': 0.8,
                  'max_num': 100,
                  's_type': 'tight',
                  'control_factor': {'size': 1},
                 }

    control_factor = para_d['control_factor']

    pool = FormStockPool(financial_dict, use_risk_model=risk_model, start_date=start_d,
                         factor_dict_for_scores=fd_for_scores, special_market_dict=special_market_dict,
                         risk_factor=risk_factor, control_factor=control_factor, opt_para_dict=para_d, method=method,
                         score_mode=score_m, select_mode=select_m, benchmark=bm,
                         percent_n=percent_n, industry_dict=indus_d, update_ornot=update_ornot)

    if bt_or_latest == 'latest':
        newest_pool, stock_pool = pool.latest_pool(method=method, add_infor=True)
        newest_pool.to_csv(r'D:\Database_Stock\股票池_最终\盈利成长因子选股下个月股票池.csv', encoding='gbk')
        # save_path = r'D:\pythoncode\IndexEnhancement\股票池_最终'
        # tod = datetime.today()
        # per_month = datetime(tod.year, tod.month - 1, 1)
        # per_pre_month = datetime(tod.year, tod.month - 2, 1)
        #
        # sl = pool.special_month_pool(per_pre_month, stock_pool=stock_pool)
        # res1, res2 = month_return_compare_to_market_index(sl, per_month)
        # res1.to_csv(os.path.join(save_path, '盈利成长因子上个月组合表现.csv'), encoding='gbk')
        # res2.to_csv(os.path.join(save_path, '盈利成长因子上个月个股表现.csv'), encoding='gbk')

    elif bt_or_latest == 'bt':
        res, nv = pool.run_test()
        latest_wei = pool.latest_wei_and_stock()
        return res, nv, latest_wei


if '__main__' == __name__:

    para_set_mud = 'my_way'  #  'regular'      # 'regular', 'my_way'
    # 参数组合说明与设置：
    if para_set_mud == 'regular':
        # 1，对于常规测算的使用打分法的的多因子选股模型，建议配置参数为：
        bt_or_latest = 'bt'
        method = 'score'    
        special_market_dict = None
        score_mode = 'all_same'
        use_risk_model = True
        select_type = None
        bm = 'HS300'
        fd = None
        update_ornot = 'update'         # 'update'   'renew'
        start_date = datetime(2010, 1, 1)
        risk_factor = ['size', 'volatility']
        alpha_factor = { }

    elif para_set_mud == 'ei':
        # 2，对于我的主要研究方向，不同行业使用不同的基本面因子选股的策略，检验配置的参数为：
        method = 'score'    
        score_mode = 'each_industry'
        special_market_dict = {'delete': '创业板'}
        use_risk_model = False
        select_type = None
        bm = 'WindA'
        indus_d = {'keep': '汽车'}
    elif para_set_mud == 'my_way':
        # 3，对于我前面盈利成长选股策略的跟踪，检验配置的参数为：
        indus_d = {'handle_type': 'keep',
                   'to_handle_indus': ['国防军工']
                  }
        method = 'score'    
        special_market_dict = None
        score_mode = 'all_same'
        use_risk_model = False
        select_type = 'total_num'
        bm = '国防军工'
        start_date = None
        start_date = datetime(2010, 1, 1)
        bt_or_latest = 'latest'            # 'bt' , 'latest'
        update_ornot = 'renew'         # 'renew'  'update'
        fd = {}
        for key, values in factor_dict_to_concate.items():
            fd.update({key: list(values.keys())})

        percent_n = 0.1
        use_risk_model = False
        para_dict = None
        risk_factor = None

    # 若不使用风险模型，则把use_risk_model设置成False
    # 根据基准的不同，可以设置 bm 为 'HS300','ZZ500','SZ50','WindA'
    if use_risk_model:
        # 如果使用风险模型，风险模型的参数
        para_dict = {'is_enhance': True,
                     'lamda': 2,
                     'turnover': 0.4,          # 0.3,
                     'te': None,                # 0.4,
                     'industry_expose_control': True,    # 是否行业中性
                     'industry_max_expose': 0.02,
                     'size_neutral': True,
                     'in_benchmark': False,
                     'in_benchmark_wei': 0.9,     # 0.8,
                     'max_num': 150,
                     's_type': 'tight',             # 'tight',
                     'control_factor': {'size': 0},
                    }

        if para_dict['te']:
            risk_factor = list(default_dict.keys())
            risk_factor.remove('size')
    # 如果需要剔除某些行业或者保留某些行业，可以设置indus_d参数， 如仅保留家电，可设置为：indus_d = {'keep': '家用电器'}
    # 如果需要剔除钢铁，则 indus_d = {'delete': '钢铁'}
    # select_type = None                            # 'total_num' 'twice_sort' None
    # bt_or_latest = 'bt'                           # 'bt' 'latest'  'latest_pool_daily'
    #
    # method = 'score'
    # special_market_dict = None
    # score_mode = 'all_same'
    # use_risk_model = True
    # select_type = None
    # bm = 'ZZ500'
    # update_ornot = 'update'     #  'update'
    # start_date = datetime(2013, 1, 1)
    #
    # para_dict = {'is_enhance': True,
    #              'lamda': 10,
    #              'turnover': None,
    #              'te': None,
    #              'industry_expose_control': False,
    #              'industry_max_expose': 0.0,
    #              'size_neutral': False,
    #              'in_benchmark': True,
    #              'max_num': 100,
    #              }

    res, nv, latest_wei = growth_stock_pool(method=method, score_m=score_mode, select_type=select_type,
                                            risk_model=use_risk_model, bt_or_latest=bt_or_latest, percent_n=percent_n,
                                            risk_factor=risk_factor, special_market_dict=special_market_dict,
                                            update_ornot=update_ornot, para_d=para_dict, indus_d=indus_d,
                                            bm=bm, start_d=start_date, fd_for_scores=fd)

    save_path = r'D:\Database_Stock\临时'
    res.to_csv(os.path.join(save_path, '沪深300_final_指标.csv'), encoding='gbk')
    nv.to_csv(os.path.join(save_path, '沪深300_final_净值.csv'), encoding='gbk')



