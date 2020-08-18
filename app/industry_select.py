from datetime import datetime
import pandas as pd
import numpy as np
from utility.tool0 import Data
from utility.factor_data_preprocess import info_cols, add_to_panels, apply_func2,\
                                           concat_factor_2, simple_func, drop_some, winsorize
from utility.stock_pool import compute_icir
import os
from utility.constant import root_dair, industry_factor_names

'''
基于多因子打分法的行业轮动模型，针对申万三级行业。
行业层面有效因子较少，主要使用动量、盈利、成长三大类因子。
最后的结果需要更多的结合行业基本面逻辑进行选择判断，定性分析因素占比远大于定量因素。
具体因子可见 constant.py 中定义的常量 industry_factor_names
原始报告可见《行业多因子轮动模型――金融工程专题报告》。
'''


class IndustrySelect:
    def __init__(self, alpha_factor_names=industry_factor_names, update_only=True, max_num=20):
        self._data = Data()
        self.stock_factor_path = os.path.join(root_dair, '因子预处理模块', '因子')
        self.industry_factor_path = os.path.join(root_dair, '行业多因子', '申万三级', '因子')
        self.save_path = os.path.join(root_dair, '行业多因子', '申万三级')

        self.icir_e = None
        self.factors = alpha_factor_names
        self.update_only = update_only
        self.max_num = max_num
        self.indus_selected = None

    # 定义不同因子的合成方式，财务因子使用中位数法合成，价量因子使用市值加权法合成
    def compose_way(self):
        median_factors_0 = ['Sales_G_q', 'Profit_G_q', 'ROE_G_q', 'ROE_q', 'ROE_ttm', 'ROA_q', 'ROA_ttm',
                            'grossprofitmargin_q',
                            'grossprofitmargin_ttm', 'profitmargin_q', 'profitmargin_ttm', 'assetturnover_q',
                            'assetturnover_ttm', 'operationcashflowratio_q', 'operationcashflowratio_ttm',
                            'SUE', 'REVSU']

        median_factors_1 = []
        # 改成首字母大写后续字母小写的格式
        for col in median_factors_0:
            new_c = col[0].upper() + col[1:].lower()
            median_factors_1.append(new_c)

        compose_way = {'median': median_factors_1}
        self.compute_factor(compose_way)

    # 根据不同因子的合成方式，计算行业因子，并保存到指定文件夹下
    def compute_factor(self, compose_way):
        data_path = self.stock_factor_path
        indus_save_path = self.industry_factor_path

        stock_basic_inform = self._data.stock_basic_inform
        # 创建文件夹
        if not os.path.exists(os.path.join(indus_save_path)):
            os.makedirs(os.path.join(indus_save_path))

        if not self.update_only:
            to_process_f = os.listdir(data_path)
        elif self.update_only:
            fls = os.listdir(data_path)
            processed_list = os.listdir(indus_save_path)
            to_process_f = [f for f in fls if f not in processed_list]

        if len(to_process_f) == 0:
            print('无需要处理的数据')
            return None

        for panel_f in to_process_f:
            print(panel_f)
            # panel_f = os.listdir(date_path)[0]
            panel_dat = pd.read_csv(os.path.join(data_path, panel_f),
                                    encoding='gbk', engine='python',
                                    index_col=['Code'])

            # 需要先对股票因子做两个常规处理
            data_to_process = drop_some(panel_dat)
            # data_to_process.empty
            data_to_process = winsorize(data_to_process)
            data_to_process = pd.concat([data_to_process, stock_basic_inform['申万三级行业']], axis=1, join='inner')
            factors_to_concat = list((set(panel_dat.columns) - (set(info_cols) - set(['Pct_chg_nm']))))
            grouped = data_to_process.groupby('申万三级行业')

            ind_factor = pd.DataFrame()
            for name, group in grouped:
                factor_dat = group[factors_to_concat]
                mv = group['Mkt_cap_float']
                factor_dat = factor_dat.applymap(apply_func2)
                factor_concated = {}
                for factor_name, factors in factor_dat.iteritems():
                    if factor_name == 'Lncap_barra':
                        tmp_f = np.log(np.sum(group['Mkt_cap_float']))
                        factor_concated.update({factor_name: tmp_f})
                        continue

                    # 不同类型因子有不同的合成方式
                    factor_concat_way = 'mv_weighted'
                    for concat_way, factorlist in compose_way.items():
                        factorlist_tmp = [fa.lower() for fa in factorlist]
                        if factor_name.lower() in factorlist_tmp:
                            factor_concat_way = concat_way
                    tmp_f = simple_func(factors, mv=group['Mkt_cap_float'], type=factor_concat_way)

                    factor_concated.update({factor_name: tmp_f})

                factor_concated = pd.DataFrame(factor_concated)
                factor_concated.index = [name]
                factor_concated.loc[name, 'Mkt_cap_float'] = np.sum(mv)  # 市值采用行业市值和
                if 'Industry_zx' in group.columns:
                    factor_concated.loc[name, 'Industry_zx'] = group.loc[group.index[0], 'Industry_zx']
                if 'Industry_sw' in group.columns:
                    factor_concated.loc[name, 'Industry_sw'] = group.loc[group.index[0], 'Industry_sw']
                ind_factor = pd.concat([ind_factor, factor_concated], axis=0)

            ind_factor.index.name = 'Name'
            ind_factor.to_csv(os.path.join(indus_save_path, panel_f), encoding='gbk')

    # 计算指定因子的icir值
    def compute_icir(self):
        icir = compute_icir(12, self.industry_factor_path, self.save_path, self.factors)
        icir_e = icir.shift(1, axis=0).dropna(axis=0)
        self.icir_e = icir_e
        return icir_e

    # 通过icir加权打分法选择行业
    def select_indus(self):
        max_num = self.max_num

        indus_selected = pd.DataFrame()

        factor_panel_path = os.path.join(root_dair, '行业多因子', '申万三级', '因子')
        save_path = os.path.join(root_dair, '行业多因子', '申万三级')
        panel_list = os.listdir(factor_panel_path)
        for fn in panel_list:
            f_datetime = datetime.strptime(fn.split('.')[0], "%Y-%m-%d")

            data = pd.read_csv(os.path.join(factor_panel_path, fn), engine='python', encoding='gbk')
            data.set_index('Name', inplace=True)

            if f_datetime not in list(self.icir_e.index):
                continue
            icir_se = self.icir_e.loc[f_datetime, :]

            val = data[self.factors].values
            wei = np.array(icir_se[self.factors] / np.nansum(icir_se[self.factors]))
            scores = pd.Series(index=data.index, data=np.dot(val, wei))

            # 排序
            scores_sorted = scores.sort_values(ascending=False)
            selected_indus = list(scores_sorted.index[:max_num])

            selected_df = pd.DataFrame(index=range(0, len(selected_indus)), columns=[f_datetime], data=selected_indus)
            indus_selected = pd.concat([indus_selected, selected_df], axis=1)

        indus_selected.to_csv(os.path.join(save_path, '行业选择结果.csv'), encoding='gbk')
        self.indus_selected = indus_selected

    def show_newest(self):
        print(self.indus_selected[self.indus_selected.columns[-1]])


if __name__ == '__main__':
    ise = IndustrySelect()
    ise.compute_icir()
    ise.select_indus()
    ise.show_newest()

    # select_indus()
    # generate_palte_pct(plate_to_indus)











