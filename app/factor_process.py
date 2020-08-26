# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:47:54 2019

@author: admin
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from utility.factor_data_preprocess import drop_some, fill_na, winsorize,\
                                        neutralize, standardize, process_input_names
from utility.single_factor_test import get_firt_industry_list
from utility.constant import info_cols
# from canslim.update_stock_pool import form_stock_pool
from utility.tool0 import Data
from utility.constant import data_dair, root_dair, industry_benchmark


def main(p_dict, fp, is_ind_neu, is_size_neu, is_plate_neu, special_plate=None,
         selection=None):
    """
    is_ind_neu : 是否做行业中性化处理，对股票多因子需要，做行业多因子时不需要
    输出： 预处理后的因子截面数据（如2009-01-23.csv文件）

    顺序：缺失值填充、去极值、中性化、标准化
    （因输入的截面数据中所含财务类因子默认已经过
    财务日期对齐处理，故在此不再进行该步处理）
    注：针对无需处理的因子，如Rps，把因子名称添加到constant文件中的info_cols变量中，相关函数会通过import该变量的方式导入
    并跳过处理过程
    """

    file_path = p_dict['file_path']
    save_path = p_dict['save_path']

    # 读取原始因子截面数据
    try:
        data = pd.read_csv(os.path.join(file_path, fp), engine='python',
                           encoding='gbk')
    except Exception as e:
        print('debug')
    if 'No' in data.columns:
        data = data.set_index('No')

    # 若针对特定板块，则删除其他板块的股票数据
    if special_plate:
        data_ = Data()
        stock_basic = data_.stock_basic_inform
        sw_1 = stock_basic[['申万一级行业']]
        stock_list = list(sw_1.index[sw_1[sw_1.columns[0]] == special_plate])
        codes = [i for i in data.index if data.loc[i, 'Code'] in stock_list]
        data = data.loc[codes, :]
        data.index = range(0, len(data))

    # 历史回测：删除一些未上市的股票、下个月未开盘的股票
    # 跟踪：无动作
    data_to_process = drop_some(data)

    # 预处理步骤依次进行
    data_to_process = fill_na(data_to_process)                                    # 缺失值填充
    if len(data_to_process) == 0:
        print('debug')
    data_to_process = winsorize(data_to_process)                                  # 去极值
    if is_ind_neu or is_size_neu:
        data_to_process = neutralize(data_to_process, ind_neu=is_ind_neu, size_neu=is_size_neu)  # 中性化
    data_to_process = standardize(data_to_process)                                # 标准化

    data_final = data_to_process

    if data_final.index.name != 'No':
        data_final.index = range(1, len(data_final)+1)
        data_final.index.name = 'No'

    data_final.to_csv(os.path.join(save_path, fp), encoding='gbk')


def factor_preprocess_update():
    is_ind_neu = True  # 股票时需要，为True，行业时或单行业测试时不需要，为False,
    is_plate_neu = False  # 板块中性
    is_size_neu = True  # 是否需要对市值做中性化

    path_dict = {
                 'file_path': os.path.join(root_dair, '因子预处理模块', '因子'),
                 'save_path': os.path.join(root_dair, '因子预处理模块', '因子（已预处理）'),
                 }

    fls = os.listdir(path_dict['file_path'])
    processed_list = os.listdir(path_dict['save_path'])
    to_process_f = [f for f in fls if f not in processed_list]

    if len(to_process_f) == 0:
        print('无需要处理的数据')

    # 对所有横截面数据进行遍历
    for fpath in to_process_f:
        print('目前处理的月份为：')
        print(fpath)
        main(path_dict, fpath, is_ind_neu, is_size_neu, is_plate_neu)
    print('因子截面数据已全部处理！')


def factor_preprocess_for_indus_factor():
    is_ind_neu = False  # 股票时需要，为True，行业时或单行业测试时不需要，为False,
    is_plate_neu = False  # 板块中性
    is_size_neu = False  # 是否需要对市值做中性化

    path_dict = {
                 'file_path': os.path.join(root_dair, '行业多因子', 'second_industry', '因子'),
                 'save_path': os.path.join(root_dair, '行业多因子', 'second_industry', '因子（已预处理）'),
                 }

    fls = os.listdir(path_dict['file_path'])
    processed_list = os.listdir(path_dict['save_path'])
    to_process_f = [f for f in fls if f not in processed_list]

    if len(to_process_f) == 0:
        print('无需要处理的数据')
    else:
        # 对所有横截面数据进行遍历
        for fpath in to_process_f:
            print('目前处理的月份为：')
            print(fpath)
            main(path_dict, fpath, is_ind_neu, is_size_neu, is_plate_neu)
        print('因子截面数据已全部处理！')


if __name__ == '__main__':

    is_ind_neu = False         # 股票时需要，为True，行业时或单行业测试时不需要，为False,
    is_size_neu = True        # 是否需要对市值做中性化
    formed_stock_pool = None
    test_type = 'each_industry'       # 'stock'  'each_industry'  #
    indus_list = get_firt_industry_list()
    is_update = False          # 是否仅对更新的数据进行处理

    # 收集需要处理的因子名称
    # factor_names = input("请输入需处理的因子名称（请使用英文逗号','分隔多个因子名称，输入'a'代表全部处理）：")
    factor_names = process_input_names('a')

    # 股票
    if test_type == 'stock':
        path_dict = {
            'file_path': os.path.join(root_dair, '因子预处理模块', '因子'),
            'save_path': os.path.join(root_dair, '因子预处理模块', '因子（已预处理）'),
        }
        # 创建处理后因子的存放目录
        if not os.path.exists(path_dict['save_path']):
            os.makedirs(path_dict['save_path'])

        if is_update:
            to_deal_fpath = [fp for fp in os.listdir(path_dict['file_path'])
                             if fp not in os.listdir(path_dict['save_path'])]
        else:
            to_deal_fpath = os.listdir(path_dict['file_path'])[:]

            # 对所有横截面数据进行遍历
        for fpath in to_deal_fpath:
            # fpath = '2017-11-30.csv'
            print('目前处理的月份为：')
            print(fpath)
            # if fpath == '2018-02-28.csv':
            #     print('here')
            main(path_dict, fpath, is_ind_neu, is_size_neu, factor_names)
        print('因子截面数据已全部处理！')

    elif test_type == 'each_industry':
        is_ind_neu = False
        completed = []
        for indus in indus_list:
        # if True:
            # if indus in completed:
            #     continue
            # special_plate = '建筑材料'  # indus         # '建筑材料'  # '采掘'  #
            print(indus)

            path_dict = {
                'file_path': os.path.join(root_dair, '因子预处理模块', '因子'),
                'save_path': os.path.join(root_dair, '分行业研究', indus, '因子（已预处理）'),
            }
            # 创建处理后因子的存放目录
            if not os.path.exists(path_dict['save_path']):
                os.makedirs(path_dict['save_path'])

            # 对所有横截面数据进行遍历
            for fpath in os.listdir(path_dict['file_path'])[:]:
                main(path_dict, fpath, is_ind_neu, is_size_neu, factor_names, special_plate=indus)
            print('{}行业的因子截面数据已全部处理！'.format(indus))

