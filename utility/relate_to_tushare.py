import pandas as pd
import numpy as np
import os
from time import sleep
import copy
import os
from datetime import datetime, timedelta
import tushare as ts
from utility.tool0 import Data
from utility.constant import NH_index_dict, tds_interface, date_dair
from WindPy import *

# Wind里面的后缀是CFE, tushare的后缀是CFX

if tds_interface == 'tushare':
    token_path = r'D:\文档\OneDrive\earning money\入职后资料\token_tushare.txt'
    if os.path.exists(token_path):
        f = open(token_path)
        token = f.read()
        f.close()

        ts.set_token(token)
        pro = ts.pro_api()


# 得到月度频率数据的日期
def generate_months_ends():
    tds = trade_days()
    months_end = []
    for i in range(1, len(tds)):
        if tds[i].month != tds[i-1].month:
            months_end.append(tds[i-1])
        elif i == len(tds) - 1:
            months_end.append(tds[i])

    # 看最后一个值是不是月末最后一天
    w.start()
    toma = w.tdaysoffset(1, months_end[-1], usedf=True)[1]
    toma = toma.iloc[0, 0]
    if toma.month == months_end[-1].month:
        months_end.pop()

    return months_end


# 返回股指主力合约的映射关系
def get_domain():
    try:
        domain_if = pro.fut_mapping(ts_code='IF.CFX')
        domain_ih = pro.fut_mapping(ts_code='IH.CFX')
        domain_ic = pro.fut_mapping(ts_code='IC.CFX')

        return domain_if, domain_ih, domain_ic
    except Exception as e:
        return None


def update_stock_future_dat():
    update_each_future()
    update_futmap()


def update_each_future():

    # 1, 下载所有的日度价量信息
    data = Data()

    # 得到所有股指期货历史合约的code和上市日期、退市日期
    his_df = pro.fut_basic(exchange='CFFEX', fields='ts_code,symbol,fut_code,name,list_date,delist_date')
    # 删除一些连续啊之类名字的合约
    his_df.drop(his_df.index[pd.isna(his_df['list_date'])], axis=0, inplace=True)
    # 保留股指的，删除国债期货的
    save_l = []
    for i, v in his_df['symbol'].items():
        if 'IF' in v or 'IH' in v or 'IC' in v:
            save_l.append(i)

    his_df = his_df.loc[save_l, :]
    his_df = his_df.drop(['symbol'], axis=1)
    his_df = his_df.set_index('ts_code')
    his_df = his_df.sort_values("list_date")

    try:
        all_his_open_df = data.sf_open_daily
        all_his_close_df = data.sf_close_daily
        all_his_vol_df = data.sf_vol_daily
        all_his_amount_df = data.sf_amount_daily
        all_his_oi_df = data.sf_oi_daily
        start_d = all_his_open_df.columns[-1]
    except Exception as e:
        all_his_open_df = pd.DataFrame()
        all_his_close_df = pd.DataFrame()
        all_his_vol_df = pd.DataFrame()
        all_his_amount_df = pd.DataFrame()
        all_his_oi_df = pd.DataFrame()
        start_d = datetime(2009, 1, 1).strftime("%Y%m%d")

    # 新的合约
    new_contracts = [c for c in his_df.index if c not in all_his_open_df.index]
    # 已经有的合约
    his_contracts = [c for c in his_df.index if c in all_his_open_df.index]

    # todo 明天验证一下
    for i, se in his_df.loc[his_contracts, :].iterrows():
        # 确定截至日期
        if datetime.strptime(se['delist_date'], "%Y%m%d") >= datetime.today():
            ed = (datetime.today() - timedelta(1)).strftime("%Y%m%d")
        else:
            # 该合约已经退市，做下一个循环
            continue

        res_tmp = pro.fut_daily(ts_code=i, start_date=start_d, end_date=ed)

        res_tmp.set_index('trade_date', inplace=True)
        res_tmp.index = pd.to_datetime(res_tmp.index)
        res_tmp.sort_index(inplace=True)

        all_his_open_df.loc[i, res_tmp.index] = res_tmp['open']
        all_his_close_df.loc[i, res_tmp.index] = res_tmp['close']
        all_his_vol_df.loc[i, res_tmp.index] = res_tmp['vol']
        all_his_amount_df.loc[i, res_tmp.index] = res_tmp['amount']
        all_his_oi_df.loc[i, res_tmp.index] = res_tmp['oi']
        # 您每分钟最多访问该接口20次
        sleep(3)

    if len(new_contracts) > 0:
        # 有新的合约上市交易
        for i, se in his_df.loc[new_contracts, :].iterrows():
            ed = (datetime.today() - timedelta(1)).strftime("%Y%m%d")

            res_tmp = pro.fut_daily(ts_code=i, start_date=se['list_date'], end_date=ed)

            res_tmp.set_index('trade_date', inplace=True)
            res_tmp.index = pd.to_datetime(res_tmp.index)
            res_tmp.sort_index(inplace=True)

            all_his_open_df = pd.concat([all_his_open_df, pd.DataFrame({i: res_tmp['open']}).T], axis=0)
            all_his_close_df = pd.concat([all_his_close_df, pd.DataFrame({i: res_tmp['close']}).T], axis=0)
            all_his_vol_df = pd.concat([all_his_vol_df, pd.DataFrame({i: res_tmp['vol']}).T], axis=0)
            all_his_amount_df = pd.concat([all_his_amount_df, pd.DataFrame({i: res_tmp['amount']}).T], axis=0)
            all_his_oi_df = pd.concat([all_his_oi_df, pd.DataFrame({i: res_tmp['oi']}).T], axis=0)
            # 您每分钟最多访问该接口20次
            sleep(3)

    p = os.path.join(date_dair, 'index', 'stock_future')
    data.save(all_his_open_df, 'sf_open_daily', save_path=p)
    data.save(all_his_close_df, 'sf_close_daily', save_path=p)
    data.save(all_his_vol_df, 'sf_vol_daily', save_path=p)
    data.save(all_his_amount_df, 'sf_amount_daily', save_path=p)
    data.save(all_his_oi_df, 'sf_oi_daily', save_path=p)
    print('期货合约价量数据下载完毕')


def update_futmap():
    data = Data()
    try:
        fut_map = data.fut_map.T
        start_d = fut_map.index[-1].strftime("%Y%m%d")
    except Exception as e:
        fut_map = pd.DataFrame()
        start_d = datetime(2009, 1, 1).strftime("%Y%m%d")

    end_d = datetime.today().strftime("%Y%m%d")

    domain_if = pro.fut_mapping(ts_code='IF.CFX', start_date=start_d, end_date=end_d)
    domain_ih = pro.fut_mapping(ts_code='IH.CFX', start_date=start_d, end_date=end_d)
    domain_ic = pro.fut_mapping(ts_code='IC.CFX', start_date=start_d, end_date=end_d)

    domain_if.set_index('trade_date', inplace=True)
    domain_ih.set_index('trade_date', inplace=True)
    domain_ic.set_index('trade_date', inplace=True)

    domain_if.drop('ts_code', axis=1, inplace=True)
    domain_ih.drop('ts_code', axis=1, inplace=True)
    domain_ic.drop('ts_code', axis=1, inplace=True)

    domain_if.columns = ['IF']
    domain_ih.columns = ['IH']
    domain_ic.columns = ['IC']

    domain_if.index = pd.to_datetime(domain_if.index)
    domain_if.sort_index(inplace=True)
    domain_ih.index = pd.to_datetime(domain_ih.index)
    domain_ih.sort_index(inplace=True)
    domain_ic.index = pd.to_datetime(domain_ic.index)
    domain_ic.sort_index(inplace=True)

    domain_fut = pd.concat([domain_if, domain_ih, domain_ic], axis=1)
    domain_fut = domain_fut.applymap(lambda x: x.split('.CFX')[0] + '.CFE' if isinstance(x, str) else x)

    dupl = [i for i in domain_fut.index if i in fut_map.index]
    domain_fut.drop(dupl, axis=0, inplace=True)

    fut_map = pd.concat([fut_map, domain_fut], axis=0)
    p = os.path.join(date_dair, 'index')
    data.save(fut_map, 'fut_map', save_path=p)

    print('期货主力合约映射表下载完毕')


def get_main_net_buy_amout():
    # 主力净买入额：大单买入金额+特大单买入金额-大单卖出金额-特大单卖出金额
    # buy_lg_amount	float	Y	大单买入金额（万元）
    # sell_lg_amount	float	Y	大单卖出金额（万元）
    # buy_elg_amount	float	Y	特大单买入金额（万元）
    # sell_elg_amount	float	Y	特大单卖出金额（万元
    stock_data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

    history_file = r'D:\pythoncode\IndexEnhancement\barra_cne6\download_from_juyuan\main_net_buy_amount.csv'
    if os.path.exists(history_file):
        dat_pd = pd.read_csv(history_file, engine='python')
        dat_pd = dat_pd.set_index(dat_pd.columns[0])
        dat_pd.columns = pd.to_datetime(dat_pd.columns)

        stocks = [i for i in stock_data['ts_code'] if i not in dat_pd.index]
    else:
        dat_pd = pd.DataFrame()
        stocks = stock_data['ts_code']

    for code in stocks:
        print(code)
        try:
            df = pro.moneyflow(ts_code=code, start_date='20090101', end_date='20191115')
        except Exception as e:
            break
        tmp_s = df['buy_lg_amount'] + df['buy_elg_amount'] - df['sell_lg_amount'] - df['sell_elg_amount']
        tmp_pd = pd.DataFrame(data=tmp_s.values, index=pd.to_datetime(df['trade_date']).values, columns=[code])
        tmp_pd = tmp_pd.sort_index()
        dat_pd = pd.concat([dat_pd, tmp_pd.T], axis=0)
        time.sleep(0.5)

    if len(stocks) > 0:
        save_path = r'D:\pythoncode\IndexEnhancement\barra_cne6\download_from_juyuan'
        dat_pd.to_csv(os.path.join(save_path, 'main_net_buy_amount.csv'), encoding='gbk')


def update_main_net_buy_amout():
    history_file = r'D:\pythoncode\IndexEnhancement\barra_cne6\download_from_juyuan\main_net_buy_amount.csv'
    dat_pd = pd.read_csv(history_file, engine='python')
    dat_pd = dat_pd.set_index(dat_pd.columns[0])
    dat_pd.columns = pd.to_datetime(dat_pd.columns)

    st = dat_pd.columns[-1] + timedelta(1)
    ed = datetime.today() - timedelta(1)
    st = st.strftime("%Y%m%d")
    ed = ed.strftime("%Y%m%d")

    days = pro.trade_cal(exchange='', start_date=st, end_date=ed)
    days = days[days['is_open'] == 1]

    if days.empty:
        print("现金流数据：已经更新到最新数据，自动退出")
        return None

    add_pd = pd.DataFrame()
    for d in days['cal_date']:
        df = pro.moneyflow(trade_date=d)
        df = df.set_index('ts_code')
        tmp_s = df['buy_lg_amount'] + df['buy_elg_amount'] - df['sell_lg_amount'] - df['sell_elg_amount']
        tmp_pd = pd.DataFrame(tmp_s, columns=[datetime.strptime(d, "%Y%m%d")])
        add_pd = pd.concat([add_pd, tmp_pd], axis=1)

    dat_pd = pd.concat([dat_pd, add_pd], axis=1)
    save_path = r'D:\pythoncode\IndexEnhancement\barra_cne6\download_from_juyuan'
    dat_pd.to_csv(os.path.join(save_path, 'main_net_buy_amount.csv'), encoding='gbk')


# 计算主力净买入额占流通市值比率
def get_net_buy_rate():

    data = Data()
    negotiablemv = data.negotiablemv_daily
    main_net_buy_amount = data.main_net_buy_amount
    dat_pd = main_net_buy_amount/negotiablemv

    dat_pd.dropna(how='all', axis=1, inplace=True)

    save_path = r'D:\pythoncode\IndexEnhancement\barra_cne6\download_from_juyuan'
    dat_pd.to_csv(os.path.join(save_path, 'main_net_buy_ratio.csv'), encoding='gbk')


def update_adj_factor():
    history_file = os.path.join(date_dair, 'download_from_juyuan', 'adjfactor.csv')
    dat_pd = pd.read_csv(history_file, engine='python')
    dat_pd = dat_pd.set_index(dat_pd.columns[0])
    dat_pd.columns = pd.to_datetime(dat_pd.columns)

    st = dat_pd.columns[-1] + timedelta(1)

    if datetime.today().hour < 16:
        ed = datetime.today() - timedelta(1)
    else:
        ed = datetime.today()

    st = st.strftime("%Y%m%d")
    ed = ed.strftime("%Y%m%d")

    days = pro.trade_cal(exchange='', start_date=st, end_date=ed)
    days = days[days['is_open'] == 1]

    if days.empty:
        print("复权数据：已经更新到最新数据，自动退出")
        return None

    add_pd = pd.DataFrame()
    for d in days['cal_date']:
        df = pro.query('adj_factor', trade_date=d)
        df = df.set_index('ts_code')
        tmp_s = df['adj_factor']
        tmp_pd = pd.DataFrame(data=tmp_s.values, index=tmp_s.index, columns=[datetime.strptime(d, "%Y%m%d")])
        add_pd = pd.concat([add_pd, tmp_pd], axis=1)

    dat_pd = pd.concat([dat_pd, add_pd], axis=1)
    dat_pd.to_csv(history_file, encoding='gbk')


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


def insert_to_df(inserted_df, new_df):
    dat_df = copy.deepcopy(inserted_df)
    cls = [col for col in new_df.columns]
    for c in cls:
        dat_df[c] = new_df[c]

    return dat_df


def update_stock_daily_price():

    data = Data()
    openprice_pd = data.openprice_daily
    highprice_pd = data.highprice_daily
    closeprice_pd = data.closeprice_daily
    lowprice_pd = data.lowprice_daily
    changePCT_pd = data.changepct_daily
    t_volume = data.turnovervolume_daily             # 成交量
    t_value = data.turnovervalue_daily               # 成交额（万元）

    st = np.min([openprice_pd.columns[-1], highprice_pd.columns[-1],
                 closeprice_pd.columns[-1], lowprice_pd.columns[-1],
                 changePCT_pd.columns[-1], t_volume.columns[-1],
                 t_value.columns[-1]])

    if datetime.today().hour < 16:
        ed = datetime.today() - timedelta(1)
    else:
        ed = datetime.today()

    tds = trade_days()
    days_to_update = [d for d in tds if st <= d <= ed]

    if len(days_to_update) == 0:
        print('日度行情数据：已经更新到最新数据，无需更新，自动退出')
        return None
    # st = st.strftime("%Y%m%d")
    # ed = ed.strftime("%Y%m%d")

    for d in days_to_update:
        # d = days_to_update[0]
        d_str = d.strftime("%Y%m%d")
        df_tmp = pro.daily(trade_date=d_str)
        df_tmp = df_tmp.set_index('ts_code')

        openprice_pd = insert_to_df(openprice_pd, pd.DataFrame({d: df_tmp['open']}))
        highprice_pd = insert_to_df(highprice_pd, pd.DataFrame({d: df_tmp['high']}))
        lowprice_pd = insert_to_df(lowprice_pd, pd.DataFrame({d: df_tmp['low']}))
        closeprice_pd = insert_to_df(closeprice_pd, pd.DataFrame({d: df_tmp['close']}))
        changePCT_pd = insert_to_df(changePCT_pd, pd.DataFrame({d: df_tmp['pct_chg']}))

        t_volume = insert_to_df(t_volume, pd.DataFrame({d: df_tmp['vol']}) /100 )     # ts里面单位是手，聚源里面是万股
        t_value = insert_to_df(t_value, pd.DataFrame({d: df_tmp['amount']}) /10 )     # ts里面单位是千，聚源里面是万

    openprice_pd.index.name = 'Code'
    highprice_pd.index.name = 'Code'
    lowprice_pd.index.name = 'Code'
    closeprice_pd.index.name = 'Code'
    changePCT_pd.index.name = 'Code'
    t_volume.index.name = 'Code'
    t_value.index.name = 'Code'

    history_file = os.path.join(date_dair, 'download_from_juyuan')
    data.save(openprice_pd, 'OpenPrice_daily.csv', save_path=history_file)
    data.save(highprice_pd, 'HighPrice_daily.csv', save_path=history_file)
    data.save(lowprice_pd, 'ClosePrice_daily.csv', save_path=history_file)
    data.save(closeprice_pd, 'LowPrice_daily.csv', save_path=history_file)
    data.save(changePCT_pd, 'ChangePCT_daily.csv', save_path=history_file)
    data.save(t_volume, 'TurnoverVolume_daily.csv', save_path=history_file)
    data.save(t_value, 'TurnoverValue_daily.csv', save_path=history_file)


# 更新月度价格变动百分比数据
def update_pct_monthly():
    data = Data()
    close_price = data.closeprice_daily
    adj = data.adjfactor
    months_ends = generate_months_ends()
    close_price = adj * close_price
    new_columns = [c for c in close_price.columns if c >= datetime(2006, 1, 1)]
    close_price = close_price[new_columns]

    new_columns = [c for c in close_price.columns if c in months_ends]
    close_me = close_price[new_columns]

    pct_monthly_pd = close_me / close_me.shift(1, axis=1) - 1
    history_file = os.path.join(date_dair, 'download_from_juyuan')
    pct_monthly_pd.to_csv(os.path.join(history_file, 'ChangePCT_monthly.csv'), encoding='gbk')


def update_daily_basic():

    data = Data()
    pb_df = data.pb_daily
    pe_df = data.pe_daily
    turnover_df = data.turnoverrate_daily
    negotiablemv_df = data.negotiablemv_daily  # 流通市值(万元)
    totalmv_df = data.totalmv_daily            # 总市值(万元)

    st = np.min([pb_df.columns[-1], pe_df.columns[-1],
                 turnover_df.columns[-1], negotiablemv_df.columns[-1],
                 totalmv_df.columns[-1]])

    # turnover_rate_f    float    换手率（自由流通股）
    # total_mv    float  总市值 （万元）

    ed = datetime.today() - timedelta(1)

    tds = trade_days()
    days_to_update = [d for d in tds if st <= d <= ed]

    if len(days_to_update) == 0:
        print('日度指标数据：已经更新到最新数据，无需更新，自动退出')
        return None

    for d in days_to_update:
        # d = days_to_update[0]
        d_str = d.strftime("%Y%m%d")
        tmp_df = pro.daily_basic(ts_code='', trade_date=d_str, fields='ts_code,turnover_rate,pe,pb,total_mv,circ_mv')

        tmp_df = tmp_df.set_index('ts_code').sort_index()

        pb_df = insert_to_df(pb_df, pd.DataFrame({d: tmp_df['pb']}))
        pe_df = insert_to_df(pe_df, pd.DataFrame({d: tmp_df['pe']}))
        turnover_df = insert_to_df(turnover_df, pd.DataFrame({d: tmp_df['turnover_rate']}))
        totalmv_df = insert_to_df(totalmv_df, pd.DataFrame({d: tmp_df['total_mv']}))
        negotiablemv_df = insert_to_df(negotiablemv_df, pd.DataFrame({d: tmp_df['circ_mv']}))

        sleep(0.33)

    history_file = os.path.join(date_dair, 'download_from_juyuan')
    data.save(pb_df, 'pb_daily.csv', save_path=history_file)
    data.save(pe_df, 'pe_daily.csv', save_path=history_file)
    data.save(negotiablemv_df, 'NegotiableMV_daily.csv', save_path=history_file)
    data.save(totalmv_df, 'TotalMV_daily.csv', save_path=history_file)
    data.save(turnover_df, 'TurnoverRate_daily.csv', save_path=history_file)

    # compute_month_value()

# pb_monthly和pe_monthly的计算方式为取当月的最后一天
def compute_month_value():
    data = Data()
    basic_path = os.path.join(date_dair, 'download_from_juyuan')
    months_ends = generate_months_ends()

    pb_daily = data.pb_daily
    my_cols = [m for m in months_ends if m in pb_daily.columns]
    pb_monthly = pb_daily[my_cols]
    pb_monthly.to_csv(os.path.join(basic_path, 'pb_monthly.csv'), encoding='gbk')

    pe_daily = data.pe_daily
    my_cols = [m for m in months_ends if m in pe_daily.columns]
    pe_monthly = pe_daily[my_cols]
    pe_monthly.to_csv(os.path.join(basic_path, 'pe_monthly.csv'), encoding='gbk')


def stocks_basis():
    data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,market,list_date')
    return data


# 更新期货的南华指数日线数据
def update_future_price():
    data = Data()
    try:
        nh_future_price_daily = data.nh_future_price_daily.T
        st_date = nh_future_price_daily.index[-1] - timedelta(10)
    except Exception as e:
        nh_future_price_daily = pd.DataFrame()
        st_date = datetime(2010, 1, 1)

    ed_date = datetime.today().strftime("%Y%m%d")

    add_df = pd.DataFrame()
    for key in NH_index_dict.keys():
        df_tmp = pro.index_daily(ts_code=key, start_date=st_date.strftime('%Y%m%d'), end_date=ed_date)
        df_tmp = df_tmp.set_index('trade_date')
        df_tmp.index = pd.to_datetime(df_tmp.index)
        df_tmp.sort_index(inplace=True)
        add_df = pd.concat([add_df, pd.DataFrame({key: df_tmp['close']})], axis=1)

    # nh_future_price_daily
    to_del_index = [i for i in add_df.index if i in nh_future_price_daily.index]
    nh_future_price_daily.drop(to_del_index, axis=0, inplace=True)

    nh_future_price_daily = pd.concat([nh_future_price_daily, add_df], axis=0)
    save_p = r'D:\commodity_datebase\price_data'
    data.save(nh_future_price_daily, 'nh_future_price_daily.csv', save_path=save_p)

def establish_fund_data():
    '''
    问题1：基金不全，场内场外有可能都不全。eg.165515.OF在场内就没有
    问题2：基金经理不只显示初始基金的话，wind-基金经理明细有14159条记录，tushare有21277条记录
    问题3：基金规模等因子，按照时间拼接，缺失严重，考虑改为只取2000行最大记录，但是按照基金进行拼接（最大100只基金每次）
    问题4：单位净值中有重复的索引，所幸的是重复项的值大致相同
    '''
    try:
        os.makedirs(date_dair + './fund' + './download_from_juyuan')
        basic_path = os.path.join(date_dair, 'fund', 'download_from_juyuan')
    except Exception as e:
        basic_path = os.path.join(date_dair, 'fund', 'download_from_juyuan')

    tds=trade_days()

    df1 = pro.fund_basic(market='E')
    df2 = pro.fund_basic(market='O')
    fund_basic = pd.concat([df1, df2])
    fund_basic = fund_basic.set_index('ts_code')

    fund_company = pro.fund_company()
    fund_company = fund_company.set_index('name')

    fund_list = list(fund_basic.index.values)

    fund_manager = pd.DataFrame()
    for i in range(0, len(fund_list), 100):
        seg_fund_list = ''
        for code in fund_list[i:min(i + 100, len(fund_list))]:
            seg_fund_list = seg_fund_list + code + ','
        seg_fund_list = seg_fund_list[:-1]
        manager = pro.fund_manager(ts_code=seg_fund_list)
        fund_manager = pd.concat([fund_manager, manager])

    fund_portfolio = {}
    for code in fund_list:
        portfolio = pro.fund_portfolio(ts_code=code)
        # 每分钟最多访问该接口60次
        sleep(60 / 60)
        fund_portfolio[code] = portfolio

    fund_div = {}
    for code in fund_list:
        div = pro.fund_div(ts_code=code)
        # 每分钟最多访问该接口60次
        sleep(60 / 60)
        fund_div[code] = div

    # 取得贼慢
    fund_share = pd.DataFrame()
    st_date = datetime(2009, 1, 1)
    ed_date = st_date + timedelta(1999)
    while (ed_date <= datetime.today()):
        def gen_dates(b_date, days):
            day = timedelta(days=1)
            for i in range(days):
                yield b_date + day * i

        data = []
        for d in gen_dates(st_date, ((ed_date - st_date).days + 1)):
            if d in tds:
                data.append(d)
        if (ed_date.strftime("%Y%m%d") != datetime.today().strftime("%Y%m%d")):
            add_df = pd.DataFrame()
            for code in fund_list:
                df_tmp = pro.fund_share(ts_code=code, start_date=st_date.strftime('%Y%m%d'),
                                        end_date=ed_date.strftime("%Y%m%d"))
                sleep(60 / 400)
                if len(df_tmp) == 0:
                    colname = df_tmp.columns
                    df_tmp = pd.DataFrame(np.zeros((len(data), 5)))
                    df_tmp.columns = colname
                    df_tmp.index = data
                else:
                    df_tmp = df_tmp.set_index('trade_date')
                    df_tmp.index = pd.to_datetime(df_tmp.index)
                add_df = pd.concat([add_df, pd.DataFrame({code: df_tmp['fd_share']})], axis=1)
            fund_share = pd.concat([fund_share, add_df])
            st_date = ed_date + timedelta(1)
            ed_date = min((st_date + timedelta(1999)), datetime.today())
        else:
            add_df = pd.DataFrame()
            for code in fund_list:
                df_tmp = pro.fund_share(ts_code=code, start_date=st_date.strftime('%Y%m%d'),
                                        end_date=ed_date.strftime("%Y%m%d"))
                sleep(60 / 400)
                if len(df_tmp) == 0:
                    colname = df_tmp.columns
                    df_tmp = pd.DataFrame(np.zeros((len(data), 5)))
                    df_tmp.columns = colname
                    df_tmp.index = data
                else:
                    df_tmp = df_tmp.set_index('trade_date')
                    df_tmp.index = pd.to_datetime(df_tmp.index)
                add_df = pd.concat([add_df, pd.DataFrame({code: df_tmp['fd_share']})], axis=1)
            fund_share = pd.concat([fund_share, add_df])
            break

    fund_adj = pd.DataFrame()
    st_date = datetime(2009, 1, 1)
    ed_date = st_date + timedelta(1999)
    while (ed_date <= datetime.today()):
        def gen_dates(b_date, days):
            day = timedelta(days=1)
            for i in range(days):
                yield b_date + day * i

        data = []
        for d in gen_dates(st_date, ((ed_date - st_date).days + 1)):
            if d in tds:
                data.append(d)
        if (ed_date.strftime("%Y%m%d") != datetime.today().strftime("%Y%m%d")):
            add_df = pd.DataFrame()
            for code in fund_list:
                df_tmp = pro.fund_adj(ts_code=code, start_date=st_date.strftime('%Y%m%d'),
                                      end_date=ed_date.strftime("%Y%m%d"))
                sleep(60 / 400)
                if len(df_tmp) == 0:
                    colname = df_tmp.columns
                    df_tmp = pd.DataFrame(np.zeros((len(data), 3)))
                    df_tmp.columns = colname
                    df_tmp.index = data
                else:
                    df_tmp = df_tmp.set_index('trade_date')
                    df_tmp.index = pd.to_datetime(df_tmp.index)
                add_df = pd.concat([add_df, pd.DataFrame({code: df_tmp['adj_factor']})], axis=1)
            fund_adj = pd.concat([fund_adj, add_df])
            st_date = ed_date + timedelta(1)
            ed_date = min((st_date + timedelta(1999)), datetime.today())
        else:
            add_df = pd.DataFrame()
            for code in fund_list:
                df_tmp = pro.fund_adj(ts_code=code, start_date=st_date.strftime('%Y%m%d'),
                                      end_date=ed_date.strftime("%Y%m%d"))
                sleep(60 / 400)
                if len(df_tmp) == 0:
                    colname = df_tmp.columns
                    df_tmp = pd.DataFrame(np.zeros((len(data), 3)))
                    df_tmp.columns = colname
                    df_tmp.index = data
                else:
                    df_tmp = df_tmp.set_index('trade_date')
                    df_tmp.index = pd.to_datetime(df_tmp.index)
                add_df = pd.concat([add_df, pd.DataFrame({code: df_tmp['adj_factor']})], axis=1)
            fund_adj = pd.concat([fund_adj, add_df])
            break

    unit_nav = pd.DataFrame()
    accum_nav = pd.DataFrame()
    accum_div = pd.DataFrame()
    net_asset = pd.DataFrame()
    total_netasset = pd.DataFrame()
    adj_nav = pd.DataFrame()
    for d in reversed(tds):
        df = pro.fund_nav(end_date=d.strftime('%Y%m%d'))
        if len(df) == 0:
            continue
        else:
            df = df.set_index('ts_code')
            df = df.groupby(df.index).first()
            unit_nav = pd.concat([unit_nav, pd.DataFrame({d.strftime('%Y%m%d'): df['unit_nav']})], axis=1)
            accum_nav = pd.concat([accum_nav, pd.DataFrame({d.strftime('%Y%m%d'): df['accum_nav']})], axis=1)
            accum_div = pd.concat([accum_div, pd.DataFrame({d.strftime('%Y%m%d'): df['accum_div']})], axis=1)
            net_asset = pd.concat([net_asset, pd.DataFrame({d.strftime('%Y%m%d'): df['net_asset']})], axis=1)
            total_netasset = pd.concat([total_netasset, pd.DataFrame({d.strftime('%Y%m%d'): df['total_netasset']})],
                                       axis=1)
            adj_nav = pd.concat([adj_nav, pd.DataFrame({d.strftime('%Y%m%d'): df['adj_nav']})], axis=1)

    fund_open = pd.DataFrame()
    fund_high = pd.DataFrame()
    fund_low = pd.DataFrame()
    fund_close = pd.DataFrame()
    fund_pre_close = pd.DataFrame()
    fund_change = pd.DataFrame()
    fund_pct_change = pd.DataFrame()
    fund_vol = pd.DataFrame()
    fund_amount = pd.DataFrame()
    st_date = datetime(2009, 1, 1)
    ed_date = st_date + timedelta(799)
    while (ed_date <= datetime.today()):
        def gen_dates(b_date, days):
            day = timedelta(days=1)
            for i in range(days):
                yield b_date + day * i

        data = []
        for d in gen_dates(st_date, ((ed_date - st_date).days + 1)):
            if d in tds:
                data.append(d)
        if (ed_date.strftime("%Y%m%d") != datetime.today().strftime("%Y%m%d")):
            add_df_open = pd.DataFrame()
            add_df_high = pd.DataFrame()
            add_df_low = pd.DataFrame()
            add_df_close = pd.DataFrame()
            add_df_pre_close = pd.DataFrame()
            add_df_change = pd.DataFrame()
            add_df_pct_change = pd.DataFrame()
            add_df_vol = pd.DataFrame()
            add_df_amount = pd.DataFrame()

            for code in fund_list:
                df_tmp = pro.fund_daily(ts_code=code, start_date=st_date.strftime('%Y%m%d'),
                                        end_date=ed_date.strftime("%Y%m%d"))
                sleep(60 / 250)
                if len(df_tmp) == 0:
                    colname = df_tmp.columns
                    df_tmp = pd.DataFrame(np.zeros((len(data), 11)))
                    df_tmp.columns = colname
                    df_tmp.index = data
                else:
                    df_tmp = df_tmp.set_index('trade_date')
                    df_tmp.index = pd.to_datetime(df_tmp.index)
                add_df_open = pd.concat([add_df_open, pd.DataFrame({code: df_tmp['open']})], axis=1)
                add_df_high = pd.concat([add_df_high, pd.DataFrame({code: df_tmp['high']})], axis=1)
                add_df_low = pd.concat([add_df_low, pd.DataFrame({code: df_tmp['low']})], axis=1)
                add_df_close = pd.concat([add_df_close, pd.DataFrame({code: df_tmp['close']})], axis=1)
                add_df_pre_close = pd.concat([add_df_pre_close, pd.DataFrame({code: df_tmp['pre_close']})], axis=1)
                add_df_change = pd.concat([add_df_change, pd.DataFrame({code: df_tmp['change']})], axis=1)
                add_df_pct_change = pd.concat([add_df_pct_change, pd.DataFrame({code: df_tmp['pct_chg']})], axis=1)
                add_df_vol = pd.concat([add_df_vol, pd.DataFrame({code: df_tmp['vol']})], axis=1)
                add_df_amount = pd.concat([add_df_amount, pd.DataFrame({code: df_tmp['amount']})], axis=1)
            fund_open = pd.concat([fund_open, add_df_open])
            fund_high = pd.concat([fund_high, add_df_high])
            fund_low = pd.concat([fund_low, add_df_low])
            fund_close = pd.concat([fund_close, add_df_close])
            fund_pre_close = pd.concat([fund_pre_close, add_df_pre_close])
            fund_change = pd.concat([fund_change, add_df_change])
            fund_pct_change = pd.concat([fund_pct_change, add_df_pct_change])
            fund_vol = pd.concat([fund_vol, add_df_vol])
            fund_amount = pd.concat([fund_amount, add_df_amount])

            st_date = ed_date + timedelta(1)
            ed_date = min((st_date + timedelta(799)), datetime.today())
        else:
            add_df_open = pd.DataFrame()
            add_df_high = pd.DataFrame()
            add_df_low = pd.DataFrame()
            add_df_close = pd.DataFrame()
            add_df_pre_close = pd.DataFrame()
            add_df_change = pd.DataFrame()
            add_df_pct_change = pd.DataFrame()
            add_df_vol = pd.DataFrame()
            add_df_amount = pd.DataFrame()
            for code in fund_list:
                df_tmp = pro.fund_daily(ts_code=code, start_date=st_date.strftime('%Y%m%d'),
                                        end_date=ed_date.strftime("%Y%m%d"))
                sleep(60 / 250)
                if len(df_tmp) == 0:
                    colname = df_tmp.columns
                    df_tmp = pd.DataFrame(np.zeros((len(data), 11)))
                    df_tmp.columns = colname
                    df_tmp.index = data
                else:
                    df_tmp = df_tmp.set_index('trade_date')
                    df_tmp.index = pd.to_datetime(df_tmp.index)
                add_df_open = pd.concat([add_df_open, pd.DataFrame({code: df_tmp['open']})], axis=1)
                add_df_high = pd.concat([add_df_high, pd.DataFrame({code: df_tmp['high']})], axis=1)
                add_df_low = pd.concat([add_df_low, pd.DataFrame({code: df_tmp['low']})], axis=1)
                add_df_close = pd.concat([add_df_close, pd.DataFrame({code: df_tmp['close']})], axis=1)
                add_df_pre_close = pd.concat([add_df_pre_close, pd.DataFrame({code: df_tmp['pre_close']})], axis=1)
                add_df_change = pd.concat([add_df_change, pd.DataFrame({code: df_tmp['change']})], axis=1)
                add_df_pct_change = pd.concat([add_df_pct_change, pd.DataFrame({code: df_tmp['pct_chg']})], axis=1)
                add_df_vol = pd.concat([add_df_vol, pd.DataFrame({code: df_tmp['vol']})], axis=1)
                add_df_amount = pd.concat([add_df_amount, pd.DataFrame({code: df_tmp['amount']})], axis=1)
            fund_open = pd.concat([fund_open, add_df_open])
            fund_high = pd.concat([fund_high, add_df_high])
            fund_low = pd.concat([fund_low, add_df_low])
            fund_close = pd.concat([fund_close, add_df_close])
            fund_pre_close = pd.concat([fund_pre_close, add_df_pre_close])
            fund_change = pd.concat([fund_change, add_df_change])
            fund_pct_change = pd.concat([fund_pct_change, add_df_pct_change])
            fund_vol = pd.concat([fund_vol, add_df_vol])
            fund_amount = pd.concat([fund_amount, add_df_amount])
            break

    fund_basic.to_csv(os.path.join(basic_path, 'fund_basic.csv'), encoding='gbk')
    fund_company.to_csv(os.path.join(basic_path, 'fund_company.csv'), encoding='gbk')
    fund_manager.to_csv(os.path.join(basic_path, 'fund_manager.csv'), encoding='gbk')
    fund_share.to_csv(os.path.join(basic_path, 'fund_share.csv'), encoding='gbk')
    fund_adj.to_csv(os.path.join(basic_path, 'fund_adj.csv'), encoding='gbk')
    try:
        os.makedirs(date_dair + './fund' + './download_from_juyuan' + './fund_portfolio')
        for k, v in fund_portfolio.items():
            v.to_csv(os.path.join(basic_path, 'fund_portfolio', k + '.csv'), encoding='gbk')
    except Exception as e:
        for k, v in fund_portfolio.items():
            v.to_csv(os.path.join(basic_path, 'fund_portfolio', k + '.csv'), encoding='gbk')
    try:
        os.makedirs(date_dair + './fund' + './download_from_juyuan' + './fund_div')
        for k, v in fund_div.items():
            v.to_csv(os.path.join(basic_path, 'fund_div', k + '.csv'), encoding='gbk')
    except Exception as e:
        for k, v in fund_portfolio.items():
            v.to_csv(os.path.join(basic_path, 'fund_div', k + '.csv'), encoding='gbk')
    unit_nav.to_csv(os.path.join(basic_path, 'unit_nav.csv'), encoding='gbk')
    accum_nav.to_csv(os.path.join(basic_path, 'accum_nav.csv'), encoding='gbk')
    accum_div.to_csv(os.path.join(basic_path, 'accum_div.csv'), encoding='gbk')
    net_asset.to_csv(os.path.join(basic_path, 'net_asset.csv'), encoding='gbk')
    total_netasset.to_csv(os.path.join(basic_path, 'total_netasset.csv'), encoding='gbk')
    adj_nav.to_csv(os.path.join(basic_path, 'adj_nav.csv'), encoding='gbk')
    fund_open.to_csv(os.path.join(basic_path, 'fund_open.csv'), encoding='gbk')
    fund_high.to_csv(os.path.join(basic_path, 'fund_high.csv'), encoding='gbk')
    fund_low.to_csv(os.path.join(basic_path, 'fund_low.csv'), encoding='gbk')
    fund_close.to_csv(os.path.join(basic_path, 'fund_close.csv'), encoding='gbk')
    fund_pre_close.to_csv(os.path.join(basic_path, 'fund_pre_close.csv'), encoding='gbk')
    fund_change.to_csv(os.path.join(basic_path, 'fund_change.csv'), encoding='gbk')
    fund_pct_change.to_csv(os.path.join(basic_path, 'fund_pct_change.csv'), encoding='gbk')
    fund_vol.to_csv(os.path.join(basic_path, 'fund_vol.csv'), encoding='gbk')
    fund_amount.to_csv(os.path.join(basic_path, 'fund_amount.csv'), encoding='gbk')






if __name__ == '__main__':
    establish_fund_data()
    #compute_month_value()
    #update_futmap()

    #update_adj_factor()

    #update_pct_monthly()

    #update_main_net_buy_amout()
    #get_net_buy_rate()

    #update_adj_factor()
    #trade_days(freq='w')

