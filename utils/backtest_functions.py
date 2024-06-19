import os.path

import numpy as np
import pandas as pd

import itertools
from collections import OrderedDict

from utils.helper_functions import *
from utils.product_info import *

# 因子回测的函数
def get_signal_pnl(file, product, signal_name, thre_mat, reverse=1, tranct=1.1e-4, max_spread=0.61, tranct_ratio=True,
                   flag='train', test_month='202204', HEAD_PATH="order book data/", DATA_PATH="order book data/tmp pkl/", SIGNAL_PATH="order book data/ret/"): # 因子回测的函数
    print(flag)
    data = load(HEAD_PATH+"order flow tick/"+product+"/"+file)
    # data = data[data["good"]].iloc[:500, :] # todo debug
    data = data[data["good"]]
    if flag == 'test':
        S = load(SIGNAL_PATH+test_month+'_'+signal_name+"/"+product+"/"+file)
        data = data.iloc[:len(S), :] # 将原数据和预测的数据大小适配
    else:
        S = load(DATA_PATH+product+"/"+file)
    S = S['ret']
    S.reset_index(drop=True, inplace=True)
    pred = S * reverse
    data.reset_index(drop=True,inplace=True)

    result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"].values), ("close", thre_mat["close"].values),
                               ("num", 0), ("avg.pnl", 0), ("pnl", 0), ("avg.ret", 0), ("ret", 0)]),
                          index=thre_mat.index)
    count = 0
    cur_spread = data["ask"] - data["bid"] # 每个tick的买卖价差
    for thre in thre_mat.iterrows():
        count = count + 1
        buy = pred > thre[1]["open"] # buy的位置
        sell = pred < -thre[1]["open"] # sell的位置

        signal = pd.Series(data=0, index=data.index)
        if not buy.index.equals(signal.index):
            raise ValueError("Index of the boolean Series and the indexed object do not match")
        signal[buy] = 1
        signal[sell] = -1
        scratch = -thre[1]["close"] # 平仓阈值

        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[(signal == 1) & (data["next.ask"] > 0) & (data["next.bid"] > 0) & (cur_spread < max_spread)] = 1
        position_pos[(pred < -scratch) & (data["next.bid"] > 0) & (cur_spread < max_spread)] = 0
        position_pos.ffill(inplace=True)
        pre_pos = position_pos.shift(1)
        notional_position_pos = pd.Series(data=0, index=data.index)
        notional_position_pos[position_pos==1] = 1
        notional_position_pos[(position_pos==1) & (pre_pos==1)] = np.nan
        notional_position_pos[(notional_position_pos==1)] = 1/data["next.ask"][(notional_position_pos==1)] # 多头仓位
        notional_position_pos.ffill(inplace=True)

        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[(signal == -1) & (data["next.ask"] > 0) & (data["next.bid"] > 0) & (cur_spread < max_spread)] = -1
        position_neg[(pred > scratch) & (data["next.ask"] > 0) & (cur_spread < max_spread)] = 0
        position_neg.ffill(inplace=True)
        pre_neg = position_neg.shift(1)
        notional_position_neg = pd.Series(data=0, index=data.index)
        notional_position_neg[position_neg==-1] = -1
        notional_position_neg[(position_neg==-1) & (pre_neg==-1)] = np.nan
        notional_position_neg[(notional_position_neg==-1)] = -1/data["next.bid"][(notional_position_neg==-1)] # 空头仓位
        notional_position_neg.ffill(inplace=True)

        position = position_pos + position_neg # 总仓位方向
        position.fillna(0)
        notional_position = notional_position_pos + notional_position_neg # 总仓位(小数), 仓位信息是在ticks结束位置得到

        position.iloc[0] = 0
        position.iloc[-2:] = 0 # 日内交易, 日内平仓
        notional_position.iloc[0] = 0
        notional_position.iloc[-2:] = 0

        change_pos = position - position.shift(1)
        change_pos.iloc[0] = 0
        change_buy = change_pos > 0
        change_sell = change_pos < 0
        change_base = pd.Series(data=0, index=data.index)

        notional_change_pos = notional_position - notional_position.shift(1)
        notional_change_pos.iloc[0] = 0

        if (tranct_ratio):
            change_base[change_buy] = data["next.ask"][change_buy]*(1+tranct) # 由于信号是在ticks结束发出, 因此用下一条的卖价来成交
            change_base[change_sell] = data["next.bid"][change_sell]*(1-tranct) # 由于信号是在ticks结束发出, 因此用下一条的买价来成交
        else:
            change_base[change_buy] = data["next.ask"][change_buy]+tranct
            change_base[change_sell] = data["next.bid"][change_sell]-tranct

        final_pnl = -sum(change_base * change_pos) # 买入损失资金, 卖出获得资金, 因此前面要加负号 (等手数交易)
        ret = -sum(change_base * notional_change_pos) # 这个才是收益  (等资金交易)
        num = sum((position!=0) & (change_pos!=0)) # 交易的次数
        if num == 0:
            result.loc[thre[0], ("num", "avg.pnl", "pnl", "avg.ret", "ret")] = (0,0,0,0,0)
            return result
        else:
            avg_pnl = np.divide(final_pnl, num)
            avg_ret = np.divide(ret, num)
            result.loc[thre[0], ("num", "avg.pnl", "pnl", "avg.ret", "ret")] = (num, avg_pnl, final_pnl, avg_ret, ret)

    return result

# 统计不同阈值的num, avg.pnl, total.pnl, sharpe, drawdown, max.drawdown, avg.ret, total.ret, sharpe.ret, drawdown.ret, max.drawdown.ret, mar, mar.ret
def get_hft_summary(result, thre_mat):
    all_result = pd.DataFrame(data={"daily.result": result})
    daily_num = all_result['daily.result'].apply(lambda x: x["num"]) # index are dates(file); columns are thresholds
    daily_pnl = all_result['daily.result'].apply(lambda x: x["pnl"])
    daily_ret = all_result['daily.result'].apply(lambda x: x["ret"])
    total_num = daily_num.sum()

    if len(total_num) != len(thre_mat):
        raise selfException("Mismatch!")

    total_pnl = daily_pnl.sum()
    avg_pnl = zero_divide(total_pnl, total_num)
    total_sharp = sharpe(daily_pnl)
    total_drawdown = drawdown(daily_pnl)
    total_max_drawdown = max_drawdown(daily_pnl)

    total_ret = daily_ret.sum()
    avg_ret = zero_divide(total_ret, total_num)
    sharpe_ret = sharpe(daily_ret)
    drawdown_ret = drawdown(daily_ret)
    max_drawdown_ret = max_drawdown(daily_ret)

    final_result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"]), ("close", thre_mat["close"]), ("num", total_num),
                                                 ("avg.pnl", avg_pnl), ("total.pnl", total_pnl), ("sharpe", total_sharp),
                                                 ("drawdown", total_drawdown), ("max.drawdown", total_max_drawdown),
                                                  ("avg.ret", avg_ret), ("total.ret",total_ret), ("sharpe.ret", sharpe_ret),
                                                  ("drawdown.ret", drawdown_ret), ("max.drawdown.ret", max_drawdown_ret),
                                                 ("mar", total_pnl/total_max_drawdown), ("mar.ret", total_ret/max_drawdown_ret)]),
                                index=thre_mat.index) # index are thresholds

    return OrderedDict([("final.result", final_result), ("daily.num", daily_num), ("daily.pnl", daily_pnl), ("daily.ret", daily_ret)])

# 训练集和测试集分开统计
def get_signal_stat(signal_name, thre_mat, product, all_dates_pkl, CORE_NUM, train_sample, test_sample, reverse=1, tranct=1.1e-4,
                    max_spread=0.61, tranct_ratio=True, test_month='202204', HEAD_PATH="order book data/", DATA_PATH="order book data/tmp pkl/", SIGNAL_PATH="order book data/ret/"):
    # 并行计算训练集每日的回测结果, 整合到train_stat
    train_result = parLapply(CORE_NUM, all_dates_pkl[train_sample], get_signal_pnl, product=product,
                             signal_name=signal_name, thre_mat=thre_mat,
                             reverse=reverse, tranct=tranct, max_spread=max_spread, tranct_ratio=tranct_ratio,
                             flag='train',
                             HEAD_PATH=HEAD_PATH, DATA_PATH=DATA_PATH, SIGNAL_PATH=SIGNAL_PATH)
    train_stat = get_hft_summary(train_result, thre_mat)
    # 并行计算测试集每日的回测结果, 整合到test_stat
    test_result = parLapply(CORE_NUM, all_dates_pkl[test_sample], get_signal_pnl, product=product,
                            signal_name=signal_name, thre_mat=thre_mat,
                            reverse=reverse, tranct=tranct, max_spread=max_spread, tranct_ratio=tranct_ratio,
                            flag='test', test_month=test_month,
                            HEAD_PATH=HEAD_PATH, DATA_PATH=DATA_PATH, SIGNAL_PATH=SIGNAL_PATH)
    test_stat = get_hft_summary(test_result, thre_mat)

    return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat)])

# 趋势和反转分开统计
def evaluate_signal(signal_name, all_dates_pkl, product, CORE_NUM, HEAD_PATH="order book data/", DATA_PATH="order book data/tmp pkl/", SIGNAL_PATH="order book data/ret/",
                    train_sample=None, test_sample=None, test_month='202204', save_path="signal result/", reverse=0):
    all_train_signal = pd.DataFrame(columns=[signal_name])
    for date_pkl in all_dates_pkl[train_sample]:
        train_signal = load(DATA_PATH+product+"/"+date_pkl)
        train_signal = pd.DataFrame(index=train_signal.index, columns=[signal_name], data=train_signal[signal_name])
        all_train_signal = pd.concat([all_train_signal, train_signal], axis=0)

    tranct = product_info[product]["tranct"]
    tranct_ratio = product_info[product]["tranct.ratio"]
    spread = product_info[product]["spread"]
    max_spread = spread * 1.1

    open_list = np.quantile(abs(all_train_signal), np.arange(0.99,0.999,0.001))
    thre_list = []
    for cartesian in itertools.product(open_list, np.array([0.2, 0.4, 0.6, 0.8, 1.0])):
        thre_list.append((cartesian[0], -cartesian[0] * cartesian[1]))
    thre_list = np.array(thre_list)
    thre_mat = pd.DataFrame(data=OrderedDict([("open", thre_list[:, 0]), ("close", thre_list[:, 1])]))

    if reverse>=0:
        print("reverse=1")
        trend_signal_stat = get_signal_stat(signal_name, thre_mat, product, all_dates_pkl, CORE_NUM, train_sample, test_sample, reverse=1, tranct=tranct,
                                            max_spread=max_spread, tranct_ratio=tranct_ratio, test_month=test_month,
                                            HEAD_PATH=HEAD_PATH, DATA_PATH=DATA_PATH, SIGNAL_PATH=SIGNAL_PATH)
    if reverse<=0:
        print("reverse=-1")
        reverse_signal_stat = get_signal_stat(signal_name, thre_mat, product, all_dates_pkl, CORE_NUM, train_sample, test_sample, reverse=-1, tranct=tranct,
                                              max_spread=max_spread, tranct_ratio=tranct_ratio, test_month=test_month,
                                              HEAD_PATH=HEAD_PATH, DATA_PATH=DATA_PATH, SIGNAL_PATH=SIGNAL_PATH)

    os.makedirs(HEAD_PATH + save_path, exist_ok=True)
    if reverse==0:
        stat_result = OrderedDict([("trend.signal.stat", trend_signal_stat), ("reverse.signal.stat", reverse_signal_stat)])
        save(stat_result, HEAD_PATH+save_path+product+"."+test_month+'_'+signal_name+".pkl")
    elif reverse==1:
        save(trend_signal_stat, HEAD_PATH+save_path+product+"."+test_month+'_'+signal_name+".trend.pkl")
    elif reverse==-1:
        save(reverse_signal_stat, HEAD_PATH+save_path+product+"."+test_month+'_'+signal_name+".reverse.pkl")



# ----------------------------------------以下是因子回测以及从训练集到测试集的函数-----------------------------------------

def calculate_evaluate_signal(product_list, all_dates_pkl, signal_name, HEAD_PATH="order book data/", DATA_PATH="order book data/tmp pkl/", SIGNAL_PATH="order book data/ret/",
                              train_sample=None, test_sample=None, test_month='202204', save_path="signal result/", reverse=1):
    """计算因子评价指标, 保存到signal result atr, 先分趋势和反转, 再分训练集和测试集"""
    print("calculating evaluation signal")
    for product in product_list:
        print(product)
        evaluate_signal(signal_name=signal_name, all_dates_pkl=all_dates_pkl, product=product, CORE_NUM=12, HEAD_PATH=HEAD_PATH, DATA_PATH=DATA_PATH, SIGNAL_PATH=SIGNAL_PATH,
                        train_sample=train_sample, test_sample=test_sample, test_month=test_month, save_path=save_path, reverse=reverse)

def train_and_test(product_list, format_dates, signal_name, train_sample, test_sample, min_pnl, min_num, test_month='202204', HEAD_PATH="order book data/", save_path="signal result/"):
    """训练集筛选阈值组合用于测试集, min_pnl和min_num是阈值组合回测效果的筛选阈值, 即avg.pnl要超过多少倍的买卖价差, 以及交易次数要超过多少次"""
    print("filter combinations in the training set and test them in the testing set")
    test_all_pnl = np.zeros([sum(test_sample), len(product_list)])
    train_all_pnl = np.zeros([sum(train_sample), len(product_list)])

    i = 0
    chosen_product_list = []
    for product in product_list:
        spread = product_info[product]["spread"]
        signal_stat = load(HEAD_PATH + save_path + product + "." + test_month + '_' + signal_name + ".trend.pkl")
        train_stat = signal_stat["train.stat"]
        # 筛选标准为 平均pnl大于min_pnl倍的买卖价差 且 交易的次数大于min_num
        good_strat = (train_stat["final.result"]["avg.pnl"] > min_pnl * spread) & (train_stat["final.result"]["num"] > min_num)
        if sum(good_strat) < 1:
            continue
        train_pnl = train_stat["daily.ret"].loc[:, good_strat].sum(axis=1) / sum(good_strat)
        # 好的阈值组合在测试集上测试, 取平均
        test_stat = signal_stat["test.stat"]
        test_pnl = test_stat["daily.ret"].loc[:, good_strat].sum(axis=1) / sum(good_strat)
        print(product, "train sharpe ", sharpe(train_pnl), "test sharpe ", sharpe(test_pnl))
        # 各品种分开记录
        train_all_pnl[:, i] = train_pnl
        test_all_pnl[:, i] = test_pnl
        chosen_product_list.append(i)
        i = i + 1

    if len(product_list) > 1:
        best_weight = get_Markowitz_weight(train_pnl=train_all_pnl, chosen_product_list=chosen_product_list)
        best_weight = np.array(best_weight)
        train_portfolio = np.array(np.dot(train_all_pnl, best_weight))
        test_portfolio = np.array(np.dot(test_all_pnl, best_weight))
    else:
        train_portfolio = train_all_pnl[:, 0]
        test_portfolio = test_all_pnl[:, 0]

    # train_portfolio = np.array(np.mean(train_all_pnl, axis=1))  # 各品种取均值, 可以根据马科维兹配置权重
    # test_portfolio = np.array(np.mean(test_all_pnl, axis=1))
    all_portfolio = np.append(train_portfolio, test_portfolio)

    plt.figure(1, figsize=(16, 10))
    plt.title("")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.title("portfolio")
    plt.plot(format_dates[train_sample | test_sample], all_portfolio.cumsum())
    plt.plot(format_dates[test_sample], all_portfolio.cumsum()[len(train_portfolio):])
    plt.show()
    print("train sharpe: ", sharpe(train_portfolio), "test sharpe: ", sharpe(test_portfolio))
