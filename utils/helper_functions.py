import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as ts
import math

from scipy.stats import kurtosis
from scipy.stats import skew

import seaborn as sns

import sklearn
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import cross_validate, KFold

import xgboost as xgb

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss


import _pickle as cPickle
import gzip

import dask
from dask import compute, delayed
import warnings

import functools

from collections import OrderedDict

import statsmodels.formula.api as smf


def load(path):
    with gzip.open(path, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)

def save(data, path):
    serialized = cPickle.dumps(data)
    with gzip.open(path, 'wb', compresslevel=1) as file_object:
        file_object.write(serialized)

def parLapply(CORE_NUM, iterable, func, *args, **kwargs):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute([delayed(f_par)(item) for item in iterable])[0]
        return result

def zero_divide(x, y):
    """returns 0 if the numerator or denominator is 0"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.divide(x, y)
    if hasattr(y, "__len__"):
        res[y == 0] = 0
    elif y == 0:
        if hasattr(x, "__len__"):
            res = np.zeros(len(x))
        else:
            res = 0
    return res

def sharpe(x):
    return zero_divide(np.mean(x) * np.sqrt(250), np.std(x, ddof=1))

def drawdown(x):
    y = np.cumsum(x)
    return np.max(y) - np.max(y[-1:])

def max_drawdown(x):
    y = np.cumsum(x)
    return np.max(np.maximum.accumulate(y) - y)


def get_weight(m, s, m_star):
    """calculate the weight of mean-variance  计算马科维茨均值方差投资组合模型有效边界上各样本点的权重"""
    s_inv = np.linalg.inv(s)
    ones = np.repeat(1, len(m))
    s_inv_ones = np.dot(s_inv, ones)
    s_inv_m = np.dot(s_inv, m)
    A = np.dot(m, s_inv_ones)
    B = np.dot(m, s_inv_m)
    C = np.dot(ones, s_inv_ones)
    D = B * C - A**2
    return ((B - m_star * A) * s_inv_ones + (m_star * C - A) * s_inv_m) / D

def get_Markowitz_weight(train_pnl, chosen_product_list):
    """
    calculate the weight of mean-variance  计算马科维茨均值方差投资组合模型有效边界上最小方差的样本点的权重
    Args:
        train_pnl: pd.dataframe, 训练集各品种的return序列
    Returns:
        best_weight: pd.series, 各品种的马科维兹最优权重
    """
    best_weight = np.zeros(train_pnl.shape[1])
    if chosen_product_list == []:
        return pd.Series(best_weight)
    elif len(chosen_product_list) == 1:
        best_weight[chosen_product_list] = 1
        return pd.Series(best_weight)
    else:
        train_pnl = train_pnl[:, chosen_product_list]
        mean_return = train_pnl.mean(axis=0)  # 各品种pnl序列的均值
        cov_return = np.cov(train_pnl, rowvar=False)  # 各品种间的协方差矩阵
        n_point = 30  # 计算的样本数
        n_strat = train_pnl.shape[1]  # 需配置的品种数
        m_grid = np.linspace(min(mean_return), max(mean_return), n_point)  # 目标均值网格
        cov_grid = np.repeat(np.nan, n_point)  # 每个样本上的方差初始化
        weight_grid = np.zeros((n_strat, n_point))  # 每个样本上的权重初始化
        # here we don't use the first value and last value to avoid scientific computing error  这里我们不使用第一个值和最后一个值来避免科学计算错误
        for i in range(1, n_point - 1):
            w = get_weight(mean_return, cov_return, m_grid[i])  # 利用马科维茨模型求目标均值网格的各品种权重
            cov_grid[i] = np.sqrt(np.dot(np.dot(w, cov_return), w))
            weight_grid[:, i] = w
        best_index = np.nanargmin(cov_grid)  # 最小方差对应的index
        tmp = weight_grid[:, best_index]
        best_weight[chosen_product_list] = tmp
        best_weight = pd.Series(best_weight)
        return best_weight

