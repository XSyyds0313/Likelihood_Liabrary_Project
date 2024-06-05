import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R_square(pred, true):
    """
    计算R-square
    """
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))


def metric(pred, true):
    """"计算误差指标MAE, MSE, RMSE, MAPE, MSPE"""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r_square = R_square(pred, true)

    return mae, mse, rmse, mape, mspe, r_square



def accuracy(pred, true):
    """
    计算准确率
    """
    return np.mean(pred == true)

def precision(pred, true, dire):
    """
    计算精确率
    """
    true_positive = np.sum((pred == dire) & (true == dire))
    false_positive = np.sum((pred == dire) & (true != dire))
    return true_positive / (true_positive + false_positive + 1e-10)

def recall(pred, true, dire):
    """
    计算召回率
    """
    true_positive = np.sum((pred == dire) & (true == dire))
    false_negative = np.sum((pred != dire) & (true == dire))
    return true_positive / (true_positive + false_negative + 1e-10)

def f_score(pred, true, dire):
    """
    计算F-score
    """
    p = precision(pred, true, dire)
    r = recall(pred, true, dire)
    return 2 * (p * r) / (p + r + 1e-10)

def classify_metric(pred, true):
    """"计算误差指标MAE, MSE, RMSE, MAPE, MSPE"""
    total_accuracy = accuracy(pred, true)
    rise_precision = precision(pred, true, 1)
    rise_recall = recall(pred, true, 1)
    rise_f_score = f_score(pred, true, 1)
    down_precision = precision(pred, true, -1)
    down_recall = recall(pred, true, -1)
    down_f_score = f_score(pred, true, -1)
    vibrate_precision = precision(pred, true, 0)
    vibrate_recall = recall(pred, true, 0)
    vibrate_f_score = f_score(pred, true, 0)
    return total_accuracy, rise_precision, rise_recall, rise_f_score, down_precision, down_recall, down_f_score, vibrate_precision, vibrate_recall, vibrate_f_score

