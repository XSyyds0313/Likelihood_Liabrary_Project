import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, classify_metric
from utils.helper_functions import save, load

class Task12(nn.Module):
    def __init__(self, configs):
        super(Task12, self).__init__()
        self.args = configs
        self.fc1 = nn.Linear(configs.c_out, 1)

    def forward(self, output):
        output = self.fc1(output)
        return output

    def reshape_pred_label(self, pred, label):
        # input:  pred is (B, L, 1);  label is (B, L, 1)
        # output:  pred is (B, L, 1);  label is (B, L, 1)
        label = label.view(pred.shape)
        return pred, label

    def finish_task(self, date, preds, trues, timestamps, setting, metrics_folder_path):
        mae, mse, rmse, mape, mspe, r_square = metric(preds, trues)  # 计算误差指标
        print('mse:{}, mae:{}, r_square:{}'.format(mse, mae, r_square))
        f = open(os.path.join(metrics_folder_path, date + ".txt"), 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, r_square:{}'.format(mse, mae, r_square))
        f.write('\n')
        f.write('\n')
        f.close()


class Task3(nn.Module):
    def __init__(self, configs):
        super(Task3, self).__init__()
        self.args = configs
        self.pred_len = configs.pred_len
        self.num_classes = configs.num_class
        self.fc3 = nn.Linear(configs.c_out * configs.pred_len, self.num_classes)

    def forward(self, output):
        batch_size, _, dec_in = output.shape
        output = output.view(batch_size, self.pred_len * dec_in)
        output = self.fc3(output)
        output = torch.softmax(output, dim=1)
        return output

    def reshape_pred_label(self, pred, label):
        # input:  pred is (B, 3);  label is (B, L, 1)
        # output:  pred is (B, 1);  label is (B, 1)
        categories = torch.tensor([-1, 0, 1], dtype=torch.float32)
        prob_distribution = torch.softmax(pred, dim=1)
        pred = torch.sum(prob_distribution * categories, dim=1)
        label = label[:, 0, :]
        label = label.view(pred.shape)
        return pred, label

    def finish_task(self, date, preds, trues, timestamps, setting, metrics_folder_path):
        preds, trues = preds.astype(int), trues.astype(int)
        total_accuracy, rise_precision, rise_recall, rise_f_score, down_precision, down_recall, down_f_score, \
            vibrate_precision, vibrate_recall, vibrate_f_score = classify_metric(preds, trues)
        print('total_accuracy:{}'.format(total_accuracy))
        print('rise_precision:{}, rise_recall:{}, rise_f_score:{}'.format(rise_precision, rise_recall, rise_f_score))
        print('down_precision:{}, down_recall:{}, down_f_score:{}'.format(down_precision, down_recall, down_f_score))
        print('vibrate_precision:{}, vibrate_recall:{}, vibrate_f_score:{}'.format(vibrate_precision, vibrate_recall, vibrate_f_score))
        f = open(os.path.join(metrics_folder_path, date + ".txt"), 'a')
        f.write(setting + "  \n")
        f.write('total_accuracy:{}'.format(total_accuracy))
        f.write('rise_precision:{}, rise_recall:{}, rise_f_score:{}'.format(rise_precision, rise_recall, rise_f_score))
        f.write('down_precision:{}, down_recall:{}, down_f_score:{}'.format(down_precision, down_recall, down_f_score))
        f.write('vibrate_precision:{}, vibrate_recall:{}, vibrate_f_score:{}'.format(vibrate_precision, vibrate_recall, vibrate_f_score))
        f.write('\n')
        f.write('\n')
        f.close()


class Task4(nn.Module):
    def __init__(self, configs):
        super(Task4, self).__init__()
        self.args = configs
        self.pred_len = configs.pred_len
        self.fc4 = nn.Linear(configs.c_out * configs.pred_len, 1)

    def forward(self, output):
        batch_size, _, dec_in = output.shape
        output = output.view(batch_size, self.pred_len * dec_in)
        output = self.fc4(output)
        return output

    def reshape_pred_label(self, pred, label):
        # input:  pred is (B, 1);  label is (B, L, 1)
        # output:  pred is (B, 1);  label is (B, 1)
        label = label[:, 0, :]
        label = label.view(pred.shape)
        return pred, label

    def finish_task(self, date, preds, trues, timestamps, setting, metrics_folder_path):
        factor_folder_path = './' + self.args.root_path + '/' + self.args.target + '/'
        if not os.path.exists(factor_folder_path):
            os.makedirs(factor_folder_path)
        if not os.path.exists(factor_folder_path + self.args.product + '/'):
            os.makedirs(factor_folder_path + self.args.product + '/')

        df = pd.DataFrame()
        df["ret"] = preds
        df.index = timestamps
        save(df, factor_folder_path + self.args.product + '/' + date + '.pkl')