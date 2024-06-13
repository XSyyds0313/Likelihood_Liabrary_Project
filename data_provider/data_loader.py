import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.helper_functions import load, save
import pickle
import _pickle as cPickle
import gzip
import warnings

warnings.filterwarnings('ignore')

feature_list = ['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1',
                'BidPrice2', 'BidVolume2', 'AskPrice2', 'AskVolume2',
                'BidPrice3', 'BidVolume3', 'AskPrice3', 'AskVolume3',
                'BidPrice4', 'BidVolume4', 'AskPrice4', 'AskVolume4',
                'BidPrice5', 'BidVolume5', 'AskPrice5', 'AskVolume5']

class Dataset_task12(Dataset):
    def __init__(self, root_path="order book data", flag='train', size=None,
                 features=feature_list, data_path="order book data/tmp pkl/", data = 'task1',
                 target='ret', scale=True, timeenc=0, freq='h', test_date = "20220601",
                 product="cu", train_vali_split="20220501", vali_test_split="20220601"):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['train', 'test', 'vali']
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data = data
        self.product = product
        self.train_vali_split = train_vali_split
        self.vali_test_split = vali_test_split
        self.test_date = test_date  # task1 and task2 and task 3 and task 4
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        all_dates = np.array(os.listdir(self.data_path + self.product))
        vali_start = np.where(all_dates == self.train_vali_split+".pkl")[0][0]
        test_start = np.where(all_dates == self.vali_test_split+".pkl")[0][0]

        train_set = all_dates[0:vali_start]
        vali_set = all_dates[vali_start:test_start]
        test_set = [self.test_date + ".pkl"]
        sets = {'train':train_set, 'vali':vali_set, 'test':test_set}
        chosen_set = sets[self.flag]
        df = pd.DataFrame()
        for day_pkl in chosen_set:
            df_raw = load(self.data_path+self.product+"/"+day_pkl)
            df = pd.concat([df, df_raw], axis=0)

        df_feature = df[self.features]
        scale_folder_path = './fit_scale/' + self.data + '/' + self.product
        # feature标准化
        if self.scale:
            if os.path.isfile(scale_folder_path + '/' + 'scaler.pkl'):
                with open(scale_folder_path + '/' + 'scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                train_data = df_feature  # 取train的部分训练scaler
                self.scaler.fit(train_data.values)
                with open(scale_folder_path + '/' + 'scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            data = self.scaler.transform(df_feature.values)
        else:
            data = df_feature.values

        # 处理时间戳
        df_stamp = pd.DataFrame()
        df_stamp["TimeStamp"] = df["TimeStamp"]
        df_stamp['date'] = pd.to_datetime(df["TimeStamp"])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            df_stamp['microsecond'] = df_stamp.date.apply(lambda row: row.microsecond, 1)
            data_stamp = df_stamp.drop(columns=['date', 'TimeStamp']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq) # todo
            data_stamp = data_stamp.transpose(1, 0)
        # 预处理好的数据x和y以及时间戳
        self.data_x = data
        self.data_y = data
        self.label = np.expand_dims(df[self.target].values, axis=1)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """用于data_loader"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        label_begin = r_begin + self.label_len
        label_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        label = self.label[s_begin:label_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_task3(Dataset):
    def __init__(self, root_path="order book data", flag='train', size=None,
                 features=feature_list, data_path="order book data/tmp pkl/", data = 'task3',
                 target='ret', scale=True, timeenc=0, freq='h', test_date = "20220601",
                 product="cu", train_vali_split="20220501", vali_test_split="20220601"):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['train', 'test', 'vali']
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data = data
        self.product = product
        self.train_vali_split = train_vali_split
        self.vali_test_split = vali_test_split
        self.test_date = test_date  # task1 and task2 and task3 and task 4
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        all_dates = np.array(os.listdir(self.data_path + self.product))
        vali_start = np.where(all_dates == self.train_vali_split+".pkl")[0][0]
        test_start = np.where(all_dates == self.vali_test_split+".pkl")[0][0]
        train_set = all_dates[0:vali_start]
        vali_set = all_dates[vali_start:test_start]
        test_set = [self.test_date+".pkl"]
        sets = {'train':train_set, 'vali':vali_set, 'test':test_set}
        chosen_set = sets[self.flag]
        df = pd.DataFrame()
        for day_pkl in chosen_set:
            df_raw = load(self.data_path+self.product+"/"+day_pkl)
            df = pd.concat([df, df_raw], axis=0)

        df_feature = df[self.features]
        scale_folder_path = './fit_scale/' + self.data + '/' + self.product
        # feature标准化
        if self.scale:
            if os.path.isfile(scale_folder_path + '/' + 'scaler.pkl'):
                with open(scale_folder_path + '/' + 'scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                train_data = df_feature  # 取train的部分训练scaler
                self.scaler.fit(train_data.values)
                with open(scale_folder_path + '/' + 'scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            data = self.scaler.transform(df_feature.values)
        else:
            data = df_feature.values

        # 处理时间戳
        df_stamp = pd.DataFrame()
        df_stamp["TimeStamp"] = df["TimeStamp"]
        df_stamp['date'] = pd.to_datetime(df["TimeStamp"])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            df_stamp['microsecond'] = df_stamp.date.apply(lambda row: row.microsecond, 1)
            data_stamp = df_stamp.drop(columns=['date', 'TimeStamp']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq) # todo
            data_stamp = data_stamp.transpose(1, 0)
        # 预处理好的数据x和y以及时间戳
        self.data_x = data
        self.data_y = data
        self.label = np.expand_dims(df[self.target].values, axis=1)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """用于data_loader"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        label_begin = r_begin + self.label_len
        label_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        label = self.label[s_begin:label_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_task4(Dataset):
    def __init__(self, root_path="order book data", flag='train', size=None,
                 features=feature_list, data_path="order book data/tmp pkl/", data = 'task4',
                 target='ret', scale=True, timeenc=0, freq='h', test_date = "20220601",
                 product="cu", train_vali_split="20220501", vali_test_split="20220601"):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['train', 'test', 'vali']
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data = data
        self.product = product
        self.train_vali_split = train_vali_split
        self.vali_test_split = vali_test_split
        self.test_date = test_date  # task1 and task2 and task3 and task 4
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        all_dates = np.array(os.listdir(self.data_path + self.product))
        vali_start = np.where(all_dates == self.train_vali_split+".pkl")[0][0]
        test_start = np.where(all_dates == self.vali_test_split+".pkl")[0][0]
        train_set = all_dates[0:vali_start]
        vali_set = all_dates[vali_start:test_start]
        test_set = [self.test_date+".pkl"]
        sets = {'train':train_set, 'vali':vali_set, 'test':test_set}
        chosen_set = sets[self.flag]
        df = pd.DataFrame()
        for day_pkl in chosen_set:
            df_raw = load(self.data_path+self.product+"/"+day_pkl)
            df = pd.concat([df, df_raw], axis=0)

        df_feature = df[self.features]
        scale_folder_path = './fit_scale/' + self.data + '/' + self.product
        # feature标准化
        if self.scale:
            if os.path.isfile(scale_folder_path + '/' + 'scaler.pkl'):
                with open(scale_folder_path + '/' + 'scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                train_data = df_feature  # 取train的部分训练scaler
                self.scaler.fit(train_data.values)
                with open(scale_folder_path + '/' + 'scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            data = self.scaler.transform(df_feature.values)
        else:
            data = df_feature.values

        # 处理时间戳
        df_stamp = pd.DataFrame()
        df_stamp["TimeStamp"] = df["TimeStamp"]
        df_stamp['date'] = pd.to_datetime(df["TimeStamp"])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            df_stamp['microsecond'] = df_stamp.date.apply(lambda row: row.microsecond, 1)
            data_stamp = df_stamp.drop(columns=['date', 'TimeStamp']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq) # todo
            data_stamp = data_stamp.transpose(1, 0)
        # 预处理好的数据x和y以及时间戳
        self.data_x = data
        self.data_y = data
        self.label = np.expand_dims(df[self.target].values, axis=1)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """用于data_loader"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        label_begin = r_begin + self.label_len
        label_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        label = self.label[s_begin:label_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
