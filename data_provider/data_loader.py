import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import _pickle as cPickle
import gzip
import warnings

warnings.filterwarnings('ignore')


def load(path):
    with gzip.open(path, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)

def save(data, path):
    serialized = cPickle.dumps(data)
    with gzip.open(path, 'wb', compresslevel=1) as file_object:
        file_object.write(serialized)

# 根目录HEAD_PATH
HEAD_PATH = "order book data"
# 数据保存根目录
SAVE_PATH = "order book data"
# 样本的目录
DATA_PATH_1 = HEAD_PATH + "/order book tick/"
DATA_PATH_2 = HEAD_PATH + "/order flow tick/"
TMP_DATA_PATH = HEAD_PATH + "/tmp pkl/"
product_list = ["cu", "zn", "ni"]
product = "cu"
train_vali_split = "20220501"
vali_test_split = "20220601"
test_time_list = ['20220601 10:00.0','20220601 11:00.0','20220601 14:00.0','20220601 14:30.0','20220601 22:00.0','20220601 00:30.0']
test_day_list = ['20220601', '20220602', '20220603']

class Dataset_task12(Dataset):
    def __init__(self, root_path=HEAD_PATH, flag='train', size=None,
                 features='S', data_path=TMP_DATA_PATH,
                 target='ret', scale=True, timeenc=0, freq='h',
                 product=product, train_vali_split=train_vali_split, vali_test_split=vali_test_split,
                 test_time_list=test_time_list, test_day_list=test_day_list):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.product = product
        self.train_vali_split = train_vali_split
        self.vali_test_split = vali_test_split
        self.test_time_list = test_time_list # task1 and task2
        self.test_day_list = test_day_list # task 3
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        all_dates = np.array(os.listdir(self.data_path + self.product))
        vali_start = np.where(all_dates == self.train_vali_split+".pkl")
        test_start = np.where(all_dates == self.vali_test_split+".pkl")

        test_day_set = set([x[:8] for x in self.test_time_list])
        train_set = all_dates[0:vali_start]
        vali_set = all_dates[vali_start:test_start]
        test_set = all_dates[test_day_set]
        sets = {'train':train_set, 'vali':vali_set, 'test':test_set}
        chosen_set = sets[self.flag]
        df = pd.DataFrame()
        for day_pkl in chosen_set:
            df_raw = load(TMP_DATA_PATH+self.product+"/"+day_pkl)
            df = pd.concat([df, df_raw], axis=0)

        # 从原始数据取feature(多个feature或单个feature)
        if self.features == 'M' or self.features == 'MS': # many to many or many to one
            cols_data = df.columns[1:] # 0位置是date
            df_data = df[cols_data]
        elif self.features == 'S': # one to one
            df_data = df[[self.target]]

        # feature标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # 取train的部分训练scaler todo
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 处理时间戳
        df_stamp = df["TimeStamp"]
        df_stamp['date'] = pd.to_datetime(df["TimeStamp"])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            df_stamp['microsecond'] = df_stamp.date.apply(lambda row: row.microsecond, 1)
            data_stamp = df_stamp.drop(['date', 'TimeStamp'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq) # todo
            data_stamp = data_stamp.transpose(1, 0)
        # 预处理好的数据x和y以及时间戳
        self.data_x = data
        self.data_y = data
        self.label = data[self.target]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """用于data_loader"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        label_begin = r_begin + self.label_len
        label_end = r_begin + self.label_len + 1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        label = self.label[label_begin:label_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_task3(Dataset):
    def __init__(self, root_path=HEAD_PATH, flag='train', size=None,
                 features='S', data_path=TMP_DATA_PATH,
                 target='ret', scale=True, timeenc=0, freq='h',
                 product=product, train_vali_split=train_vali_split, vali_test_split=vali_test_split,
                 test_time_list=test_time_list, test_day_list=test_day_list):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.product = product
        self.train_vali_split = train_vali_split
        self.vali_test_split = vali_test_split
        self.test_time_list = test_time_list # task1 and task2
        self.test_day_list = test_day_list # task3
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        all_dates = np.array(os.listdir(self.data_path + self.product))
        vali_start = np.where(all_dates == self.train_vali_split+".pkl")
        test_start = np.where(all_dates == self.vali_test_split+".pkl")
        test_days_index = [np.where(all_dates == x+".pkl") for x in self.test_day_list]
        train_set = all_dates[0:vali_start]
        vali_set = all_dates[vali_start:test_start]
        test_set = all_dates[test_days_index]
        sets = {'train':train_set, 'vali':vali_set, 'test':test_set}
        chosen_set = sets[self.flag]
        df = pd.DataFrame()
        for day_pkl in chosen_set:
            df_raw = load(TMP_DATA_PATH+self.product+"/"+day_pkl)
            df = pd.concat([df, df_raw], axis=0)

        # 从原始数据取feature(多个feature或单个feature)
        if self.features == 'M' or self.features == 'MS': # many to many or many to one
            cols_data = df.columns[1:] # 0位置是date
            df_data = df[cols_data]
        elif self.features == 'S': # one to one
            df_data = df[[self.target]]

        # feature标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # 取train的部分训练scaler todo
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 处理时间戳
        df_stamp = df["TimeStamp"]
        df_stamp['date'] = pd.to_datetime(df["TimeStamp"])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            df_stamp['microsecond'] = df_stamp.date.apply(lambda row: row.microsecond, 1)
            data_stamp = df_stamp.drop(['date', 'TimeStamp'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq) # todo
            data_stamp = data_stamp.transpose(1, 0)
        # 预处理好的数据x和y以及时间戳
        self.data_x = data
        self.data_y = data
        self.label = data[self.target]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """用于data_loader"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        label_begin = r_begin + self.label_len
        label_end = r_begin + self.label_len + 1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        label = self.label[label_begin:label_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_task4(Dataset):
    def __init__(self, root_path=HEAD_PATH, flag='train', size=None,
                 features='S', data_path=TMP_DATA_PATH,
                 target='ret', scale=True, timeenc=0, freq='h',
                 product=product, train_vali_split=train_vali_split, vali_test_split=vali_test_split,
                 test_time_list=test_time_list, test_day_list=test_day_list):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.product = product
        self.train_vali_split = train_vali_split
        self.vali_test_split = vali_test_split
        self.test_time_list = test_time_list  # task1 and task2
        self.test_day_list = test_day_list  # task3
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        all_dates = np.array(os.listdir(self.data_path + self.product))
        vali_start = np.where(all_dates == self.train_vali_split+".pkl")
        test_start = np.where(all_dates == self.vali_test_split+".pkl")
        train_set = all_dates[0:vali_start]
        vali_set = all_dates[vali_start:test_start]
        test_set = all_dates[test_start:]
        sets = {'train':train_set, 'vali':vali_set, 'test':test_set}
        chosen_set = sets[self.flag]
        df = pd.DataFrame()
        for day_pkl in chosen_set:
            df_raw = load(TMP_DATA_PATH+self.product+"/"+day_pkl)
            df = pd.concat([df, df_raw], axis=0)

        # 从原始数据取feature(多个feature或单个feature)
        if self.features == 'M' or self.features == 'MS': # many to many or many to one
            cols_data = df.columns[1:] # 0位置是date
            df_data = df[cols_data]
        elif self.features == 'S': # one to one
            df_data = df[[self.target]]

        # feature标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # 取train的部分训练scaler todo
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 处理时间戳
        df_stamp = df["TimeStamp"]
        df_stamp['date'] = pd.to_datetime(df["TimeStamp"])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            df_stamp['microsecond'] = df_stamp.date.apply(lambda row: row.microsecond, 1)
            data_stamp = df_stamp.drop(['date', 'TimeStamp'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq) # todo
            data_stamp = data_stamp.transpose(1, 0)
        # 预处理好的数据x和y以及时间戳
        self.data_x = data
        self.data_y = data
        self.label = data[self.target]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """用于data_loader"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        label_begin = r_begin + self.label_len
        label_end = r_begin + self.label_len + 1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        label = self.label[label_begin:label_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)










class Dataset_ETT_hour(Dataset):

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # 按数量确定train, vali, test
        # 分别对应train, vali, test最前和最后两个个样本的位置
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # 从原始数据取feature(多个feature或单个feature)
        if self.features == 'M' or self.features == 'MS': # many to many or many to one
            cols_data = df_raw.columns[1:] # 0位置是date
            df_data = df_raw[cols_data]
        elif self.features == 'S': # one to one
            df_data = df_raw[[self.target]]
        # feature标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # 取train的部分训练scaler
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # 处理时间戳
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        # 预处理好的数据x和y以及时间戳
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """用于data_loader"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len] # todo
        # border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4] # todo
        border1s = [0, 12 * 24 * 4 - self.seq_len, 12 * 24 * 4 + 4 * 24 * 4 - self.seq_len]
        border2s = [12 * 24 * 4, 12 * 24 * 4 + 4 * 24 * 4, 12 * 24 * 4 + 8 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date') # feature列表, 不含label
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test # 按比例确定train, vali, test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw) # pre阶段利用原数据集最后的seq_len个数据, 预测后面的pred_len个数据

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        # 从tmp_stamp.date.values[-1]开始, 生成periods+1个日期, 间隔为freq; 后面会删除第一个日期, 作为要预测的日期序列
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:]) # input(seq_len)和predict(pre_len)的日期合在一起
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
            z = np.zeros([self.pred_len, seq_y.shape[1]])
            seq_y = np.concatenate([seq_y, z], axis=0)
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
            z = np.zeros([self.pred_len, seq_y.shape[1]])
            seq_y = np.concatenate([seq_y, z], axis=0)
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)