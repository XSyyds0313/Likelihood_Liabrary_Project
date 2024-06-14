import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import train_data_provider, test_data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, classify_metric
from utils.tasks import Task12, Task3, Task4
from utils.helper_functions import save, load, parLapply

from models import FEDformer, Autoformer, Informer, Transformer, Crossformer, xLSTM
from ns_models import ns_Transformer, ns_Informer, ns_Autoformer
from i_models import iTransformer, iInformer

import warnings
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        task_set = {"task1":Task12, "task2":Task12, "task3":Task3, "task4":Task4}
        self.task_object = task_set[self.args.data](self.args)

    def _build_model(self):
        """类初始化时搭建model架构"""
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'FEDformer': FEDformer,
            'ns_Transformer': ns_Transformer,
            'ns_Informer': ns_Informer,
            'ns_Autoformer': ns_Autoformer,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'Crossformer': Crossformer,
            'xLSTM': xLSTM,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_train_data(self, flag):
        """train, test, predict3个阶段时调用, flag is in ['train', 'val'], 获取data_loder"""
        # data_set, data_loader = data_provider(self.args, flag)
        data_set, data_loader = train_data_provider(self.args, flag)
        return data_set, data_loader

    def _get_test_data(self, date):
        """train, test, predict3个阶段时调用, flag is in ['train', 'val', 'test', 'predict'], 获取data_loder"""
        # data_set, data_loader = data_provider(self.args, flag)
        data_set, data_loader = test_data_provider(self.args, date)
        return data_set, data_loader

    def _select_optimizer(self):
        """根据设定的学习率初始化优化器"""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """损失函数"""
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        """训练函数"""
        scale_folder_path = './fit_scale/' + self.args.data + '/' + self.args.product
        if not os.path.exists(scale_folder_path):
            os.makedirs(scale_folder_path)

        train_data, train_loader = self._get_train_data(flag='train')
        vali_data, vali_loader = self._get_train_data(flag='vali')

        # path = os.path.join(self.args.checkpoints, setting)
        path = self.args.checkpoints + self.args.data + '/' + self.args.product
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) # 提前停止对象初始化

        model_optim = self._select_optimizer() # 优化器
        criterion = self._select_criterion() # 损失函数

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            print("training epoch: ", epoch)
            iter_count = 0
            train_loss = []

            self.model.train() # nn.model.train()将模型设置为训练模式
            epoch_time = time.time()
            # 导出各批次的样本
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(train_loader):
                # 取出的3维数据, 分别是batch_size, seq_len(x) or label_len+pre_len(y), features; mark代表时间编码数据
                iter_count += 1 # 记录每100个中批次的序号
                print("i", i)
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                label = label[:, self.args.seq_len:, :] # label is (B, L, 1)

                # decoder input
                # dec_inp表示将batch_y的第二个维度label_len+pre_len, pre_len部分用0代替
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp: # use automatic mixed precision training
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention: # whether to output attention in ecoder
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                pred = self.task_object(outputs)  # pred is (B, L, 1) or (B, 3) or (B, 1)
                pred, label = self.task_object.reshape_pred_label(pred, label)  # pred and label is (B, L, 1) or (B, 1) or (B, 1)
                pred, label = pred.float(), label.float()
                loss = criterion(pred, label)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0: # 每100个批次重置iter_count并计算剩余时间
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i) # 剩余时间
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward() # 向前求导
                    model_optim.step() # 参数迭代

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion) # 验证训练得到的模型在验证集上的损失

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path) # 根据验证集损失判断是否提前停止并不断保存模型状态字典
            if early_stopping.early_stop: # 若提前停止则跳出epoch循环
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args) # 4种方式(args.lradj)调整优化器optimizer中的学习率lr

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        """验证函数, 在train函数中调用"""
        total_loss = []
        self.model.eval() # nn.model.eval()将模型设置为评估模式
        with torch.no_grad(): # 不求导
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                label = label[:, self.args.seq_len:, :]  # label is (B, L, 1)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :]
                pred = self.task_object(outputs)  # pred is (B, L, 1) or (B, 3) or (B, 1)
                pred, label = self.task_object.reshape_pred_label(pred, label)  # pred and label is (B, L, 1) or (B, 1) or (B, 1)
                pred, label = pred.detach().cpu(), label.detach().cpu()
                loss = criterion(pred, label)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train() # 将模型设置回评估模式
        return total_loss

    def test(self, setting, test=0):
        """测试函数"""
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(self.args.checkpoints + self.args.data + '/' + self.args.product + '/checkpoint.pth'))

        picture_folder_path = './picture_results/' + self.args.data + '/' + self.args.product + '/' # 用于保存true和predict的图片
        if not os.path.exists(picture_folder_path):
            os.makedirs(picture_folder_path)
        # result save
        metrics_folder_path = './metrics_results/' + self.args.data + '/' + self.args.product + '/'  # 用于保存指标数据
        if not os.path.exists(metrics_folder_path):
            os.makedirs(metrics_folder_path)

        self.model.eval()

        def test_each_date(date_pkl):
            test_data, test_loader = self._get_test_data(date=date_pkl[:8])
            timestamps = []
            preds = []
            trues = []
            all_times = load(self.args.data_path+self.args.product+'/'+date_pkl)
            all_times = list(all_times['TimeStamp'])

            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, label_all) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    label = label_all[:, self.args.seq_len:, :]  # label is (1, L, 1)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.pred_len:, :]
                    pred = self.task_object(outputs)  # pred is (1, L, 1) or (1, 3) or (1, 1)
                    pred, label = self.task_object.reshape_pred_label(pred, label)  # pred and label is (1, L, 1) or (1, 1) or (1, 1)
                    pred, label = pred.detach().cpu().numpy(), label.detach().cpu().numpy()

                    if test_data.scale and self.args.inverse:
                        shape = pred.shape
                        pred = test_data.inverse_transform(pred.squeeze(0)).reshape(shape)
                        label = test_data.inverse_transform(label.squeeze(0)).reshape(shape)

                    timestamps.append(all_times[i+self.args.seq_len])
                    preds.append(pred)
                    trues.append(label)

                    if self.args.data == "task1":
                        if i % 4096 == 0:
                            gt = label_all
                            pr = np.concatenate((label_all[:, :self.args.seq_len, :], pred), axis=1)
                            visual(gt, pr, os.path.join(picture_folder_path, date_pkl[:8] + "_" + str(i//4096) + '.pdf'))  # 可视化并保存

            preds = np.array(preds)
            trues = np.array(trues)
            timestamps = np.array(timestamps) # (n)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]) # (n, L, 1) or (n, 1) or (n, 1)
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1]) # (n, L, 1) or (n, 1) or (n, 1)
            print('test shape:', preds.shape, trues.shape)

            task_object = self.task_object
            task_object.finish_task(date_pkl, preds, trues, timestamps, setting, metrics_folder_path)


        if self.args.data == 'task4':
            all_dates = np.array(os.listdir(self.args.data_path + self.args.product))
            test_start = np.where(all_dates == self.args.vali_test_split + ".pkl")[0][0]
            test_day_pkl_list = all_dates[test_start:]
        else:
            test_day_pkl_list = [x+'.pkl' for x in self.args.test_day_list]

        parLapply(self.args.CORE_NUM, test_day_pkl_list, test_each_date)


        return

