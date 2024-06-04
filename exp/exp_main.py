import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider, data_provider_task
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

from models import FEDformer, Autoformer, Informer, Transformer, Crossformer
from ns_models import ns_Transformer, ns_Informer, ns_Autoformer
from i_models import iTransformer, iInformer

import warnings
warnings.filterwarnings('ignore')


class Task1(nn.Module):
    def __init__(self, configs):
        super(Task1, self).__init__()
        self.fc1 = nn.Linear(configs.c_out, 1)

    def forward(self, output):
        output = self.fc1(output)
        return output

class Task3(nn.Module):
    def __init__(self, configs):
        super(Task3, self).__init__()
        self.pred_len = configs.pred_len
        self.num_classes = configs.num_classes
        self.fc3 = nn.Linear(configs.c_out * configs.pred_len, self.num_classes)

    def forward(self, output):
        batch_size, _, dec_in = output.shape
        output = output.view(batch_size, self.pred_len * dec_in)
        output = self.fc3(output)
        output = torch.softmax(output, dim=1)
        return output

class Task4(nn.Module):
    def __init__(self, configs):
        super(Task4, self).__init__()
        self.pred_len = configs.pred_len
        self.fc4 = nn.Linear(configs.c_out * configs.pred_len, 1)

    def forward(self, output):
        batch_size, _, dec_in = output.shape
        output = output.view(batch_size, self.pred_len * dec_in)
        output = self.fc4(output)
        return output

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        task_set = {"task1":Task1, "task2":Task1, "task3":Task3, "task4":Task4}
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
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """train, test, predict3个阶段时调用, flag is in ['train', 'val', 'test', 'predict'], 获取data_loder"""
        # data_set, data_loader = data_provider(self.args, flag)
        data_set, data_loader = data_provider_task(self.args, flag)
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
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
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
                print("i, iter_count", i, iter_count)
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        pred = self.task_object(outputs)
                        label = label.squeeze()
                        loss = criterion(pred, label)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    pred = self.task_object(outputs)
                    label = label.squeeze()
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
            test_loss = self.vali(test_data, test_loader, criterion) # 验证训练得到的模型在测试集上的损失

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
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
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                pred = self.task_object(outputs).detach().cpu()
                label = label.squeeze().detach().cpu()
                loss = criterion(pred, label)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train() # 将模型设置回评估模式
        return total_loss

    def test(self, setting, test=0):
        """测试函数"""
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/' # 用于保存true和predict的图片
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, label) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                pred = self.task_object(outputs).detach().cpu().numpy()
                label = label.squeeze().detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    pred = test_data.inverse_transform(pred.squeeze(0)).reshape(shape)
                    label = test_data.inverse_transform(label.squeeze(0)).reshape(shape)

                preds.append(pred)
                trues.append(label)

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], label[0, :, -1]), axis=0) # input(seq_len)和true(pre_len)拼接起来
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0) # input(seq_len)和predict(pre_len)拼接起来
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf')) # 可视化并保存

        if self.args.data == "task1":
            pass
        elif self.args.data == "task2":
            pass
        elif self.args.data == "task3":
            pass
        elif self.args.data == "task4":
            pass
        else:
            return None


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/' # 用于保存指标数据
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues) # 计算误差指标
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        """预测函数"""
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader): # pre阶段的batch_size是1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                if pred_data.scale and self.args.inverse:
                    shape = pred.shape
                    pred = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
