import os
import sys
import argparse
import torch
import random
import numpy as np

from exp.exp_main import Exp_Main


# 根目录HEAD_PATH
HEAD_PATH = "order book data"
# 数据保存根目录
SAVE_PATH = "order book data"
# 样本的目录
DATA_PATH_1 = HEAD_PATH + "/order book tick/"
DATA_PATH_2 = HEAD_PATH + "/order flow tick/"
# TMP_DATA_PATH = HEAD_PATH + "/tmp pkl/"
TMP_DATA_PATH = HEAD_PATH + "/tmp debug/"
product_list = ["cu", "zn", "ni"]
product = "cu"
train_vali_split = "20220501"
vali_test_split = "20220601"
feature_list = ['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1',
                'BidPrice2', 'BidVolume2', 'AskPrice2', 'AskVolume2',
                'BidPrice3', 'BidVolume3', 'AskPrice3', 'AskVolume3',
                'BidPrice4', 'BidVolume4', 'AskPrice4', 'AskVolume4',
                'BidPrice5', 'BidVolume5', 'AskPrice5', 'AskVolume5']

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Transformer-based models for Time Series Forecasting')

    # basic config
    parser.add_argument('--CORE_NUM', type=int, default=int(os.environ['NUMBER_OF_PROCESSORS']), help='core of your computer')
    parser.add_argument('--is_training', type=int, default=1, help='status') # todo
    parser.add_argument('--model', type=str, default='Transformer',
                        help='model name, options: [Transformer, Informer, Autoformer, FEDformer, ns_Transformer, ns_Informer, ns_Autoformer, iTransformer, iInformer, Crossformer]')
    parser.add_argument('--product', type=str, default='cu', help='product')

    # task
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--train_vali_split', type=str, default='20220105', help='the start day of vali set')
    parser.add_argument('--vali_test_split', type=str, default='20220106', help='the start day of test set')
    parser.add_argument('--test_day_list', type=str, default=['20220601', '20220602'], help='day list used for test in task1 and task2 and task3')

    # task 3
    parser.add_argument('--num_class', type=str, default=3, help='classes of ret')

    # data loader
    parser.add_argument('--data', type=str, default='task2', help='dataset: [task1, task2, task3, task4]')
    parser.add_argument('--root_path', type=str, default=HEAD_PATH, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=TMP_DATA_PATH, help='data file')
    parser.add_argument('--features', type=str, default=feature_list, help='features to predict target')
    parser.add_argument('--target', type=str, default='ret', help='target feature') # todo
    parser.add_argument('--freq', type=str, default='t', # todo  default='h'
                        help='freq for time features encoding, options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--detail_freq', type=str, default='t', help='like freq, but use in predict') # todo  default='h'
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length, when choose Crossformer, set to 0')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=20, help='encoder input size') # 根据特征数量修改 # todo
    parser.add_argument('--dec_in', type=int, default=20, help='decoder input size') # todo
    parser.add_argument('--c_out', type=int, default=20, help='output size') # todo
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=24, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='whether to use distilling in encoder, using this argument means not using distilling')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]') # todo
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', default=True, help='whether to predict unseen future data') # default=False

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times') # default=3 todo
    parser.add_argument('--train_epochs', type=int, default=3, help='train epochs') # default=10
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')


    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # supplementary config for NSTransformer model
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of MLP projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in MLP projector')

    # supplementary config for ITransformer model
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm in iTransformer')

    # supplementary config for Crossformer model
    parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
    parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
    parser.add_argument('--cross_factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')
    parser.add_argument('--baseline', action='store_true', default=False, help='whether to use mean of past series as baseline for prediction')


    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            # setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            setting='{}_{}_{}_sl{}_ll{}_pl{}_el{}_dl{}_df{}'.format(
                args.model,
                args.data,
                args.product,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.e_layers,
                args.d_layers,
                ii)

            exp = Exp(args)  # set experiments 初始化
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_ll{}_pl{}_el{}_dl{}_df{}'.format(
            args.model,
            args.data,
            args.product,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.e_layers,
            args.d_layers,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    print(1)