from data_provider.data_loader import Dataset_task12, Dataset_task3, Dataset_task4
from torch.utils.data import DataLoader

data_dict = {
    'task1': Dataset_task12,
    'task2': Dataset_task12,
    'task3': Dataset_task3,
    'task4': Dataset_task4,
}

# 根目录HEAD_PATH
HEAD_PATH = "order book data"
# 数据保存根目录
SAVE_PATH = "order book data"
# 样本的目录
DATA_PATH_1 = HEAD_PATH + "/order book tick/"
DATA_PATH_2 = HEAD_PATH + "/order flow tick/"
TMP_DATA_PATH = HEAD_PATH + "/tmp pkl/"
product_list = ["cu", "zn", "ni", "au", "ag"]
product = "cu"
train_vali_split = "20220501"
vali_test_split = "20220601"

def data_provider_task(args, flag):
    """
    args: 参数对象
    flag: ['train', 'vali', 'test', 'pred']
    return: data_set: 初始化的Data对象;
            data_loader: DataLoader对象
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1


    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    else: # 'train' and 'vali'
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    timeenc=timeenc,
                    freq=freq,
                    product=args.product,
                    train_vali_split=args.train_vali_split,
                    vali_test_split=args.vali_test_split,
                    test_time_list=args.test_time_list,
                    test_day_list=args.test_day_list)
    print(flag, len(data_set))
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle_flag, # 打乱顺序
                             num_workers=args.num_workers, # 使用多个子进程来加载数据
                             drop_last=drop_last) # 数据集大小不能被batch_size整除时丢弃最后的batch
    return data_set, data_loader





def data_provider(args, flag):
    """
    args: 参数对象
    flag: ['train', 'vali', 'test', 'pred']
    return: data_set: 初始化的Data对象;
            data_loader: DataLoader对象
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1


    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.detail_freq
        Data = Dataset_Pred
    else: # 'train' and 'vali'
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    timeenc=timeenc,
                    freq=freq)
    print(flag, len(data_set))
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle_flag, # 打乱顺序
                             num_workers=args.num_workers, # 使用多个子进程来加载数据
                             drop_last=drop_last) # 数据集大小不能被batch_size整除时丢弃最后的batch
    return data_set, data_loader
