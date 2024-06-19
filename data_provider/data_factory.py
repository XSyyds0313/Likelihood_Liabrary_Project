from data_provider.data_loader import Dataset_task
from torch.utils.data import DataLoader


def train_data_provider(args, flag="train"):
    """
    args: 参数对象
    flag: ['train', 'vali']
    return: data_set: 初始化的Data对象;
            data_loader: DataLoader对象
    """
    Data = Dataset_task
    timeenc = 0 if args.embed != 'timeF' else 1

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
                    data=args.data,
                    vali_set=args.vali_set,
                    train_set=args.train_set)
    print(flag, len(data_set))
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle_flag, # 打乱顺序
                             num_workers=args.num_workers, # 使用多个子进程来加载数据
                             drop_last=drop_last) # 数据集大小不能被batch_size整除时丢弃最后的batch
    return data_set, data_loader


def test_data_provider(args, date="20220601"):
    """
    args: 参数对象
    flag: ['train', 'vali', 'test', 'pred']
    return: data_set: 初始化的Data对象;
            data_loader: DataLoader对象
    """
    Data = Dataset_task
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False
    drop_last = False
    batch_size = 1
    freq = args.freq

    data_set = Data(root_path=args.root_path,
                    data_path=args.data_path,
                    flag='test',
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    timeenc=timeenc,
                    freq=freq,
                    product=args.product,
                    data=args.data,
                    vali_set=args.vali_set,
                    train_set=args.train_set,
                    test_date = date)

    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle_flag, # 打乱顺序
                             num_workers=args.num_workers, # 使用多个子进程来加载数据
                             drop_last=drop_last) # 数据集大小不能被batch_size整除时丢弃最后的batch
    return data_set, data_loader
