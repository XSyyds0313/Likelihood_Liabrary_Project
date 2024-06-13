import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
import math


class PositionalEmbedding(nn.Module):
    """位置编码, 输出(1, x.size(1), d_model)"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # pe维度为3, (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)] # pe的第二维度(len)适配x的第二维度(len)

class TokenEmbedding(nn.Module):
    """数据编码, 输出(x.size(0), x.size(1), d_model)"""
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2) # 将x的第三维度features变成d_model
        return x

class FixedEmbedding(nn.Module):
    """传统的Embedding???, 输出(x.size(0), x.size(1), d_model)"""
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    """时段编码, 输入stamp数据, 输出(x_mark.size(0), x_mark.size(1), d_model)"""
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        # 每个时间单位的取值数量
        millisecond_size = 510
        second_size = 60
        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.millisecond_embed = Embed(millisecond_size, d_model)
        self.second_embed = Embed(second_size, d_model)
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        # millisecond_x = self.minute_embed(x[:, :, 6])
        second_x = self.minute_embed(x[:, :, 5])
        minute_x = self.minute_embed(x[:, :, 4])
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # return hour_x + weekday_x + day_x + month_x + minute_x + second_x + millisecond_x # 位置信息加和 todo
        return hour_x + weekday_x + day_x + month_x + minute_x + second_x  # 位置信息加和

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)




class DataEmbedding(nn.Module):
    """根据Token, Positional和Temporal进行DataEmbedding操作, 输出(x.size(0), x.size(1), d_model)"""
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model,
                                                    embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' \
                                  else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_onlypos(nn.Module):
    """根据Token和Positional进行DataEmbedding操作, 无Temporal项, 输出(x.size(0), x.size(1), d_model)"""
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_onlypos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class DataEmbedding_wo_pos(nn.Module):
    """根据Token和Temporal进行DataEmbedding操作, 无Positional项, 输出(x.size(0), x.size(1), d_model)"""
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model,
                                                    embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' \
                                  else TimeFeatureEmbedding(d_model=d_model,
                                                            embed_type=embed_type,
                                                            freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # try:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # except:
        #     a = 1
        return self.dropout(x)

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)

class DSW_embedding(nn.Module):
    """DSW embedding, (batch, ts_len, ts_dim)->(batch, ts_dim, seg_num, d_model)"""
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)  # (batch, ts_len, ts_dim)->(batch, ts_dim, seg_num, seg_len)
        x_embed = self.linear(x_segment)  # (batch, ts_dim, seg_num, seg_len)->(batch, ts_dim, seg_num, d_model)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)  # (batch, ts_dim, seg_num, d_model)->(batch, ts_dim, seg_num, d_model)

        return x_embed