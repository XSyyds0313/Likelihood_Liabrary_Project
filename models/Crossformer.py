import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers.Crossformer_EncDec import Encoder, Decoder
from layers.Crossformer_attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from layers.Embed import DSW_embedding

from math import ceil


class Model(nn.Module):
    """
    Crossformer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.in_len = configs.seq_len
        self.out_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.merge_win = configs.win_size
        self.baseline = configs.baseline
        self.device = configs.devices if configs.devices else torch.device('cuda:0')

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(self.seg_len, configs.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, configs.enc_in,
                                 (self.pad_in_len // self.seg_len), configs.d_model))  # 4维(1,ts_dim,in_seg_num,d_model)正态分布张量参数, 作为position embedding
        self.pre_norm = nn.LayerNorm(configs.d_model)
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, configs.enc_in,
                                 (self.pad_out_len // self.seg_len), configs.d_model))  # 4维(1,ts_dim,out_seg_num,d_model)正态分布张量参数, 作为decoder的原始输入(只有位置编码, 无DSW)

        # Encoder
        self.encoder = Encoder(configs.e_layers, self.merge_win, configs.d_model, configs.n_heads, configs.d_ff,
                               block_depth=1, dropout=configs.dropout, in_seg_num=(self.pad_in_len // self.seg_len), factor=configs.cross_factor)

        # Decoder
        self.decoder = Decoder(self.seg_len, configs.e_layers + 1, configs.d_model, configs.n_heads, configs.d_ff, configs.dropout,
                               out_seg_num=(self.pad_out_len // self.seg_len), factor=configs.cross_factor)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if (self.baseline):
            base = x_enc.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_enc.shape[0]
        if (self.in_len_add != 0):
            x_enc = torch.cat((x_enc[:, :1, :].expand(-1, self.in_len_add, -1), x_enc), dim=1)

        x_enc = self.enc_value_embedding(x_enc)  # DSW_embedding
        x_enc += self.enc_pos_embedding  # position embedding
        x_enc = self.pre_norm(x_enc)

        enc_out = self.encoder(x_enc)  # 每一层的encoder输出
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)  # (b (out_seg_num seg_len) out_d)

        return base + predict_y[:, :self.out_len, :]  # out_len不一定被seg_len整除, 前面取了ceil