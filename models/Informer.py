import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding
from utils.masking import TriangularCausalMask, ProbMask
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer

class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(AttentionLayer(ProbAttention(mask_flag=False,
                                                          factor=configs.factor,
                                                          attention_dropout=configs.dropout,
                                                          output_attention=configs.output_attention),  # ProbAttention代替Transformer中的FullAttention
                                            configs.d_model,
                                            configs.n_heads),
                             configs.d_model,
                             configs.d_ff,
                             dropout=configs.dropout,
                             activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(configs.d_model) for l in range(configs.e_layers - 1)  # Informer新增了ConvLayer, 卷积层比注意力层少1
            ] if configs.distil else None, # 需要distil参数才用卷积层进行序列依次减半操作
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(AttentionLayer(ProbAttention(mask_flag=True,
                                                          factor=configs.factor,
                                                          attention_dropout=configs.dropout,
                                                          output_attention=False),  # ProbAttention代替Transformer中的FullAttention
                                            configs.d_model,
                                            configs.n_heads),
                             AttentionLayer(ProbAttention(mask_flag=False,
                                                          factor=configs.factor,
                                                          attention_dropout=configs.dropout,
                                                          output_attention=False),  # ProbAttention代替Transformer中的FullAttention
                                            configs.d_model,
                                            configs.n_heads),
                             configs.d_model,
                             configs.d_ff,
                             dropout=configs.dropout,
                             activation=configs.activation,
                )for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
