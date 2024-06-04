import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """卷积层, 分别进行卷积, 规范化, 激活, 最大池化等操作"""
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # 池化窗口大小为3, 每个2数个取一个窗口, 即序列长度减半

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """编码器层(单层)"""
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model # 前馈全连接层维度
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        ) # 多头自注意力机制
        x = x + self.dropout(new_x) # 残差链接

        y = x = self.norm1(x) # 标准化层
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) # 前馈全连接层1
        y = self.dropout(self.conv2(y).transpose(-1, 1)) # 前馈全连接层2

        return self.norm2(x + y), attn # 残差链接+标准化, 注意力attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers) # 神经网络层堆叠
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None # 卷积层堆叠
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers): # 总共迭代e_layers-1次
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x) # 最后一次没有卷积层
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask) # x依次通过各attn_layer
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x) # 最后规范化

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention # 多头自注意力机制
        self.cross_attention = cross_attention # 多头注意力机制
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0]) # 多头自注意力机制+残差链接
        x = self.norm1(x) # 规范化层

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0]) # 多头注意力机制+残差链接

        y = x = self.norm2(x) # 规范化层
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) # 前馈全连接层1
        y = self.dropout(self.conv2(y).transpose(-1, 1)) # 前馈全连接层2

        return self.norm3(x + y) # 残差链接+标准化


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers) # 神经网络层堆叠
        self.norm = norm_layer
        self.projection = projection # 最后的输出部分, 将d_model变为c_out

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask) # x依次通过各attn_layer

        if self.norm is not None:
            x = self.norm(x) # 最后规范化

        if self.projection is not None:
            x = self.projection(x) # 最后的输出部分
        return x
