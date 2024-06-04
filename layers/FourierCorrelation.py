import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    随机从0至modes中选择modes个index(乱序), 其中modes是modes和seq_len//2较小值
    用于随机选取频域的组分, 输出长度为modes的乱序index
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ########## fourier layer #############
class FourierBlock(nn.Module):
    """FEB-f"""
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain 用于对x_ft进行sampling操作
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels)) # in_channels和out_channels相当于d_models
        # 随即创建4维的[0,1]均匀分布张量 self.weights1是随机初始化的parameterized kernel
        self.weights1 = nn.Parameter(self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        """bmm算法, 用于Fourier neural operations"""
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1) # 把len放在最后一维
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1) # 对最后一维做Fourier变换
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat) # L//2+1 > len(self.index)
        for wi, i in enumerate(self.index): # x_ft通过index进行sampling操作, 再进行自定义的算子操作
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi]) # "(B,H,E),(H,E,E)->(B,H,E)"
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1)) # Fourier逆变换, 将最后一维变为L
        return (x, None)


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    """FEA-f"""
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random', activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method) # 用于decoder, 输入序列q长度为I/2+O
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method) # 来自encoder输出部分, kv长度为I

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        # 随即创建4维的[0,1]均匀分布张量 self.weights1是随机初始化的parameterized kernel
        self.weights1 = nn.Parameter(self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        """bmm算法, 用于Fourier neural operations"""
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1) # 对q进行Fourier变换
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j] # Fourier变换后进行Sampling

        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1) # 对k进行Fourier变换
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j] # Fourier变换后进行Sampling

        # perform attention mechanism on frequency domain
        # QK'
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)) # 自定义算子 (B,H,E,len(index_q)) * (B,H,E,len(index_k)) -> (B,H,len(index_q),len(index_k))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax': # 激活函数
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        # Y = sigma(QK')V
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_) # 自定义算子 (B,H,len(index_q),len(index_k)) * (B,H,E,len(index_v)) -> (B,H,E,len(index_q))

        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1) # (B,H,E,len(index_q)) * (H,E,E,len(index_q)) -> (B,H,E,len(index_q))
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i] # 相当于将(B,H,E,len(index_q))进行Padding到(B, H, E, L//2+1)
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)) # Fourier逆变换
        return (out, None)
    



