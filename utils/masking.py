import torch
import numpy as np
import math

class TriangularCausalMask():
    """输出[B, 1, L, L]的ones上三角阵"""
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    """输出(B, H, len(index), scores.shape[-1])的ones上三角阵, 最后两维对应Q和K"""
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1]) # (B, H, L, scores.shape[-1])的triu(1)矩阵(最后两维)
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :].to(device) # 根据index对_mask_ex切片
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class LocalMask():
    def __init__(self, B, L,S,device="cpu"):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            self.len = math.ceil(np.log2(L))
            self._mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask2 = ~torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=-self.len).to(device)
            self._mask = self._mask1+self._mask2
    @property
    def mask(self):
        return self._mask