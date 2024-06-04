import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    """计算单头的注意力模块, 输出(V, None)或(V, attn), 形状分别是(B,L_Q,H,E)和(B,H,L_Q,L_V)"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale # 单头的最后一维
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # 执行爱因斯坦求和约定, 即(l,e)*(s,e)' -> (l,s) 即QK'

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device) # 掩码张量对象

            scores.masked_fill_(attn_mask.mask, -np.inf) # [B,1,L,L]的QK'的上三角均取为-inf

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) # attn
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention: # 是否输出attn
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """计算单头的ProbAttention模块, 输出(context, attn)或(context, None), 形状分别是(B,H,c*ln(L_q),E)和(B,H,L_V,L_V)"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor  # 论文的c, 代表选择特征数量的超参数
        self.scale = scale # 单头的最后一维
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # sample_k: c*ln(L_k); n_top: c*ln(L_q)
        """输出(B,H,c*ln(L_q),L_K)的Q_reduce*K'以及令M最大的n_top个索引"""
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K 先根据K筛选???
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E) # 通过复制扩展到形状为(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q  服从[0, L_K)均匀分布的(L_Q, sample_k)张量
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :] # 形状为(B, H, L_Q, sample_k, E)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze() # 形状为(B, H, L_Q, sample_k)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K) # M(qi,K)
        M_top = M.topk(n_top, sorted=False)[1] # 令M最大的n_top个索引

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone() # 形状为(B, H, L_Q, D) 第三维由复制得到, 值为V.mean(dim=-2)
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """输出注意力计算结果(context_in, attns), 形状分别是(B,H,c*ln(L_q),E)和(B,H,L_V,L_V)"""
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device) # attns是正常的形状
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn # 在index初用attn(切片形状赋值)
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k) 矩阵K的阈值超参数
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q) 矩阵Q的阈值超参数

        U_part = U_part if U_part < L_K else L_K  # c*ln(L_k)和L_K取较小值
        u = u if u < L_Q else L_Q  # c*ln(L_q)和L_Q取较小值

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) # (B,H,c*ln(L_q),L_K)的Q_reduce*K' 以及 令M最大的n_top个索引

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """注意力机制层"""
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads) # 每个head的维度
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # qkv先各自通过线性层并按H分成多个头
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # 再计算注意力机制
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1) # 所有头并起来

        return self.out_projection(out), attn
