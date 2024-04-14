import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.nn import MultiheadAttention, LayerNorm, Dropout
from torch import Tensor
from typing import Optional

from torch.nn.init import xavier_uniform_

_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to("cuda")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock_V3(nn.Module):
    def __init__(self, type, in_dim, groups, dropout=0.1):
        super().__init__()

        self.type = type

        if 'cb' in type:
            self.emb_layers_1 = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_dim, in_dim * groups * 2),
            )
            self.norm_1 = nn.LayerNorm(in_dim * groups)
            self.out_layers_1 = nn.Sequential(
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Linear(in_dim * groups, in_dim * groups)),
            )

        if 'cb_squeeze' in type:
            squeeze = 1
            self.in_proj = nn.Sequential(
                nn.Linear(in_dim*groups, in_dim // squeeze),
            )

            self.emb_layers_cbs = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_dim, in_dim // squeeze * 2),
            )
            self.norm_cbs = nn.LayerNorm(in_dim//squeeze)
            self.out_layers_cbs = nn.Sequential(
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Linear(in_dim//squeeze, in_dim//squeeze)),
            )

            self.out_proj = nn.Sequential(
                zero_module(nn.Linear(in_dim // squeeze, in_dim*groups)),
                nn.LayerNorm(in_dim*groups),
                nn.SiLU(),
            )

        if 'sb' in type:
            self.emb_layers_2 = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_dim, in_dim * 1 * 2),
            )
            self.norm_2 = nn.LayerNorm(in_dim * 1)
            self.out_layers_2 = nn.Sequential(
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Linear(in_dim * 1, in_dim * 1)),
            )

        if 'xb' in type:
            self.emb_layers_4 = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_dim, in_dim * groups * 2),
            )
            self.norm_4 = nn.LayerNorm(in_dim * 1)
            self.out_layers_4 = nn.Sequential(
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Linear(in_dim * 1, in_dim * 1)),
            )

        if 'cb' in type and 'sb' in type:
            self.out_layers_3 = nn.Sequential(
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Linear(in_dim * 2, in_dim * 1)),
            )

    def forward(self, input):
        """
        h: B, T, D
        emb: B, D
        """
        h_0, emb_0 = input['x'], input['l']

        B, C, T, G = h_0.shape

        if 'cb' in self.type:
            if input['maskgit']:
                h_1 = h_0
            else:
                input['pad'] = input['pad'].repeat(B, 1, 1)
                h_1 = h_0.reshape(B, C, -1)
                h_1 = torch.cat([input['pad'], h_1[:, :, :-(G-1)]], dim=-1)
                h_1 = h_1.reshape(B, C, T, G)
            h_1 = h_1.permute(0, 1, 3, 2).reshape(B, -1, T)
            emb_out_1 = self.emb_layers_1(emb_0).unsqueeze(1)
            scale_1, shift_1 = torch.chunk(emb_out_1, 2, dim=2)
            h_1 = self.norm_1(h_1.transpose(1, 2)) * (1 + scale_1) + shift_1
            h_1 = self.out_layers_1(h_1).transpose(1, 2)
            h_1 = h_1.reshape(B, C, G, T).permute(0, 1, 3, 2)

        if 'cb_squeeze' in self.type:
            h_cbs = h_0.permute(0, 1, 3, 2).reshape(B, -1, T)
            h_cbs = self.in_proj(h_cbs.transpose(1, 2))
            emb_out_cbs = self.emb_layers_cbs(emb_0).unsqueeze(1)
            scale_cbs, shift_cbs = torch.chunk(emb_out_cbs, 2, dim=2)
            h_cbs = self.norm_cbs(h_cbs) * (1 + scale_cbs) + shift_cbs
            h_cbs = self.out_layers_cbs(h_cbs)
            h_cbs = self.out_proj(h_cbs).transpose(1, 2)
            h_cbs = h_cbs.reshape(B, C, G, T).permute(0, 1, 3, 2)

        if 'sb' in self.type:
            emb_out_2 = self.emb_layers_2(emb_0).unsqueeze(1)
            scale_2, shift_2 = torch.chunk(emb_out_2, 2, dim=2)
            h_2 = h_0.reshape(B, C, -1)
            h_2 = self.norm_2(h_2.transpose(1, 2)) * (1 + scale_2) + shift_2
            h_2 = self.out_layers_2(h_2).transpose(1, 2)
            h_2 = h_2.reshape(B, C, T, G)

        if 'xb' in self.type:
            emb_out_4 = self.emb_layers_4(emb_0).unsqueeze(1)
            scale_4, shift_4 = torch.chunk(emb_out_4, 2, dim=2)
            h_4 = h_0.permute(0, 2, 3, 1).reshape(B, -1, C)
            h_4 = self.norm_4(h_4)
            h_4 = h_4.reshape(B, T, -1)
            h_4 = h_4 * (1 + scale_4) + shift_4
            h_4 = self.out_layers_4(h_4.reshape(B, T, G, C))
            h_4 = h_4.permute(0, 3, 1, 2)

        if 'cb' in self.type and 'sb' in self.type:
            h = torch.cat([h_1, h_2], dim=1).reshape(B, C*2, -1)
            h = self.out_layers_3(h.transpose(1, 2)).transpose(1, 2).reshape(B, C, T, G)
        elif 'cb' in self.type:
            h = h_1
        elif 'sb' in self.type:
            h = h_2
        elif 'cb_squeeze' in self.type:
            h = h_cbs
        elif 'xb' in self.type:
            h = h_4

        input['x'], input['l'] = h, emb_0
        return input


class Norm(nn.Module):
    def __init__(self, fn, size):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-12)
        self.fn = fn

    def forward(self, input):
        for fn in self.fn:
            input = fn(input)
        input['x'] = self.norm(input['x'].transpose(1, 3)).transpose(1, 3)
        return input


class Residual(nn.Module):
    def __init__(self, fn, cat=False, input_dim=128):
        super().__init__()
        self.fn = fn
        self.cat = cat
        self.input_dim = input_dim

    def forward(self, input):
        x = input['x']
        output = self.fn(input)
        output['x'] = output['x'] + x
        return output


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, input):
        x = input['x']
        B, C, T, G = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        h = self.l1(x)
        h = self.activation(h)
        h = self.l2(h)
        h = h.transpose(1, 2).reshape(B, -1, T, G)
        input['x'] = h
        return input


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len, sine=True):
        """初始化。

        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        if sine:
            # 根据论文给的公式，构造出PE矩阵
            position_encoding = np.array([
                [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
                for pos in range(max_seq_len)])
            # 偶数列使用sin，奇数列使用cos
            position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
            position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

            # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
            # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
            # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
            # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
            pad_row = torch.zeros([1, d_model])
            position_encoding = torch.cat((pad_row, torch.from_numpy(position_encoding)))

            # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
            # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
            self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
            self.position_encoding.weight = nn.Parameter(position_encoding,
                                                         requires_grad=False)
        else:
            self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
            self.position_encoding.weight = nn.Parameter(torch.zeros(max_seq_len + 1, d_model),
                                                         requires_grad=True)

    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len], device=input_len.device)
        return self.position_encoding(input_pos)


def get_attn_subsequent_mask(seq, pad_id=0):
    """ Get an attention mask to avoid using the subsequent info."""
    # assert seq.dim() == 2
    batch_size, max_len, _ = seq.size()
    max_len = max_len
    sub_mask = torch.triu(
        torch.ones(max_len, max_len), diagonal=1
    ).unsqueeze(0).repeat(batch_size, 1, 1).type(torch.ByteTensor)
    if seq.is_cuda:
        sub_mask = sub_mask.cuda()
    return sub_mask


class DeepSupervisionLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, groups, sb_every_layer=True, dropout=0.1, maskgit=False, style_type=['sb']) -> None:
        super().__init__(d_model, nhead, dropout=dropout)
        self.d_model = d_model
        self.groups = groups
        self.sb_every_layer=sb_every_layer
        self.maskgit = maskgit
        self.style_type = style_type

        if self.sb_every_layer:
            self.style_gourp_1 = StylizationBlock_V3(self.style_type, d_model, groups, dropout=0.)
            self.style_gourp_2 = StylizationBlock_V3(self.style_type, d_model, groups, dropout=0.)
            self.style_gourp_3 = StylizationBlock_V3(self.style_type, d_model, groups, dropout=0.)
            if self.groups > 1:
                self.pad_1 = nn.Parameter(torch.zeros([1, d_model, groups - 1]))
                self.pad_2 = nn.Parameter(torch.zeros([1, d_model, groups - 1]))
                self.pad_3 = nn.Parameter(torch.zeros([1, d_model, groups - 1]))
                xavier_uniform_(self.pad_1)
                xavier_uniform_(self.pad_2)
                xavier_uniform_(self.pad_3)
            else:
                self.pad_1 = self.pad_2 = self.pad_3 = 0
        # elif not self.sb_every_layer:
        #     self.style_gourp = StylizationBlock_V3(self.style_type, d_model, groups, dropout=0.1)
        #     if self.groups > 1:
        #         self.pad = nn.Parameter(torch.zeros([1, d_model, groups - 1]))
        #         xavier_uniform_(self.pad)
        #     else:
        #         self.pad = 0

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        condition = memory
        x_pe = memory_mask
        c_pe = memory_key_padding_mask
        label_embed = tgt_key_padding_mask
        x = tgt
        TG, B, C = x.shape

        # x = (x + x_pe).to(torch.float32)
        # condition = (condition + c_pe).to(torch.float32)

        x = self.norm1(x + self._sa_block(x, tgt_mask, None, label_embed=label_embed))
        x = self.norm2(x + self._mha_block(x, condition, None, None, label_embed=label_embed))
        x = self.norm3(x + self._ff_block(x, label_embed))

        # if not self.sb_every_layer:
        #     x = self._style_block(x, label_embed, self.style_gourp, self.pad) + x

        return x

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], label_embed:Tensor) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        if self.sb_every_layer:
            x = self._style_block(x, label_embed, self.style_gourp_1, self.pad_1)
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], label_embed:Tensor) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        if self.sb_every_layer:
            x = self._style_block(x, label_embed, self.style_gourp_2, self.pad_2)

        return self.dropout2(x)

    def _ff_block(self, x: Tensor, label_embed:Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        if self.sb_every_layer:
            x = self._style_block(x, label_embed, self.style_gourp_3, self.pad_3)

        return self.dropout3(x)

    def _style_block(self, x, label_embed, layer, pad):
        x = x.permute(1, 2, 0)
        x = x.reshape(x.shape[0], x.shape[1], -1, self.groups)
        input_style = {'x': x, 'l': label_embed, 'pad': pad, 'maskgit': self.maskgit}
        output = layer(input_style)
        x = output['x'].reshape(output['x'].shape[0], output['x'].shape[1], -1).permute(2, 0, 1)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self,d_model, nhead, dropout=0.1, layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor):
        x = tgt
        x = self.norm(x + self._mha_block(x, memory))
        return x

    def _mha_block(self, x: Tensor, mem: Tensor, ) -> Tensor:
        x = self.multihead_attn(x, mem, mem, need_weights=False)[0]
        return self.dropout(x)


class PTransformer(nn.Module):
    def __init__(self,
                 groups=4,
                 input_dim=512,
                 condi_dim=256,
                 intermediate_dim=512,
                 num_hidden_layers=12,
                 num_code=512,
                 n_classes=4,
                 identity=False,
                 maskgit=False, ):
        super().__init__()
        self.groups = groups
        self.num_code = num_code
        self.mask_id = num_code
        self.identity = identity
        self.maskgit = maskgit
        self.choice_temperature = 4.5
        self.tp_pe = True
        try:
            print('get environ variables: sine =', bool(int(os.environ['sine'])))
            sine = bool(int(os.environ['sine']))
        except:
            print('no environ variables: sine')
            sine = True

        self.tok_emb = nn.Embedding(num_code + 1, intermediate_dim)

        self.class_embed = nn.Embedding(n_classes, intermediate_dim)
        self.condi_emb = nn.Conv1d(condi_dim, intermediate_dim, 1)
        self.PE = PositionalEncoding(intermediate_dim, 1800, sine=sine)
        if self.tp_pe:
            self.product_PE = PositionalEncoding(intermediate_dim, groups, sine=sine)

        # self.motion_block = CrossAttentionBlock(d_model=intermediate_dim, nhead=8, dropout=0.)

        decoder_layer = DeepSupervisionLayer(d_model=intermediate_dim, nhead=8, groups=groups, sb_every_layer=self.identity, dropout=0.1, maskgit=self.maskgit, style_type=['sb'])
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_hidden_layers)

        self.output_conv = nn.Sequential(
            nn.Conv1d(intermediate_dim, 512, 1),
            nn.ReLU(True),
            nn.Conv1d(512, num_code, 1))
        self.gamma = self.gamma_func("linear")
        self.epoch_ratio = 0

        if self.tp_pe and not self.maskgit:
            self.start_emb = nn.Embedding(1, intermediate_dim)

    def forward(self, x, label, condition, input_codes, epoch_ratio):

        label_embed = self.class_embed(label)

        if self.maskgit:
            x = self.random_token(x, 'mask', epoch_ratio)
            x = self.tok_emb(x)
        else:
            # x = self.random_token(x, 'random', 0.15)
            x = self.tok_emb(x)

        if x.dim() == 4:
            b, t, g, d = x.shape
        else:
            b, t, d = x.shape
            g = 1
        x = x.reshape(b, -1, d)

        if self.tp_pe:
            c_pe = x_pe = self.PE(torch.ones([x.shape[0], 1], dtype=torch.int32, device=x.device) * t)
            x_pe = x_pe.repeat_interleave(g, 1)
            product_pe = self.product_PE(torch.ones([x.shape[0], 1], dtype=torch.int32, device=x.device) * g)
            product_pe = product_pe.repeat(1, t, 1)
            x_pe = x_pe + product_pe
        else:
            x_pe = self.PE(torch.ones([x.shape[0], 1], dtype=torch.int32, device=x.device) * x.shape[1])
            c_pe = x_pe[:, :t]

        if not self.maskgit and not self.tp_pe:
            x = torch.cat([torch.zeros([x.shape[0], 1, x.shape[-1]], device=x.device), x[:, :-1]], dim=1)

        x = (x + x_pe).to(torch.float32)

        if not self.maskgit and self.tp_pe:
            x = torch.cat([self.start_emb.weight.reshape(1, 1, -1).repeat(x.shape[0], 1, 1), x[:, :-1]], dim=1)

        if input_codes is not None:
            input_codes = input_codes.reshape(b, -1)
            motion_codes = self.tok_emb(input_codes)
            motion_codes = (motion_codes + x_pe).to(torch.float32)
            motion_feat = (condition[:, :512].transpose(1, 2) + c_pe[:, :t]).to(torch.float32)
            motion_feat = self.motion_block(motion_feat.permute(1, 0, 2), motion_codes.permute(1, 0, 2)).permute(1, 2, 0)
            condition = torch.cat([motion_feat, condition[:, 512:]], 1)

        condition = self.condi_emb(condition)
        condition = condition.transpose(1, 2)
        condition = (condition + c_pe[:, :t]).to(torch.float32)

        if not self.maskgit:
            mask = get_attn_subsequent_mask(x)
        else:
            mask = [None]

        output = self.transformer_decoder(x.permute(1, 0, 2), condition.permute(1, 0, 2), tgt_mask=mask[0],
                                          memory_mask=x_pe.permute(1, 0, 2), memory_key_padding_mask=c_pe.permute(1, 0, 2),
                                          tgt_key_padding_mask=label_embed)
        output = output.permute(1, 2, 0)

        logits = self.output_conv(output).permute(0, 2, 1).contiguous()
        logits = logits.reshape(b, t, g, -1)

        return {"logits": logits}

    def predict(self, label=None, condition=None, input_codes=None):

        Batch, Time, Group = condition.shape[0], condition.shape[2], self.groups

        label_embed = self.class_embed(label)
        token = torch.zeros([Batch, Time * Group], dtype=torch.int64, device=condition.device)

        if self.tp_pe:
            c_pe = x_pe = self.PE(torch.ones([Batch, 1], dtype=torch.int32, device=condition.device) * Time)
            x_pe = x_pe.repeat_interleave(Group, 1)
            product_pe = self.product_PE(torch.ones([Batch, 1], dtype=torch.int32, device=condition.device) * Group)
            product_pe = product_pe.repeat(1, Time, 1)
            x_pe = x_pe + product_pe
        else:
            x_pe = self.PE(torch.ones([condition.shape[0], 1], dtype=torch.int32, device=condition.device) * (Time*Group))
            c_pe = x_pe[:, :Time]

        condition = self.condi_emb(condition)
        condition = condition.transpose(1, 2)
        condition = (condition + c_pe[:, :Time]).to(torch.float32)

        for i in range(token.shape[1]):
            x = self.tok_emb(token)

            # new
            if not self.maskgit and not self.tp_pe:
                x = torch.cat([torch.zeros([x.shape[0], 1, x.shape[-1]], device=x.device), x[:, :-1]], dim=1)

            x = (x + x_pe).to(torch.float32)

            if not self.maskgit and self.tp_pe:
                x = torch.cat([self.start_emb.weight.reshape(1, 1, -1).repeat(x.shape[0], 1, 1), x[:, :-1]], dim=1)

            mask = get_attn_subsequent_mask(x)
            output = self.transformer_decoder(x.permute(1, 0, 2), condition.permute(1, 0, 2), tgt_mask=mask[0],
                                              memory_mask=x_pe.permute(1, 0, 2),
                                              memory_key_padding_mask=c_pe.permute(1, 0, 2),
                                              tgt_key_padding_mask=label_embed)
            output = output.permute(1, 2, 0)

            logits = self.output_conv(output[..., i:(i+1)])

            probs = F.softmax(logits, 1)
            # new
            token.data[:, i].copy_(probs.squeeze().multinomial(1).squeeze().data)

            # if i + 1 < token.shape[1]:
            #     token.data[:, i + 1].copy_(probs.squeeze().multinomial(1).squeeze().data)
        # token.data[:, 0].copy_(probs.squeeze().multinomial(1).squeeze().data)
        # token = torch.cat([token[:, 1:], token[:, :1]], dim=1)

        return token.reshape(Batch, Time, Group)

    def random_token(self, state, type, epoch_ratio):
        B, T, G = state.shape
        state = state.reshape(B, -1)
        r = math.floor(self.gamma(epoch_ratio) * state.shape[1])
        # if epoch_ratio != self.epoch_ratio:
        #     self.epoch_ratio = epoch_ratio
        #     logging.info('now epoch ratio is {}, mask ratio is {}'.format(epoch_ratio, 1 - self.gamma(epoch_ratio)))
        sample = torch.rand(state.shape, device=state.device).topk(r, dim=1).indices
        mask = torch.zeros(state.shape, dtype=torch.bool, device=state.device)
        mask.scatter_(dim=1, index=sample, value=True)
        mask = mask.reshape(B, T, G)
        state = state.reshape(B, T, G)
        if type == 'mask':
            masked_indices = self.mask_id * torch.ones_like(state, device=state.device)
        elif type == 'random':
            masked_indices = torch.randint_like(state, 0, self.num_code, device=state.device)
        a_indices = mask * state + (~mask) * masked_indices
        return a_indices

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    @torch.no_grad()
    def sample_good(self, label=None, condition=None, input_codes=None, T=10, mode="cosine"):
        # self.transformer.eval()
        Batch, Time, Group = condition.shape[0], condition.shape[2], self.groups
        N = Time * Group
        mask_id = torch.tensor(self.mask_id, dtype=torch.int32, device=condition.device)

        inputs = torch.zeros((Batch, N), device="cuda", dtype=torch.int).fill_(mask_id)

        unknown_number_in_the_beginning = torch.sum(inputs == mask_id, dim=-1)
        gamma = self.gamma_func(mode)
        cur_ids = inputs  # [8, 257]
        for t in range(T):
            logits = self.tokens_to_logits(cur_ids.reshape(Batch, Time, Group), label, condition).reshape(Batch, N,
                                                                                                          -1)  # call transformer to get predictions [8, 257, 1024]
            # probs = F.softmax(logits, dim=-1)
            # sampled_ids = probs.squeeze().multinomial(1).squeeze().unsqueeze(0).to(torch.int32)
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample().to(torch.int32)

            unknown_map = (cur_ids == mask_id)  # which tokens need to be sampled -> bool [8, 257]
            # replace all -1 with their samples and leave the others untouched [8, 257]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs [8, 257, 1024]
            # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]
            selected_probs = torch.squeeze(
                torch.take_along_dim(probs, torch.unsqueeze(sampled_ids.to(torch.long), -1), -1), -1)
            # ignore tokens which are already sampled [8, 257]
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)

            # if selected_probs.shape[1] != 88:
            #     selected_probs[:, :16] = selected_probs[:, :16]+1

            # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
            # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            mask_len = torch.maximum(torch.zeros_like(mask_len),
                                     torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) - 1,
                                                   mask_len))
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs,
                                               temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_ids = torch.where(masking, mask_id, sampled_ids)
            # print((cur_ids == 8192).count_nonzero())

        # self.transformer.train()
        return cur_ids.reshape(Batch, Time, Group)

    def sample_max(self, label=None, condition=None, input_codes=None, T=10, mode="cosine"):
        Batch, Time, Group = condition.shape[0], condition.shape[2], self.groups
        N = Time * Group
        mask_id = torch.tensor(self.mask_id, dtype=torch.int32, device=condition.device)
        inputs = torch.zeros((Batch, N), device="cuda", dtype=torch.int).fill_(mask_id)
        cur_ids = inputs  # [8, 257]

        logits = self.tokens_to_logits(cur_ids.reshape(Batch, Time, Group), label, condition).reshape(Batch, N,
                                                                                                      -1)  # call transformer to get predictions [8, 257, 1024]
        cur_ids = F.softmax(logits, dim=-1).max(-1)[1]

        return cur_ids.reshape(Batch, Time, Group)

    def tokens_to_logits(self, x, label, condition):

        x = self.tok_emb(x)
        label_embed = self.class_embed(label)

        if x.dim() == 4:
            b, t, g, d = x.shape
        else:
            b, t, d = x.shape
            g = 1
        x = x.reshape(b, -1, d)

        if self.tp_pe:
            c_pe = x_pe = self.PE(torch.ones([b, 1], dtype=torch.int32, device=condition.device) * t)
            x_pe = x_pe.repeat_interleave(g, 1)
            product_pe = self.product_PE(torch.ones([b, 1], dtype=torch.int32, device=condition.device) * g)
            product_pe = product_pe.repeat(1, t, 1)
            x_pe = x_pe + product_pe
        else:
            x_pe = self.PE(torch.ones([condition.shape[0], 1], dtype=torch.int32, device=condition.device) * (t*g))
            c_pe = x_pe[:, :t]

        x = (x + x_pe).to(torch.float32)

        condition = self.condi_emb(condition)
        condition = condition.transpose(1, 2)
        condition = (condition + c_pe[:, :t]).to(torch.float32)

        mask = [None]
        output = self.transformer_decoder(x.permute(1, 0, 2), condition.permute(1, 0, 2), tgt_mask=mask[0],
                                          memory_mask=x_pe.permute(1, 0, 2), memory_key_padding_mask=c_pe.permute(1, 0, 2),
                                          tgt_key_padding_mask=label_embed)
        output = output.permute(1, 2, 0)
        logits = self.output_conv(output).permute(0, 2, 1).contiguous()

        logits = logits.reshape(b, t, g, -1)

        return logits

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(
            "cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

if __name__ == '__main__':
    model = VanillaTransformer(4, 512, 1024, 512, 6, 2048).to('cuda')
    state = torch.randint(0, 512, [4, 22, 4]).to('cuda')
    label = torch.randint(0, 4, [4]).to('cuda')
    condition = torch.randn([4, 1024, 22]).to('cuda')
    x = model(state, label, condition, state, 0)
