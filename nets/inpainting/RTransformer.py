import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from nets.inpainting.PTransformer import DeepSupervisionLayer, PositionalEncoding
# from nets.inpainting.MoFormer_v3 import StylizationBlock

class StylizationBlock(nn.Module):
    def __init__(self, in_dim, groups, dropout=0.1):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, in_dim*groups*2),
        )
        self.norm = nn.LayerNorm(in_dim*groups)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(in_dim*groups, in_dim*groups)),
        )

    def forward(self, input):
        """
        h: B, T, D
        emb: B, D
        """
        h, emb = input['x'], input['l']
        if len(h.shape) == 4:
            B, C, T, G = h.shape
            h = h.permute(0, 1, 3, 2).reshape(B, -1, T)
            # B, 1, 2D
            emb_out = self.emb_layers(emb).unsqueeze(1)
            # scale: B, 1, D / shift: B, 1, D
            scale, shift = torch.chunk(emb_out, 2, dim=2)
            h = self.norm(h.transpose(1, 2)) * (1 + scale) + shift
            h = self.out_layers(h).transpose(1, 2)
            h = h.reshape(B, C, G, T).permute(0, 1, 3, 2)
        elif len(h.shape) == 3:
            B, C, T = h.shape
            # B, 1, 2D
            emb_out = self.emb_layers(emb).unsqueeze(1)
            # scale: B, 1, D / shift: B, 1, D
            scale, shift = torch.chunk(emb_out, 2, dim=2)
            h = self.norm(h.transpose(1, 2)) * (1 + scale) + shift
            h = self.out_layers(h).transpose(1, 2)
        input['x'], input['l'] = h, emb
        return input

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Norm(nn.Module):
    def __init__(self, fn, size):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-12)
        self.fn = fn

    def forward(self, input):
        for fn in self.fn:
            input = fn(input)
        input['x'] = self.norm(input['x'].transpose(1, 2)).transpose(1, 2)
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
        x = x.transpose(1, 2)
        h = self.l1(x)
        h = self.activation(h)
        h = self.l2(h)
        h = h.transpose(1, 2)
        input['x'] = h
        return input



class PositionEmbedding(nn.Module):
    def __init__(self, seq_length, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(seq_length, dim))

    def forward(self, x):
        return x + self.pos_embedding


class ConditionEncoder(nn.Module):
    def __init__(self,
                 input_dim=768,
                 intermediate_dim=512,
                 num_hidden_layers=6,):
        super().__init__()
        input_dim = input_dim
        intermediate_dim = intermediate_dim
        self.win_size = 7
        d_head = 32

        self.temporal_PE = PositionEmbedding(self.win_size, d_head)
        self.input_conv = nn.Conv1d(input_dim, intermediate_dim, 1, 1, 0)

        blocks = []
        blocks.extend([
            Residual(Norm(nn.ModuleList([MLP(intermediate_dim, intermediate_dim, intermediate_dim*2)]), intermediate_dim),
                     True, intermediate_dim)
        ])
        for i in range(num_hidden_layers):
            blocks.extend([
                Residual(Norm(nn.ModuleList(
                    [TemporalAttention(1, intermediate_dim, intermediate_dim, d_head, win_size=self.win_size, dilated=False,
                                       dropout=0.1),]), intermediate_dim)),
                Residual(
                    Norm(nn.ModuleList([MLP(intermediate_dim, intermediate_dim, intermediate_dim*2)]), intermediate_dim))
            ])

        self.net = torch.nn.Sequential(*blocks)
        self.output_conv = nn.Sequential(
            nn.Conv1d(intermediate_dim, intermediate_dim*2, 1),
            nn.ReLU(True),
            nn.Conv1d(intermediate_dim*2, intermediate_dim, 1))
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initializes weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):

        full_embed = self.input_conv(x)
        input = {'x': full_embed,
                 'c': None,
                 'l': None,
                 't_pe': self.temporal_PE,
                 's_pe': None,
                 'casual': False}
        output = self.net(input)
        embed = output['x']
        output = self.output_conv(embed)

        return output


class RTransformer(nn.Module):
    def __init__(self,
                 input_dim=512,
                 condi_dim=512,
                 intermediate_dim=512,
                 num_hidden_layers=6,
                 use_label=True,
                 n_classes=4,
                 motion_context=False):
        super().__init__()
        input_dim = input_dim
        intermediate_dim = intermediate_dim
        self.win_size = 7
        self.motion_context = motion_context
        d_head = 64
        self.class_embed = nn.Embedding(n_classes, intermediate_dim) if use_label else nn.Identity()
        self.PE = PositionalEncoding(intermediate_dim, 1800)
        self.input_conv = nn.Conv1d(input_dim, intermediate_dim, 1, 1, 0)
        if self.motion_context:
            condi_dim = condi_dim + 2
        self.audio_conv = nn.Conv1d(condi_dim, intermediate_dim, 1, 1, 0)
        decoder_layer = DeepSupervisionLayer(d_model=intermediate_dim, nhead=8, groups=1, sb_every_layer=use_label)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_hidden_layers)
        self.output_conv = nn.Sequential(
            nn.Conv1d(intermediate_dim, 1024, 1),
            nn.ReLU(True),
            nn.Conv1d(1024, input_dim, 1))

    def forward(self, x, condition, mask, label):

        x = self.input_conv(x)
        x_pe = self.PE(torch.ones([x.shape[0], 1], dtype=torch.int32, device=x.device) * x.shape[2])
        c_pe = x_pe
        x = (x.transpose(1, 2) + x_pe).to(torch.float32)

        label_embed = self.class_embed(label)
        if self.motion_context:
            mask = F.one_hot(mask.to(torch.long), 2).squeeze(1).transpose(1, 2)
            condition = torch.cat([condition, mask], dim=1)
        condition = self.audio_conv(condition)
        condition = condition.transpose(1, 2)
        condition = (condition + c_pe).to(torch.float32)

        output = self.transformer_decoder(x.permute(1, 0, 2), condition.permute(1, 0, 2),
                                          tgt_key_padding_mask=label_embed)
        output = output.permute(1, 2, 0)

        output = self.output_conv(output)

        return output


class EncoderFormer(nn.Module):
    def __init__(self,
                 input_dim=512,
                 output_dim=512,
                 intermediate_dim=256,
                 num_hidden_layers=6,):
        super().__init__()
        input_dim = input_dim
        intermediate_dim = intermediate_dim
        self.PE = PositionalEncoding(intermediate_dim, 1800)
        self.input_conv = nn.Conv1d(input_dim, intermediate_dim, 1, 1, 0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=intermediate_dim, nhead=8)
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        self.output_conv = nn.Sequential(
            nn.Conv1d(intermediate_dim, 512, 1),
            nn.ReLU(True),
            nn.Conv1d(512, output_dim, 1))

    def forward(self, x):

        x = self.input_conv(x)
        x_pe = self.PE(torch.ones([x.shape[0], 1], dtype=torch.int32, device=x.device) * x.shape[2])
        x = (x.transpose(1, 2) + x_pe).to(torch.float32)

        output = self.transformer_decoder(x.permute(1, 0, 2))
        output = output.permute(1, 2, 0)

        output = self.output_conv(output)

        return output


if __name__ == '__main__':
    model = RTransformer(364, 768, 512, 8)
    x = torch.randn(8, 364, 176)
    condition = torch.randn([8, 768, 176])
    mask = torch.ones(8, 1, 176)
    label = torch.randint(0, 4, [1])
    x = model(x, condition, mask, label)
    # from thop import profile
    # flops, params = profile(model, inputs=(x, condition, mask, label))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

    # decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)
    # transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    # transformer_decoder.eval()
    # x = torch.randn(8, 768, 176)
    # condition = torch.randn([8, 768, 176])
    #
    # flops, params = profile(transformer_decoder, inputs=(x.permute(2, 0, 1), condition.permute(2, 0, 1)))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

    # model = TemporalAttention(1, 512, 512, win_size=7, dilated=False, dropout=0.)
    # temporal_PE = PositionEmbedding(7, 64)
    # input = {'x': x,
    #          'c': condition,
    #          'l': None,
    #          't_pe': temporal_PE,
    #          's_pe': None,
    #          'casual': False}
    # flops, params = profile(model, inputs=(x, condition, temporal_PE))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    #
    # decoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    # flops, params = profile(decoder_layer, inputs=([x.permute(2, 0, 1)]))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')



