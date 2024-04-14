import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from nets.spg.RQ_from_audiocraft import ResidualVectorQuantizer
# from nets.spg.RQ_from_audiocraft import ResidualVectorQuantizer
from nets.spg.residual_quantization import RQBottleneck
from nets.spg.vqvae_modules import VectorQuantizerEMA, ConvNormRelu, Res_CNR_Stack, ProductQuantization, \
    ResidualQuantization, GatedActivation
from nets.spg.wav2vec import Wav2Vec2Model


class TextEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_residual_layers, num_residual_hiddens):
        super(TextEncoder, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.project = ConvNormRelu(in_dim, out_dim, leaky=True)
        self._down_1 = ConvNormRelu(out_dim, out_dim, leaky=True, residual=True, sample='down')
        self._down_2 = ConvNormRelu(out_dim, out_dim, leaky=True, residual=True, sample='down')
        self._down_3 = ConvNormRelu(out_dim, out_dim, leaky=True, residual=True, sample='down')
        self.apply(self._init_weights)

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

    def forward(self, x, frame_num=0):
        h = self.project(x)
        h = self._down_1(h)
        h = self._down_2(h)
        h = self._down_3(h)
        return h


class AudioEncoder_Wav2(nn.Module):
    def __init__(self, in_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(AudioEncoder_Wav2, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.audio_feature_map = nn.Sequential(nn.Conv1d(in_dim, num_hiddens, 1, 1, padding=0),
                                               nn.BatchNorm1d(num_hiddens),
                                               nn.LeakyReLU(0.1))
        self._down_1 = ConvNormRelu(num_hiddens, num_hiddens, leaky=True, residual=True, sample='down')
        self._down_2 = ConvNormRelu(num_hiddens, num_hiddens, leaky=True, residual=True, sample='down')
        self._down_3 = ConvNormRelu(num_hiddens, num_hiddens, leaky=True, residual=True, sample='down')

    def forward(self, x, frame_num=None):
        feature = self.audio_feature_map(x)
        h = self._down_1(feature)
        h = self._down_2(h)
        h = self._down_3(h)
        return h


class ConditionEncoder(nn.Module):
    def __init__(self, in_dim, num_hiddens, num_residual_layers, num_residual_hiddens, dp=0.1):
        super(ConditionEncoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.feature_map = nn.Sequential(nn.Conv1d(in_dim, num_hiddens, 1, 1, padding=0),
                                         nn.BatchNorm1d(num_hiddens),
                                         nn.LeakyReLU(0.1),
                                         nn.Dropout(dp)
                                         )
        self._down_1 = ConvNormRelu(num_hiddens, num_hiddens, leaky=True, residual=True, sample='down', p=dp)
        self._down_2 = ConvNormRelu(num_hiddens, num_hiddens, leaky=True, residual=True, sample='down', p=dp)
        self._down_3 = ConvNormRelu(num_hiddens, num_hiddens, leaky=True, residual=True, sample='down', p=dp)

    def forward(self, x):
        feature = self.feature_map(x)
        h = self._down_1(feature)
        h = self._down_2(h)
        h = self._down_3(h)
        return h


class MotionEncoder(nn.Module):
    def __init__(self, in_dim, num_hiddens, num_residual_layers, num_residual_hiddens, dp=0.1):
        super(MotionEncoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.feature_map = nn.Sequential(nn.Conv1d(in_dim, num_hiddens, 1, 1, padding=0),
                                         GatedActivation(num_hiddens),
                                         nn.LeakyReLU(1),
                                         nn.Dropout(dp)
                                         )
        self._down_1 = ConvNormRelu(num_hiddens, num_hiddens, leaky=True, slope=1., residual=True, sample='down', p=dp, norm='gate')
        self._down_2 = ConvNormRelu(num_hiddens, num_hiddens, leaky=True, slope=1., residual=True, sample='down', p=dp, norm='gate')
        self._down_3 = ConvNormRelu(num_hiddens, num_hiddens, leaky=True, slope=1., residual=True, sample='down', p=dp, norm='gate')

    def forward(self, x):
        feature = self.feature_map(x)
        h = self._down_1(feature)
        h = self._down_2(h)
        h = self._down_3(h)
        return h


class EncoderTopDown(nn.Module):
    def __init__(self, in_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderTopDown, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.project = ConvNormRelu(in_dim, self._num_hiddens, leaky=True)
        self._enc_1 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, leaky_out=True)
        self._enc_2 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, leaky_out=True)
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, leaky_out=True)
        self.apply(self._init_weights)

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

    def forward(self, x, frame_num=0):
        h = self.project(x)
        h = self._enc_1(h)
        h = self._enc_2(h)
        h = self._enc_3(h)
        return h


class AudioEncoder(nn.Module):
    def __init__(self, in_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(AudioEncoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.project = ConvNormRelu(in_dim, self._num_hiddens // 4, leaky=True)

        self._enc_1 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True, leaky_out=True)
        self._down_1 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 4, leaky=True, residual=True,
                                    sample='down')
        self._enc_2 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True, leaky_out=True)
        self._down_2 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 2, leaky=True, residual=True,
                                    sample='down')
        self._enc_3 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True, leaky_out=True)
        self._down_3 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens, leaky=True, residual=True, sample='down')
        self._enc_4 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, leaky_out=True)
        self.apply(self._init_weights)

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

    def forward(self, x, frame_num=0):
        h = self.project(x)
        h = self._enc_1(h)
        h = self._down_1(h)
        h = self._enc_2(h)
        h = self._down_2(h)
        h = self._enc_3(h)
        h = self._down_3(h)
        h = self._enc_4(h)
        return h


class EncoderSC(nn.Module):
    def __init__(self, in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderSC, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        norm = 'bn'

        self.project = ConvNormRelu(in_dim, self._num_hiddens // 4, leaky=True, norm=norm)

        self._enc_1 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True, norm=norm)
        self._down_1 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 4, leaky=True, residual=True,
                                    sample='down', norm=norm)
        self._enc_2 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True, norm=norm)
        self._down_2 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 2, leaky=True, residual=True,
                                    sample='down', norm=norm)
        self._enc_3 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True, norm=norm)
        self._down_3 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens, leaky=True, residual=True, sample='down',
                                    norm=norm)
        self._enc_4 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, norm=norm)

        self.pre_vq_conv = nn.Conv1d(self._num_hiddens, embedding_dim, 1, 1)

    def forward(self, x):
        out = {}
        h = self.project(x)
        h = self._enc_1(h)
        out[1] = h
        h = self._down_1(h)
        h = self._enc_2(h)
        out[2] = h
        h = self._down_2(h)
        h = self._enc_3(h)
        out[3] = h
        h = self._down_3(h)
        h = self._enc_4(h)
        out[4] = h
        h = self.pre_vq_conv(h)
        return h, out


class DecoderSC(nn.Module):
    def __init__(self, out_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, ae=False):
        super(DecoderSC, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.aft_vq_conv = nn.Conv1d(embedding_dim, self._num_hiddens, 1, 1)

        self._dec_1 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)
        self._up_2 = ConvNormRelu(self._num_hiddens, self._num_hiddens // 2, leaky=True, residual=True, sample='up')
        self._dec_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True)
        self._up_3 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens // 4, leaky=True, residual=True,
                                  sample='up')
        self._dec_3 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)
        self._up_4 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 4, leaky=True, residual=True,
                                  sample='up')
        self._dec_4 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)

        self.project = nn.Conv1d(self._num_hiddens // 4, out_dim, 1, 1)

    def forward(self, h, out):
        h = self.aft_vq_conv(h)
        # h = h + out[4]
        h = self._dec_1(h)
        h = self._up_2(h)
        # h = h + out[3]
        h = self._dec_2(h)
        h = self._up_3(h)
        # h = h + out[2]
        h = self._dec_3(h)
        h = self._up_4(h)
        # h = h + out[1]
        h = self._dec_4(h)

        recon = self.project(h)
        return recon


class VQVAE_SC(nn.Module):
    """VQ-VAE"""

    def __init__(self, in_dim, embedding_dim, num_embeddings,
                 num_hiddens, num_residual_layers, num_residual_hiddens,
                 commitment_cost=0.25, decay=0.99, q_type='pro', groups=1, share_code=False):
        super(VQVAE_SC, self).__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.q_type = q_type

        self.encoder = EncoderSC(in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
        # if groups == 1:
        #     self.vq_layer = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        # else:
        if self.q_type == 'pro':
            self.vq_layer = ProductQuantization(embedding_dim, num_embeddings, commitment_cost, decay,
                                                num_chunks=groups, share_code=share_code)
        elif self.q_type == 'res':
            self.vq_layer = ResidualQuantization(embedding_dim, num_embeddings, commitment_cost, decay, num_chunks=groups, share_code=share_code)
        self.decoder = DecoderSC(in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, gt_poses, result_form='pose', r=0., **kwargs):
        z, QuantizationOutput, enc_feats = self.encode(gt_poses, r, **kwargs)
        enc_feats[1] = 0
        enc_feats[2] = 0
        enc_feats[3] = 0
        enc_feats[4] = 0
        e = QuantizationOutput.quantized
        eql_or_lat = QuantizationOutput.loss
        pred_poses = self.decode(e, enc_feats)

        if result_form == 'full':
            return z, QuantizationOutput, pred_poses
        elif result_form == 'part':
            return eql_or_lat, pred_poses
        else:
            return pred_poses

    def encode(self, gt_poses, r=0., **kwargs):
        z, enc_feats = self.encoder(gt_poses)
        QuantizationOutput = self.vq_layer(z, r, **kwargs)
        return z, QuantizationOutput, enc_feats

    def decode(self, e, enc_feats):
        x = self.decoder(e, enc_feats)
        return x
