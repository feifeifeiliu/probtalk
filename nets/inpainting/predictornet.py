import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from nets.inpainting.PTransformer import PTransformer
from nets.inpainting.gated_pixelcnn_1d import GatedConv, weights_init
from nets.inpainting.gated_pixelcnn_1d import GatedPixelCNN as pixelcnn

class PredictorNet(nn.Module):
    def __init__(self, knn=False, mot_dim=256, sta_dim=2048, dim=64, enc_layers=10, ar_layers=15, n_classes=10,
                 groups=1,
                 identity=False, maskgit=False, maskgit_T=8, transformer=False, text=False, audio=True, motion_context=True):
        super().__init__()
        self.dim = dim
        self.knn = knn
        self.groups = groups
        self.ar_layers = ar_layers
        self.num_code = sta_dim
        self.mask_id = self.num_code
        self.maskgit = maskgit
        self.T = maskgit_T
        self.motion_context = motion_context
        self.is_transformer = transformer
        gatedconv = GatedConv

        # Create embedding layer to embed input
        self.motion_embedding = nn.Conv1d(mot_dim + 1, dim // 4, 1, 1, padding=0)

        # Building the PixelCNN layer by layer
        self.enc = nn.ModuleList()
        d = dim // 4
        for i in range(enc_layers):
            residual = True
            if i == 0:
                kernel, stride, padding = 7, 1, 3
            elif i in [2, 4, 6]:
                kernel, stride, padding = 4, 2, 1
                if i in [4, 6]:
                    d = d * 2
            else:
                kernel, stride, padding = 3, 1, 1

            self.enc.append(
                gatedconv(d, kernel, stride, padding, residual)
            )

        if not motion_context:
            d = 0

        condi_dim = d + 256 * (audio != None) + 256 * (text != None)
        if self.is_transformer:
            self.decoder = PTransformer(groups, 512, condi_dim, 512, 6, self.num_code, n_classes,identity, maskgit)
        else:
            self.decoder = pixelcnn(groups, condi_dim, 512, 10, self.num_code, n_classes, identity, maskgit)
        self.epoch_ratio = 0

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initializes weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, masked_motion, state, mask, label, audio, text, epoch_ratio, input_codes=None, distribution=False):
        """
        masked_motion.shape = (B, Cm, W)
        state.shape = (B, W//8, groups)
        mask.shape = (B, 1, W)
        label.shape = (B, 1)
        audio.shape = (B, Ca, W)
        """
        # B, T, G = state.shape

        # ones = torch.ones_like(masked_motion[:, :, 0]).unsqueeze(dim=2)
        mm = torch.cat([masked_motion, mask], dim=1)
        mm = self.motion_embedding(mm)

        for i, layer in enumerate(self.enc):
            mm = layer(mm)
        # mm.shape = (B, Cmm, W//8)

        condition = mm
        if audio is not None:
            condition = torch.cat([condition, audio], dim=1)
        if text is not None:
            condition = torch.cat([condition, text], dim=1)
        if not self.motion_context:
            condition = condition[:, 512:]

        if self.training:
            logits = self.decoder(state, label, condition, input_codes, epoch_ratio)
        else:
            if distribution:
                return self.decoder.distribution(state, label, condition, input_codes)
            if self.maskgit:
                logits = self.decoder.sample_good(label, condition, input_codes, T=self.T)
            else:
                logits = self.decoder.predict(label, condition, input_codes)

        return logits


if __name__ == '__main__':
    model = PredictorNet(knn=False, mot_dim=100, sta_dim=2048, dim=512, enc_layers=10, ar_layers=15, n_classes=4,
                         groups=4, motion_context=False, text=None)
    masked_motion = torch.randn([8, 100, 88])
    audio = torch.randn([8, 256, 11])
    mask = torch.randint(0, 2, [8, 1, 88])
    state = torch.randint(0, 2048, [8, 11, 4])
    label = torch.randint(0, 4, [8])
    epoch_ratio = 0.5
    condition = torch.randn([8, 256, 11])
    x = model(masked_motion, state, mask, label, audio, None, epoch_ratio)
    print('wait')
