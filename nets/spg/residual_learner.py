import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nets.inpainting.PTransformer import PositionalEncoding


class ResidualLearner(nn.Module):
    def __init__(self, input_dim, is_audio, is_label, n_classes):
        super().__init__()
        self.input_dim = input_dim
        self.is_audio = is_audio
        self.is_label = is_label
        self.condi_dim = is_audio * 768 + is_label * 512

        self.input_proj = nn.Conv1d(self.input_dim, 512, 1)
        self.condi_proj = nn.Conv1d(self.condi_dim, 512, 1)

        self.PE = PositionalEncoding(512, 1800)
        self.class_embed = nn.Embedding(n_classes, 512)

        if self.is_audio or self.is_label:
            decoder_layer = nn.TransformerDecoderLayer(512, 8)
            self.decoder = nn.TransformerDecoder(decoder_layer, 6)
        else:
            decoder_layer = nn.TransformerEncoderLayer(512, 8)
            self.decoder = nn.TransformerEncoder(decoder_layer, 6)
        self.output_conv = nn.Sequential(
            nn.Conv1d(input_dim, 512, 1),
            nn.ReLU(True),
            nn.Conv1d(512, input_dim, 1))

    def forward(self, x, audio=None, label=None):
        assert audio is not None or label is not None, "missing condition"

        # process condition
        if self.is_audio and self.is_label:
            label_embed = self.class_embed(label).unsqueeze(-1)
            label_embed = label_embed.repeat(1, 1, audio.shape[-1])
            condition = torch.cat([audio, label_embed], dim=1)
        elif self.is_audio:
            condition = audio
        elif self.is_label:
            condition = self.class_embed(label).unsqueeze(-1)
        else:
            condition = None

        # get position encoding
        x_pe = self.PE(torch.ones([x.shape[0], 1], dtype=torch.int32, device=x.device) * x.shape[2])
        x = self.input_proj(x)
        x = (x.transpose(1, 2) + x_pe)

        if condition is not None:
            c_pe = self.PE(torch.ones([x.shape[0], 1], dtype=torch.int32, device=x.device) * condition.shape[2])
            condition = self.condi_proj(condition)
            condition = (condition.transpose(1, 2) + c_pe).to(torch.float32).permute(1, 0, 2)

        output = self.decoder(x.to(torch.float32).permute(1, 0, 2), condition)
        output = self.output_conv(output.permute(1, 2, 0))

        return output



def main():
    x = torch.rand([8, 512, 22])
    audio = torch.rand([8, 768, 22*8])
    label = torch.randint(0, 4, [8])
    model = ResidualLearner(512, True, True, 4)

    output = model(x, audio, label)


if __name__ == '__main__':
    main()
