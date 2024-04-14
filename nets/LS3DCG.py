'''
not exactly the same as the official repo but the results are good
'''
import sys
import os
import time

from data_utils.lower_body import c_index_3d, c_index_6d

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from nets.base import TrainWrapperBaseClass
from nets.layers import SeqEncoder1D
from losses import KeypointLoss, L1Loss, KLLoss
from data_utils.utils import get_melspec, get_mfcc_psf, get_mfcc_ta
from nets.utils import denormalize
from nets.speech2gesture import ConvNormRelu

""" from https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context.git """


class Deocoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Deocoder, self).__init__()
        self.up1 = nn.Sequential(
            ConvNormRelu(in_ch // 2 + in_ch, in_ch // 2),
            ConvNormRelu(in_ch // 2, in_ch // 2),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.up2 = nn.Sequential(
            ConvNormRelu(in_ch // 4 + in_ch // 2, in_ch // 4),
            ConvNormRelu(in_ch // 4, in_ch // 4),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.up3 = nn.Sequential(
            ConvNormRelu(in_ch // 8 + in_ch // 4, in_ch // 8),
            ConvNormRelu(in_ch // 8, in_ch // 8),
            nn.Conv1d(in_ch // 8, out_ch, 1, 1)
        )

    def forward(self, x, x1, x2, x3):
        x = F.interpolate(x, x3.shape[2])
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)
        x = F.interpolate(x, x2.shape[2])
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = F.interpolate(x, x1.shape[2])
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, n_frames, each_dim):
        super().__init__()
        self.n_frames = n_frames

        self.down1 = nn.Sequential(
            ConvNormRelu(64, 64, '1d', False),
            ConvNormRelu(64, 128, '1d', False),
        )
        self.down2 = nn.Sequential(
            ConvNormRelu(128, 128, '1d', False),
            ConvNormRelu(128, 256, '1d', False),
        )
        self.down3 = nn.Sequential(
            ConvNormRelu(256, 256, '1d', False),
            ConvNormRelu(256, 512, '1d', False),
        )
        self.down4 = nn.Sequential(
            ConvNormRelu(512, 512, '1d', False),
            ConvNormRelu(512, 1024, '1d', False),
        )

        self.down = nn.MaxPool1d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.face_decoder = Deocoder(1024, each_dim[0] + each_dim[3])
        self.body_decoder = Deocoder(1024, each_dim[1])
        self.hand_decoder = Deocoder(1024, each_dim[2])

    def forward(self, spectrogram, time_steps=None):
        if time_steps is None:
            time_steps = self.n_frames

        x1 = self.down1(spectrogram)
        x = self.down(x1)
        x2 = self.down2(x)
        x = self.down(x2)
        x3 = self.down3(x)
        x = self.down(x3)
        x = self.down4(x)
        x = self.up(x)

        face = self.face_decoder(x, x1, x2, x3)
        body = self.body_decoder(x, x1, x2, x3)
        hand = self.hand_decoder(x, x1, x2, x3)

        return face, body, hand


class Generator(nn.Module):
    def __init__(self,
                 each_dim,
                 training=False,
                 device=None
                 ):
        super().__init__()

        self.training = training
        self.device = device

        self.encoderdecoder = EncoderDecoder(15, each_dim)

    def forward(self, in_spec, time_steps=None):
        if time_steps is not None:
            self.gen_length = time_steps

        face, body, hand = self.encoderdecoder(in_spec)
        out = torch.cat([face, body, hand], dim=1)
        out = out.transpose(1, 2)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormRelu(input_dim, 128, '1d'),
            ConvNormRelu(128, 256, '1d'),
            nn.MaxPool1d(kernel_size=2),
            ConvNormRelu(256, 256, '1d'),
            ConvNormRelu(256, 512, '1d'),
            nn.MaxPool1d(kernel_size=2),
            ConvNormRelu(512, 512, '1d'),
            ConvNormRelu(512, 1024, '1d'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1024, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        out = self.net(x)
        return out


class TrainWrapper(TrainWrapperBaseClass):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0
        self.convert_to_6d = self.config.Data.pose.convert_to_6d
        self.expression = self.config.Data.pose.expression
        self.type = 'fbhe'
        self.init_params()

        self.generator = Generator(
            each_dim=self.each_dim,
            training=not self.args.infer,
            device=self.device,
        ).to(self.device)
        self.discriminator = Discriminator(
            input_dim=self.each_dim[1] + self.each_dim[2] + 64
        ).to(self.device)
        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d
        self.MSELoss = KeypointLoss().to(self.device)
        self.L1Loss = L1Loss().to(self.device)
        super().__init__(args, config)

    def __call__(self, bat):
        assert (not self.args.infer), "infer mode"
        self.global_step += 1

        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        expression = bat['expression'].to(self.device).to(torch.float32)
        jaw = poses[:, :self.each_dim[0], :]
        poses = poses[:, self.c_index, :]

        pred = self.generator(in_spec=aud)

        D_loss, D_loss_dict = self.get_loss(
            pred_poses=pred.detach(),
            gt_poses=poses,
            aud=aud,
            mode='training_D',
        )

        self.discriminator_optimizer.zero_grad()
        D_loss.backward()
        self.discriminator_optimizer.step()

        G_loss, G_loss_dict = self.get_loss(
            pred_poses=pred,
            gt_poses=poses,
            aud=aud,
            expression=expression,
            jaw=jaw,
            mode='training_G',
        )
        self.generator_optimizer.zero_grad()
        G_loss.backward()
        self.generator_optimizer.step()

        total_loss = None
        loss_dict = {}
        for key in list(D_loss_dict.keys()) + list(G_loss_dict.keys()):
            loss_dict[key] = G_loss_dict.get(key, 0) + D_loss_dict.get(key, 0)

        return total_loss, loss_dict

    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 aud=None,
                 jaw=None,
                 expression=None,
                 mode='training_G',
                 ):
        loss_dict = {}
        aud = aud.transpose(1, 2)
        gt_poses = gt_poses.transpose(1, 2)
        gt_aud = torch.cat([gt_poses, aud], dim=2)
        pred_aud = torch.cat([pred_poses[:, :, (self.each_dim[0]+self.each_dim[3]):], aud], dim=2)

        if mode == 'training_D':
            dis_real = self.discriminator(gt_aud)
            dis_fake = self.discriminator(pred_aud)
            dis_error = self.MSELoss(torch.ones_like(dis_real).to(self.device), dis_real) + self.MSELoss(
                torch.zeros_like(dis_fake).to(self.device), dis_fake)
            loss_dict['dis'] = dis_error

            return dis_error, loss_dict
        elif mode == 'training_G':
            jaw_loss = self.L1Loss(pred_poses[:, :, :self.each_dim[0]], jaw.transpose(1, 2))
            face_loss = self.MSELoss(pred_poses[:, :, self.each_dim[0]:(self.each_dim[0]+self.each_dim[3])], expression.transpose(1, 2))
            body_loss = self.L1Loss(pred_poses[:, :, (self.each_dim[0]+self.each_dim[3]):(self.each_dim[0]+self.each_dim[3]+self.each_dim[1])], gt_poses[:, :, :self.each_dim[1]])
            hand_loss = self.L1Loss(pred_poses[:, :, (self.each_dim[0]+self.each_dim[3]+self.each_dim[1]):], gt_poses[:, :, self.each_dim[1]:])
            l1_loss = jaw_loss + face_loss + body_loss + hand_loss

            dis_output = self.discriminator(pred_aud)
            gen_error = self.MSELoss(torch.ones_like(dis_output).to(self.device), dis_output)
            gen_loss = self.config.Train.weights.keypoint_loss_weight * l1_loss + self.config.Train.weights.gan_loss_weight * gen_error

            loss_dict['gen'] = gen_error
            loss_dict['jaw_loss'] = jaw_loss
            loss_dict['face_loss'] = face_loss
            loss_dict['body_loss'] = body_loss
            loss_dict['hand_loss'] = hand_loss
            return gen_loss, loss_dict
        else:
            raise ValueError(mode)

    def infer_on_audio(self, aud_fn, fps=30, B=1, **kwargs):
        output = []
        assert self.args.infer, "train mode"
        self.generator.eval()

        aud_feat = get_mfcc_ta(aud_fn, fps=fps, encoder_choice='mfcc').transpose(1, 0)
        aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            pred_poses = self.generator(aud_feat).transpose(1, 2)
        end = time.time()
        pred_poses = torch.cat([pred_poses[:, :self.each_dim[0]], pred_poses[:, (self.each_dim[0]+self.each_dim[3]):], pred_poses[:, self.each_dim[0]:(self.each_dim[0]+self.each_dim[3])]], 1)

        return pred_poses, end-start

    def generate(self, aud, id):
        self.generator.eval()
        pred_poses = self.generator(aud)
        return pred_poses


if __name__ == '__main__':
    from trainer.options import parse_args

    parser = parse_args()
    args = parser.parse_args(
        ['--exp_name', '0', '--data_root', '0', '--speakers', '0', '--pre_pose_length', '4', '--generate_length', '64',
         '--infer'])

    generator = TrainWrapper(args)

    aud_fn = '../sample_audio/jon.wav'
    initial_pose = torch.randn(64, 108, 4)
    norm_stats = (np.random.randn(108), np.random.randn(108))
    output = generator.infer_on_audio(aud_fn, initial_pose, norm_stats)

    print(output.shape)
