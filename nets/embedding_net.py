"""
https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context.git
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import smplx

from data_utils.get_j import get_joints
from data_utils.lower_body import c_index_3d, c_index_6d, poses2pred3D
from nets.base import TrainWrapperBaseClass
from data_utils.consts import smplx_hyperparams


betas_dim = smplx_hyperparams['betas_dim']
exp_dim = smplx_hyperparams['expression_dim']


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net


class PoseEncoderConv(nn.Module):
    def __init__(self, length, dim):
        super().__init__()

        self.net = nn.Sequential(
            ConvNormRelu(dim, 32, batchnorm=True),
            ConvNormRelu(32, 64, batchnorm=True),
            ConvNormRelu(64, 64, True, batchnorm=True),
            nn.Conv1d(64, 32, 3)
        )

        self.out_net = nn.Sequential(
            nn.Linear(1280, 512),  # for 90 frames
            nn.BatchNorm1d(512),
            nn.LeakyReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
        )

        self.fc_mu = nn.Linear(128, 128)
        self.fc_logvar = nn.Linear(128, 128)

    def forward(self, poses, variational_encoding):
        # encode
        # poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)

        # return out, None, None
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar


class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim):
        super().__init__()

        feat_size = 128

        self.pre_net = nn.Sequential(
            nn.Linear(feat_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 720),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose1d(8, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, dim, 3),
        )

    def forward(self, feat):

        out = self.pre_net(feat)
        out = out.view(feat.shape[0], 8, -1)
        out = self.net(out)
        # out = out.transpose(1, 2)
        return out

class EmbeddingNet(nn.Module):
    def __init__(self, args, pose_dim, n_frames):
        super().__init__()
        self.context_encoder = None
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
        self.decoder = PoseDecoderConv(n_frames, pose_dim)

    def forward(self, poses, variational_encoding=False):
        # poses
        poses_feat, _, _ = self.pose_encoder(poses, variational_encoding)

        # decoder
        latent_feat = poses_feat

        out_poses = self.decoder(latent_feat)

        return poses_feat, out_poses

    def extract(self, x):
        self.pose_encoder.eval()
        feat, _, _ = self.pose_encoder(x, False)
        return feat.transpose(0, 1), x

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False


class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0

        self.convert_to_6d = self.config.Data.pose.convert_to_6d
        self.epoch = 0
        self.init_params()

        self.use_joints = False
        self.type = config.Model.vq_type

        if self.use_joints:
            dim = 127*3
        else:
            if self.type == 'fbhe':
                dim = self.full_dim
            elif self.type == 'bh':
                dim = self.each_dim[1] + self.each_dim[2]
            elif self.type == 'fe':
                dim = self.each_dim[0] + self.each_dim[3]

        self.generator = EmbeddingNet(None, dim, 90).to(self.device)

        self.discriminator = None
        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d



        if True:
            smplx_path = './visualise/'
            dtype = torch.float32
            model_params = dict(model_path=smplx_path,
                                model_type='smplx',
                                create_global_orient=True,
                                create_body_pose=True,
                                create_betas=True,
                                num_betas=betas_dim,
                                create_left_hand_pose=True,
                                create_right_hand_pose=True,
                                use_pca=False,
                                flat_hand_mean=False,
                                create_expression=True,
                                num_expression_coeffs=exp_dim,
                                num_pca_comps=12,
                                create_jaw_pose=True,
                                create_leye_pose=True,
                                create_reye_pose=True,
                                create_transl=False,
                                dtype=dtype, )
            self.smplx_model = smplx.create(**model_params).to(self.device)

        if torch.cuda.device_count() > 1:
            self.generator = torch.nn.DataParallel(self.generator)
            self.smplx_model = torch.nn.DataParallel(self.smplx_model)

        super().__init__(args, config)

    def init_optimizer(self):
        print('using Adam')
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=float(self.config.Train.learning_rate.generator_learning_rate),
            betas=[0.9, 0.999]
        )

    def state_dict(self):
        if isinstance(self.generator, torch.nn.DataParallel):
            model_state = {
                'generator': self.generator.module.state_dict(),
                'generator_optim': self.generator_optimizer.state_dict(),
            }
        else:
            model_state = {
                'generator': self.generator.state_dict(),
                'generator_optim': self.generator_optimizer.state_dict(),
            }
        return model_state

    def load_state_dict(self, state_dict):

        from collections import OrderedDict
        new_state_dict = OrderedDict()  # create new OrderedDict that does not contain `module.`
        for k, v in state_dict.items():
            sub_dict = OrderedDict()
            if v is not None:
                for k1, v1 in v.items():
                    name = k1.replace('module.', '')
                    sub_dict[name] = v1
            new_state_dict[k] = sub_dict
        state_dict = new_state_dict
        if 'generator' in state_dict:
            self.generator.load_state_dict(state_dict['generator'])
        else:
            self.generator.load_state_dict(state_dict)

        if 'generator_optim' in state_dict and self.generator_optimizer is not None:
            self.generator_optimizer.load_state_dict(state_dict['generator_optim'])

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None

        poses = bat['poses'].to(self.device).to(torch.float32)
        exp = bat['expression'].to(self.device).to(torch.float32)
        B, T = poses.shape[0], poses.shape[2]

        if self.use_joints:
            poses = torch.cat([poses, exp], dim=1)
            poses = poses2pred3D(poses)
            with torch.no_grad():
                self.smplx_model.eval()
                betas = torch.zeros([1, 300]).to('cuda').to(torch.float32)
                joints = get_joints(self.smplx_model, betas, poses.transpose(1,2), bat=16).reshape(B, T, -1).transpose(1, 2).to(torch.float32)
        else:
            jaw = poses[:, :self.each_dim[0], :]
            bh_poses = poses[:, self.c_index, :]
            full_poses = torch.cat([jaw, bh_poses, exp], dim=1)

            if self.type == 'fbhe':
                joints = full_poses
            elif self.type == 'bh':
                joints = bh_poses
            elif self.type == 'fe':
                joints = torch.cat([jaw, exp], dim=1)

        rec_joints = self.generator(joints)[1]
        self.generator_optimizer.zero_grad()
        loss, loss_dict = self.get_loss(rec_joints, joints)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm(self.generator.parameters(), self.config.Train.max_gradient_norm)
        loss_dict['grad'] = grad.item()
        self.generator_optimizer.step()

        return total_loss, loss_dict


    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 ):
        loss_dict = {}

        rec_loss = torch.mean(torch.abs(pred_poses - gt_poses))
        v_pr = pred_poses[:, 1:] - pred_poses[:, :-1]
        v_gt = gt_poses[:, 1:] - gt_poses[:, :-1]
        velocity_loss = torch.mean(torch.abs(v_pr - v_gt))

        gen_loss = rec_loss + velocity_loss

        loss_dict['rec_loss'] = rec_loss
        loss_dict['velocity_loss'] = velocity_loss

        return gen_loss, loss_dict

    def extract(self, joints):
        self.generator.eval()

        if self.type == 'fbhe':
            joints = joints
        elif self.type == 'bh':
            joints = joints[:, self.each_dim[0]:(-self.each_dim[3])]
        elif self.type == 'fe':
            joints = torch.cat([joints[:, :self.each_dim[0]], joints[:, -self.each_dim[3]:]], dim=1)

        with torch.no_grad():
            feat, x = self.generator.extract(joints)
        return feat.transpose(0,1), x


if __name__ == '__main__':
    # for model debugging
    n_frames = 64
    pose_dim = 10
    encoder = PoseEncoderConv(n_frames, pose_dim)
    decoder = PoseDecoderConv(n_frames, pose_dim)

    poses = torch.randn((4, n_frames, pose_dim))
    feat, _, _ = encoder(poses, True)
    recon_poses = decoder(feat)

    print('input', poses.shape)
    print('feat', feat.shape)
    print('output', recon_poses.shape)
