import os
import sys

import torch
from torch.optim.lr_scheduler import StepLR
import time

from data_utils.consts import get_speaker_id

sys.path.append(os.getcwd())

from nets.layers import *
from nets.base import TrainWrapperBaseClass
from nets.spg.gated_pixelcnn_v2 import GatedPixelCNN as pixelcnn
from nets.spg.vqvae_1d import VQVAE as s2g_body, Wav2VecEncoder
from nets.spg.vqvae_1d import AudioEncoder
from nets.utils import parse_audio, denormalize
from data_utils import get_mfcc, get_melspec, get_mfcc_old, get_mfcc_psf, get_mfcc_psf_min, get_mfcc_ta
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from data_utils.lower_body import c_index_3d, c_index_6d
from data_utils.utils import smooth_geom, get_mfcc_sepa


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
        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.type = 'bh'
        self.init_params()
        self.audio = True
        self.composition = self.config.Model.composition
        self.bh_model = self.config.Model.bh_model
        self.num_classes = get_speaker_id(config.Data.data_root).__len__()

        if self.audio:
            self.audioencoder = AudioEncoder(in_dim=64, num_hiddens=256, num_residual_layers=2, num_residual_hiddens=256).to(self.device)
        else:
            self.audioencoder = None
        dim, layer = 512, 10
        self.generator = pixelcnn(2048, dim, layer, self.num_classes, self.audio, self.bh_model).to(self.device)
        self.g_body = s2g_body(self.each_dim[1], embedding_dim=64, num_embeddings=config.Model.code_num, num_hiddens=1024,
                               num_residual_layers=2, num_residual_hiddens=512).to(self.device)
        self.g_hand = s2g_body(self.each_dim[2], embedding_dim=64, num_embeddings=config.Model.code_num, num_hiddens=1024,
                               num_residual_layers=2, num_residual_hiddens=512).to(self.device)

        model_path = self.config.Model.vq_path
        model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.g_body.load_state_dict(model_ckpt['generator']['g_body'])
        self.g_hand.load_state_dict(model_ckpt['generator']['g_hand'])

        if torch.cuda.device_count() > 1:
            self.g_body = torch.nn.DataParallel(self.g_body, device_ids=[0, 1])
            self.g_hand = torch.nn.DataParallel(self.g_hand, device_ids=[0, 1])
            self.generator = torch.nn.DataParallel(self.generator, device_ids=[0, 1])
            if self.audioencoder is not None:
                self.audioencoder = torch.nn.DataParallel(self.audioencoder, device_ids=[0, 1])

        self.discriminator = None
        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d

        super().__init__(args, config)

    def init_optimizer(self):

        print('using Adam')
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.config.Train.learning_rate.generator_learning_rate,
            betas=[0.9, 0.999]
        )
        if self.audioencoder is not None:
            opt = self.config.Model.AudioOpt
            if opt == 'Adam':
                self.audioencoder_optimizer = optim.Adam(
                    self.audioencoder.parameters(),
                    lr=self.config.Train.learning_rate.generator_learning_rate,
                    betas=[0.9, 0.999]
                )
            else:
                print('using SGD')
                self.audioencoder_optimizer = optim.SGD(
                filter(lambda p: p.requires_grad,self.audioencoder.parameters()),
                lr=self.config.Train.learning_rate.generator_learning_rate*10,
                momentum=0.9,
                nesterov=False,
        )

    def state_dict(self):
        model_state = {
            'generator': self.generator.state_dict(),
            'generator_optim': self.generator_optimizer.state_dict(),
            'audioencoder': self.audioencoder.state_dict() if self.audio else None,
            'audioencoder_optim': self.audioencoder_optimizer.state_dict() if self.audio else None,
            'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
            'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
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

        if self.discriminator is not None:
            self.discriminator.load_state_dict(state_dict['discriminator'])

            if 'discriminator_optim' in state_dict and self.discriminator_optimizer is not None:
                self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optim'])

        if 'audioencoder' in state_dict and self.audioencoder is not None:
            self.audioencoder.load_state_dict(state_dict['audioencoder'])

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)

        id = bat['speaker'].to(self.device)
        # id = F.one_hot(id, self.num_classes)

        poses = poses[:, self.c_index, :]

        gt_poses = poses

        with torch.no_grad():
            self.g_body.eval()
            self.g_hand.eval()
            if torch.cuda.device_count() > 1:
                _, body_latents = self.g_body.module.encode(gt_poses=gt_poses[:, :self.each_dim[1]])
                _, hand_latents = self.g_hand.module.encode(gt_poses=gt_poses[:, self.each_dim[1]:])
            else:
                _, body_latents = self.g_body.encode(gt_poses=gt_poses[:, :self.each_dim[1]])
                _, hand_latents = self.g_hand.encode(gt_poses=gt_poses[:, self.each_dim[1]:])
            latents = torch.cat([body_latents.unsqueeze(dim=-1), hand_latents.unsqueeze(dim=-1)], dim=-1)
            latents = latents.detach()

        if self.audio:
            audio = self.audioencoder(aud[:, :], frame_num=latents.shape[1]*4).unsqueeze(dim=-1).repeat(1, 1, 1, 2)
            logits = self.generator(latents[:, :], id, audio)
        else:
            logits = self.generator(latents, id)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        self.generator_optimizer.zero_grad()
        if self.audio:
            self.audioencoder_optimizer.zero_grad()

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), latents.view(-1))
        loss.backward()

        grad = torch.nn.utils.clip_grad_norm(self.generator.parameters(), self.config.Train.max_gradient_norm)

        if torch.isnan(grad).sum() > 0:
            print('fuck')

        loss_dict['grad'] = grad.item()
        loss_dict['ce_loss'] = loss.item()
        self.generator_optimizer.step()
        if self.audio:
            self.audioencoder_optimizer.step()

        return total_loss, loss_dict

    def infer_on_audio(self, aud_fn, fm_dict, frame, id=None, B=1, **kwargs):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        self.generator.eval()
        self.g_body.eval()
        self.g_hand.eval()

        aud_feat = get_mfcc_ta(aud_fn, fps=30, fm_dict=fm_dict, encoder_choice='mfcc')
        aud_feat = aud_feat.transpose(1, 0)
        aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)

        if id is None:
            id = torch.tensor([0]).to(self.device)
        else:
            id = id.repeat(B)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            if self.audio:
                self.audioencoder.eval()
                audio = self.audioencoder(aud_feat, frame_num=frame).unsqueeze(dim=-1).repeat(1, 1, 1, 2)
                latents = self.generator.generate(id, shape=[audio.shape[2], 2], batch_size=B, aud_feat=audio)
            else:
                latents = self.generator.generate(id, shape=[aud_feat.shape[1]//4, 2], batch_size=B)

            body_latents = latents[..., 0]
            hand_latents = latents[..., 1]

            body, _ = self.g_body.decode(b=body_latents.shape[0], w=body_latents.shape[1], latents=body_latents)
            hand, _ = self.g_hand.decode(b=hand_latents.shape[0], w=hand_latents.shape[1], latents=hand_latents)

            pred_poses = torch.cat([body, hand], dim=1)

        output = pred_poses
        end = time.time()
        return output, end-start

    def infer(self, aud_feat, frame, id, B, pre_latents=None, pre_audio=None, pre_pose=None):
        audio = self.audioencoder(aud_feat.transpose(1, 2), frame_num=frame).unsqueeze(dim=-1).repeat(1, 1, 1, 2)
        latents = self.generator.generate(id, shape=[audio.shape[2], 2], batch_size=B, aud_feat=audio,
                                          pre_latents=pre_latents, pre_audio=pre_audio)

        body_latents = latents[..., 0]
        hand_latents = latents[..., 1]

        body, _ = self.g_body.decode(b=body_latents.shape[0], w=body_latents.shape[1],
                                  latents=body_latents, pre_state=pre_pose['b'])
        hand, _ = self.g_hand.decode(b=hand_latents.shape[0], w=hand_latents.shape[1],
                                  latents=hand_latents, pre_state=pre_pose['h'])

        return latents, audio, body, hand

    def generate(self, aud, id, frame_num=0):

        self.generator.eval()
        self.g_body.eval()
        self.g_hand.eval()
        aud_feat = aud.permute(0, 2, 1)
        if self.audio:
            self.audioencoder.eval()
            audio = self.audioencoder(aud_feat.transpose(1, 2), frame_num=frame_num).unsqueeze(dim=-1).repeat(1, 1, 1, 2)
            latents = self.generator.generate(id, shape=[audio.shape[2], 2], batch_size=aud.shape[0], aud_feat=audio)
        else:
            latents = self.generator.generate(id, shape=[aud_feat.shape[1] // 4, 2], batch_size=aud.shape[0])

        body_latents = latents[..., 0]
        hand_latents = latents[..., 1]

        body = self.g_body.decode(b=body_latents.shape[0], w=body_latents.shape[1], latents=body_latents)
        hand = self.g_hand.decode(b=hand_latents.shape[0], w=hand_latents.shape[1], latents=hand_latents)

        pred_poses = torch.cat([body, hand], dim=1).transpose(1, 2)
        return pred_poses
