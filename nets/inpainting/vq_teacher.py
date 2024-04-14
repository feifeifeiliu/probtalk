import os
import sys
import time

import torch
from torch.optim.lr_scheduler import StepLR

sys.path.append(os.getcwd())

from nets.layers import *
from nets.base import TrainWrapperBaseClass
from nets.inpainting.gated_pixelcnn_1d import GatedPixelCNN as pixelcnn
from nets.inpainting.vqvae_1d_sc import VQVAE_SC as s2g_body
from nets.spg.vqvae_1d import AudioEncoder
from nets.utils import parse_audio, denormalize
from data_utils import get_mfcc, get_melspec, get_mfcc_old, get_mfcc_psf, get_mfcc_psf_min, get_mfcc_ta
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from data_utils.lower_body import c_index_3d, c_index_6d
from data_utils.utils import smooth_geom, get_mfcc_sepa


def freeze_model(model, to_freeze_dict, keep_step=None):

    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
        else:
            pass

    return model


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
        self.type = self.config.Model.vq_type
        self.init_params()

        emb_dim = config.Model.groups * config.Model.code_dim
        # emb_dim = 128

        if self.type == 'fbhe':
            in_dim = self.full_dim
        elif self.type == 'bh':
            in_dim = self.each_dim[1] + self.each_dim[2]
        elif self.type == 'fe':
            in_dim = self.each_dim[0] + self.each_dim[3]

        self.VQ = s2g_body(in_dim, embedding_dim=emb_dim,
                           num_embeddings=config.Model.code_num,
                           num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=512,
                           q_type=config.Model.q_type, groups=config.Model.groups, share_code=config.Model.share_code).to(self.device)

        self.discriminator = None

        self.load_pretrain = False
        if self.load_pretrain:
            model_path = self.config.Model.vq_path
            model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
            self.VQ.load_state_dict(model_ckpt['generator']['VQ'], strict=False)
            self.VQ = freeze_model(model=self.VQ, to_freeze_dict=model_ckpt['generator']['VQ'])

        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d

        if torch.cuda.device_count() > 1:
            self.VQ = torch.nn.DataParallel(self.VQ)

        super().__init__(args, config)

    # def parameters(self):
    #     return self.parameters()
    def init_optimizer(self):
        if self.load_pretrain:
            params = self.VQ.vq_layer.residual_learner.parameters()
        else:
            params = self.VQ.parameters()

        print('using Adam')
        self.generator_optimizer = optim.Adam([
            {'params': params},],
            lr=self.config.Train.learning_rate.generator_learning_rate,
            betas=[0.9, 0.999]
        )

    def state_dict(self):
        if isinstance(self.VQ, torch.nn.DataParallel):
            model_state = {
                'VQ': self.VQ.module.state_dict(),
                'generator_optim': self.generator_optimizer.state_dict(),
            }
        else:
            model_state = {
                'VQ': self.VQ.state_dict(),
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

        if 'VQ' in state_dict:
            self.VQ.load_state_dict(state_dict['VQ'])

        if 'generator_optim' in state_dict:
            self.generator_optimizer.load_state_dict(state_dict['generator_optim'])

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        expression = bat['expression'].to(self.device).to(torch.float32)

        id = bat['speaker'].to(self.device)
        # id = F.one_hot(id, self.num_classes)

        jaw = poses[:, :self.each_dim[0], :]

        poses = poses[:, self.c_index, :]

        if self.type == 'fbhe':
            poses = torch.cat([jaw, poses, expression], dim=1)
        elif self.type == 'bh':
            poses = poses
        elif self.type == 'fe':
            poses = torch.cat([jaw, expression], dim=1)

        gt_poses = poses
        bs, n, t = gt_poses.size()
        epoch = bat['epoch']
        input_poses = gt_poses

        # if self.load_pretrain:
        #     self.VQ.encoder.eval()
        #     self.VQ.decoder.eval()
        z, QuantizationOutput, pred_poses = self.VQ(gt_poses=input_poses, result_form='full', audio=aud, label=id, load_pretrain=self.load_pretrain)
        eql_or_lat = QuantizationOutput.loss
        eql_or_lat = eql_or_lat.mean()

        self.generator_optimizer.zero_grad()
        loss, loss_dict = self.get_loss(pred_poses.transpose(1,2), gt_poses.transpose(1,2), eql_or_lat)
        grad = torch.nn.utils.clip_grad_norm_(self.VQ.parameters(), self.config.Train.max_gradient_norm)
        loss_dict['q_loss'] = QuantizationOutput.q_loss
        loss_dict['lr_loss'] = QuantizationOutput.lr_loss
        loss_dict['grad'] = grad.item()
        loss.backward()

        # loss_dict['ce_loss'] = loss.item()
        self.generator_optimizer.step()

        return total_loss, loss_dict

    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 e_q_loss,
                 ):
        loss_dict = {}

        if self.type == 'fbhe':
            face_loss = F.smooth_l1_loss(pred_poses[..., :self.dim_list[1]], gt_poses[..., :self.dim_list[1]], beta=0.01)
            body_loss = F.smooth_l1_loss(pred_poses[..., self.dim_list[1]:self.dim_list[3]], gt_poses[..., self.dim_list[1]:self.dim_list[3]], beta=0.01)
            hand_loss = F.smooth_l1_loss(pred_poses[..., self.dim_list[3]:self.dim_list[4]], gt_poses[..., self.dim_list[3]:self.dim_list[4]], beta=0.01)
            exp_loss = F.smooth_l1_loss(pred_poses[..., self.dim_list[4]:], gt_poses[..., self.dim_list[4]:], beta=0.01)
            weight = 0.4
            rec_loss = (face_loss + exp_loss) * (0.5 - weight) + (body_loss + hand_loss) * weight
            # rec_loss = (face_loss + body_loss + hand_loss + exp_loss)/4
            # rec_loss = (face_loss + exp_loss) * 0.1 + (body_loss + hand_loss) * 0.4
            loss_dict['face_loss'] = face_loss
            loss_dict['body_loss'] = body_loss
            loss_dict['hand_loss'] = hand_loss
            loss_dict['exp_loss'] = exp_loss
        elif self.type == 'bh':
            body_loss = F.smooth_l1_loss(pred_poses[..., :self.each_dim[1]],
                                         gt_poses[..., :self.each_dim[1]], beta=0.01)
            hand_loss = F.smooth_l1_loss(pred_poses[..., self.each_dim[1]:],
                                         gt_poses[..., self.each_dim[1]:], beta=0.01)
            rec_loss = (body_loss + hand_loss)/2
            loss_dict['body_loss'] = body_loss
            loss_dict['hand_loss'] = hand_loss
        elif self.type == 'fe':
            jaw_loss = F.smooth_l1_loss(pred_poses[..., self.dim_list[0]:self.dim_list[1]],
                                         gt_poses[..., self.dim_list[0]:self.dim_list[1]], beta=0.01)
            exp_loss = F.smooth_l1_loss(pred_poses[..., -self.each_dim[3]:],
                                         gt_poses[..., -self.each_dim[3]:], beta=0.01)
            rec_loss = (jaw_loss + exp_loss) / 2
            loss_dict['jaw_loss'] = jaw_loss
            loss_dict['exp_loss'] = exp_loss
            # rec_loss = torch.mean(torch.abs(pred_poses - gt_poses))
            # loss_dict['rec_loss'] = rec_loss
        v_pr = pred_poses[:, 1:, :-self.each_dim[3]] - pred_poses[:, :-1, :-self.each_dim[3]]
        v_gt = gt_poses[:, 1:, :-self.each_dim[3]] - gt_poses[:, :-1, :-self.each_dim[3]]
        # v_pr = pred_poses[:, 1:] - pred_poses[:, :-1]
        # v_gt = gt_poses[:, 1:] - gt_poses[:, :-1]
        velocity_loss = torch.mean(torch.abs(v_pr - v_gt))

        gen_loss = rec_loss + e_q_loss + velocity_loss
        loss_dict['velocity_loss'] = velocity_loss
        loss_dict['e_q_loss'] = e_q_loss

        return gen_loss, loss_dict

    def infer_on_batch(self, gt_poses, aud, text, id, B, **kwargs):

        assert self.args.infer, "train mode"
        aud = aud.repeat(B, 1, 1)
        text = text.repeat(B, 1, 1)
        id = id.repeat(B)
        self.VQ.eval()
        if gt_poses.shape[1] == 238 or gt_poses.shape[1] == 376:
            if self.type == 'bh':
                gt_poses = gt_poses[:, self.each_dim[0]:-self.each_dim[3]]

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():

            z, _ = self.VQ.encoder(gt_poses)
            e, _ = self.VQ.vq_layer(z, audio=aud, label=id, load_pretrain=True)
            pred_poses = self.VQ.decode(e, None)
        end = time.time()
        return pred_poses, end-start

    def infer_on_audio(self, gt_poses, aud, text, id, B, **kwargs):

        assert self.args.infer, "train mode"
        aud = aud.repeat(B, 1, 1)
        text = text.repeat(B, 1, 1)
        id = id.repeat(B)
        self.VQ.eval()
        if gt_poses.shape[1] == 238 or gt_poses.shape[1] == 376:
            if self.type == 'bh':
                gt_poses = gt_poses[:, self.each_dim[0]:-self.each_dim[3]]

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():

            z, _ = self.VQ.encoder(gt_poses)
            qo = self.VQ.vq_layer(z, audio=aud, label=id, load_pretrain=False)
            e = qo.quantized
            pred_poses = self.VQ.decode(e, None)
        end = time.time()
        return pred_poses, end-start

    def get_latents(self, gt_poses=None, **kwargs):

        self.VQ.eval()

        gt_poses = torch.cat([gt_poses[:, :self.each_dim[0]], gt_poses[:, self.c_index], gt_poses[:, -self.each_dim[3]:]], 1)

        with torch.no_grad():
            _, gt_e, gt_latent, _ = self.VQ.encode(gt_poses=gt_poses)
        return gt_latent
