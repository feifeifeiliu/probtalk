import math
import os
import sys
import time

import torch
from torch.optim.lr_scheduler import StepLR
import random

from data_utils.foundation_models import get_textfeat
from data_utils.mesh_dataset import renaming_suffix

sys.path.append(os.getcwd())

from nets.layers import *
from nets.base import TrainWrapperBaseClass
from nets.inpainting.RTransformer import RTransformer
from nets.inpainting.gated_pixelcnn_1d import GatedRefineNet as pixelcnn, Stage2
from nets.inpainting.vqvae_1d_sc import VQVAE_SC as s2g_body
from nets.inpainting.vqvae_1d_sc import AudioEncoder, EncoderTopDown
from nets.utils import parse_audio, denormalize
from data_utils import get_mfcc, get_melspec, get_mfcc_old, get_mfcc_psf, get_mfcc_psf_min, get_mfcc_ta
import numpy as np
import torch.optim as optim
from nets.spg.wav2vec import Wav2Vec2Model
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
        self.type = self.config.Model.hf_type
        self.vqtype = self.config.Model.hf_vq_type
        self.init_params()
        self.audio = True
        self.composition = self.config.Model.composition
        self.bh_model = self.config.Model.bh_model
        self.two_stage = self.config.Model.two_stage
        self.motion_context = self.config.Model.motion_context

        if self.type == 'fbhe':
            in_dim = self.full_dim
        elif self.type == 'bh':
            in_dim = self.each_dim[1] + self.each_dim[2]
        elif self.type == 'fe':
            in_dim = self.each_dim[0] + self.each_dim[3]

        emb_dim = config.Model.groups * 128
        self.knn = self.config.Model.knn
        if self.knn:
            state_dim = emb_dim
        else:
            state_dim = config.Model.code_num
        self.RTransformer = RTransformer(in_dim, 768, 512, 6, use_label=config.Model.identity,
                                         motion_context=self.motion_context).to(self.device)
        # self.AudEnc = EncoderFormer(768, 512, 512, 3).to(self.device)
        self.AudEnc = nn.Identity()
        # self.AudEnc = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)  # "vitouphy/wav2vec2-xls-r-300m-phoneme""facebook/wav2vec2-base-960h"
        # self.AudEnc.feature_extractor._freeze_parameters()
        if self.vqtype == 'fbhe':
            vq_dim = self.full_dim
        elif self.vqtype == 'bh':
            vq_dim = self.each_dim[1] + self.each_dim[2]
        elif self.vqtype == 'fe':
            vq_dim = self.each_dim[0] + self.each_dim[3]
        else:
            vq_dim = 0
        if self.vqtype != 'novq':
            self.VQ = s2g_body(vq_dim, embedding_dim=emb_dim, q_type=config.Model.q_type,
                               num_embeddings=config.Model.code_num,
                               num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=512,
                               groups=config.Model.groups, share_code=config.Model.share_code).to(self.device)
            model_path = self.config.Model.vq_path
            model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
            self.VQ.load_state_dict(model_ckpt['generator']['VQ'])
        else:
            self.VQ = nn.Identity()

        self.in_dim = in_dim
        self.vq_dim = vq_dim

        self.discriminator = None
        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d

        self.random_mask = torch.zeros(
            [self.config.Data.pose.generate_length - 29, 1, self.config.Data.pose.generate_length])
        for i in range(self.random_mask.shape[0]):
            self.random_mask[i, 0, i:(i + 30)] = 1

        super().__init__(args, config)

    def to_parallel(self):
        if torch.cuda.device_count() > 1:
            self.RTransformer = torch.nn.DataParallel(self.RTransformer)
            # self.TexEnc = torch.nn.DataParallel(self.TexEnc)
            self.AudEnc = torch.nn.DataParallel(self.AudEnc)
            self.VQ = torch.nn.DataParallel(self.VQ)

    # def parameters(self):
    #     return self.parameters()
    def init_optimizer(self):

        print('using Adam')
        self.generator_optimizer = optim.AdamW(
            [{'params': self.RTransformer.parameters()},
             # {'params': self.TexEnc.parameters()},
             {'params': self.AudEnc.parameters()},
             ],
            lr=self.config.Train.learning_rate.generator_learning_rate,
            betas=[0.9, 0.99]
        )

        # print('using SGD')
        # self.generator_optimizer = optim.SGD(
        #     [{'params': self.HFTransformer.parameters()},
        #      # {'params': self.TexEnc.parameters()},
        #      {'params': self.AudEnc.parameters()}, ],
        #     lr=0.001,
        #     momentum=0.9,
        #     nesterov=False,
        # )

    def state_dict(self):
        if isinstance(self.RTransformer, torch.nn.DataParallel):
            model_state = {
                'HFTransformer': self.RTransformer.module.state_dict(),
                # 'TexEnc': self.TexEnc.module.state_dict(),
                'AudEnc': self.AudEnc.module.state_dict(),
                'generator_optim': self.generator_optimizer.state_dict(),
            }
        else:
            model_state = {
                'HFTransformer': self.RTransformer.state_dict(),
                # 'TexEnc': self.TexEnc.state_dict(),
                'AudEnc': self.AudEnc.state_dict(),
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

        if 'HFTransformer' in state_dict:
            self.RTransformer.load_state_dict(state_dict['HFTransformer'])

        if 'AudEnc' in state_dict:
            self.AudEnc.load_state_dict(state_dict['AudEnc'])

        # if 'TexEnc' in state_dict:
        #     self.TexEnc.load_state_dict(state_dict['TexEnc'])

        if 'generator_optim' in state_dict:
            self.generator_optimizer.load_state_dict(state_dict['generator_optim'])

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        fm_dict = bat['fm_dict']
        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        expression = bat['expression'].to(self.device).to(torch.float32)

        id = bat['speaker'].to(self.device)
        # id = F.one_hot(id, self.num_classes)

        jaw = poses[:, :self.each_dim[0], :]

        poses = poses[:, self.c_index, :]
        full_poses = torch.cat([jaw, poses, expression], dim=1)

        if self.vqtype == 'fbhe':
            vq_poses = full_poses
        elif self.vqtype == 'bh':
            vq_poses = poses
        elif self.vqtype == 'fe':
            vq_poses = torch.cat([jaw, expression], dim=1)

        if self.type == 'fbhe':
            poses = full_poses
        elif self.type == 'bh':
            poses = poses
        elif self.type == 'fe':
            poses = torch.cat([jaw, expression], dim=1)

        gt_poses = poses
        bs, n, t = gt_poses.size()
        # assert t == aud.shape[2], 't= ' + str(t) + 'aud= ' + str(aud.shape[2])
        epoch = bat['epoch']
        # mask_ratio = (epoch / 20) * 1
        # mask_head = torch.ones([bs, 1, 1], device=self.device) * 0.5
        # mask_head = torch.bernoulli(mask_head).repeat(1, 1, 30)
        # mask_body = torch.ones([bs, 1, t - 30], device=self.device) * 0.01
        # mask_body = torch.bernoulli(mask_body)
        # mask = torch.cat([mask_head, mask_body], dim=-1)

        if self.motion_context:
            # version 1
            # mask = torch.zeros(bs, poses.shape[-1])
            # for i in range(bs):
            #     start = random.randint(0, poses.shape[-1] - 30)
            #     mask[i, start:start + 30] = 1
            # version 2
            select = torch.randint(0, self.random_mask.shape[0], (bs,))
            mask = self.random_mask[select]
            mask[:64] = 0
            mask = mask.to(self.device)
        else:
            mask = torch.zeros([bs, 1, t], device=self.device)

        blank = torch.zeros([bs, self.full_dim, t], device=self.device)

        if self.vqtype != 'novq':
            with torch.no_grad():
                self.VQ.eval()
                vq_poses = self.VQ(gt_poses=vq_poses)
                vq_poses = F.interpolate(vq_poses, t, mode='linear')  # , align_corners=False

            if self.vqtype == 'fbhe':
                blank = vq_poses
            elif self.vqtype == 'bh':
                blank[:, self.each_dim[0]:(-self.each_dim[3])] = vq_poses
            elif self.vqtype == 'fe':
                blank[:, :self.each_dim[0]] = vq_poses[:, :self.each_dim[0]]
                blank[:, -self.each_dim[3]:] = vq_poses[:, -self.each_dim[3]:]

        if self.type == 'fbhe':
            blank = blank
        elif self.type == 'bh':
            blank = blank[:, self.each_dim[0]:(-self.each_dim[3])]
        elif self.type == 'fe':
            blank = torch.cat([blank[:, :self.each_dim[0]], blank[:, -self.each_dim[3]:]], dim=1)
        vq_poses = blank

        rand_aug = True
        if rand_aug:
            noise = torch.randn_like(vq_poses, device=vq_poses.device)
            vq_poses = vq_poses + 0.05 * noise

        input_poses = gt_poses * mask + vq_poses * (1 - mask)

        # with torch.no_grad():
        #     aud = fm_dict['aud_p'](aud, sampling_rate=16000, return_tensors="pt")
        #     for key in aud.data:
        #         aud.data[key] = aud.data[key].to('cuda')
        #
        #     aud = fm_dict['aud_m'](input_values=aud.data['input_values'].squeeze())
        #     aud = aud.last_hidden_state
        #     aud = F.interpolate(aud.transpose(1, 2), size=gt_poses.shape[2], mode='linear')

        # aud = self.AudEnc(input_values=aud, frame_num=t).last_hidden_state.transpose(1, 2)
        aud = self.AudEnc(aud)

        # mask_2 = torch.ones([bs, t], device=self.device) * mask_ratio
        # mask_2 = torch.bernoulli(mask_2).unsqueeze(dim=-1)
        #
        # input_poses = input_poses * mask_2

        # if epoch > 9:
        #     with torch.no_grad():
        #         self_poses = self.HFTransformer(input_poses.transpose(1, 2), aud, mask.transpose(1, 2))
        #         self_poses = gt_poses * mask + self_poses.transpose(1, 2) * (1-mask)
        #
        #     input_poses = torch.cat([input_poses[:(bs//2)], self_poses[(bs//2):]], dim=0)

        pred_poses = self.RTransformer(input_poses, aud, mask, id)
        loss, loss_dict = self.get_loss(pred_poses, gt_poses, self.type, mask)

        self.generator_optimizer.zero_grad()
        loss.backward()
        grad_HF = torch.nn.utils.clip_grad_norm(self.RTransformer.parameters(), self.config.Train.max_gradient_norm)
        grad_A = torch.nn.utils.clip_grad_norm(self.AudEnc.parameters(), self.config.Train.max_gradient_norm)
        # grad_T = torch.nn.utils.clip_grad_norm(self.TexEnc.parameters(), self.config.Train.max_gradient_norm)
        loss_dict['loss'] = loss.item()
        loss_dict['grad_HF'] = grad_HF.item()
        loss_dict['grad_A'] = grad_A.item()
        # loss_dict['grad_T'] = grad_T.item()

        self.generator_optimizer.step()

        return total_loss, loss_dict

    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 type,
                 mask
                 ):

        loss_dict = {}
        nmask = (1 - mask)

        t_mask = mask.sum()
        t_nmask = nmask.sum()

        # compute loss with mask
        # mface_loss = torch.abs(pred_poses[..., :self.dim_list[1]]*mask-gt_poses[..., :self.dim_list[1]]*mask).sum()/(t_mask*self.each_dim[0])
        # mbody_loss = torch.abs(pred_poses[..., self.dim_list[1]:self.dim_list[3]]*mask-
        #                        gt_poses[..., self.dim_list[1]:self.dim_list[3]]*mask).sum()/(t_mask*self.each_dim[1])
        # mhand_loss = torch.abs(pred_poses[..., self.dim_list[3]:self.dim_list[4]]*mask-
        #                        gt_poses[..., self.dim_list[3]:self.dim_list[4]]*mask).sum()/(t_mask*self.each_dim[2])
        # mexp_loss = torch.abs(pred_poses[..., self.dim_list[4]:]*mask-gt_poses[..., self.dim_list[4]:]*mask).sum()/(t_mask*self.each_dim[3])
        # mrec_loss = (mface_loss + mbody_loss + mhand_loss + mexp_loss) / 4
        # loss_dict['mrec_loss'] = mrec_loss

        # compute loss without mask
        if self.type == 'fbhe':
            face_loss = torch.abs(
                pred_poses[:, :self.dim_list[1]] * nmask - gt_poses[:, :self.dim_list[1]] * nmask).sum() / (
                                    t_nmask * self.each_dim[0])
            body_loss = torch.abs(pred_poses[:, self.dim_list[1]:self.dim_list[3]] * nmask -
                                  gt_poses[:, self.dim_list[1]:self.dim_list[3]] * nmask).sum() / (
                                    t_nmask * self.each_dim[1])
            hand_loss = torch.abs(pred_poses[:, self.dim_list[3]:self.dim_list[4]] * nmask -
                                  gt_poses[:, self.dim_list[3]:self.dim_list[4]] * nmask).sum() / (
                                    t_nmask * self.each_dim[2])
            exp_loss = torch.abs(
                pred_poses[:, self.dim_list[4]:] * nmask - gt_poses[:, self.dim_list[4]:] * nmask).sum() / (
                                   t_nmask * self.each_dim[3])
            # exp_loss = ((pred_poses[:, self.dim_list[4]:] * nmask - gt_poses[:, self.dim_list[4]:] * nmask)**2).sum() / (
            #                        t_nmask * self.each_dim[3])
        elif self.type == 'fe':
            face_loss = torch.abs(
                pred_poses[:, :self.dim_list[1]] * nmask - gt_poses[:, :self.dim_list[1]] * nmask).sum() / (
                                t_nmask * self.each_dim[0])
            body_loss = hand_loss = 0
            exp_loss = ((pred_poses[:, self.dim_list[1]:] * nmask - gt_poses[:,
                                                                    self.dim_list[1]:] * nmask) ** 2).sum() / (
                               t_nmask * self.each_dim[3])
        elif self.type == 'bh':
            body_loss = torch.abs(pred_poses[:, :self.each_dim[1]] * nmask -
                                  gt_poses[:, :self.each_dim[1]] * nmask).sum() / (
                                t_nmask * self.each_dim[1])
            hand_loss = torch.abs(pred_poses[:, self.each_dim[1]:] * nmask -
                                  gt_poses[:, self.each_dim[1]:] * nmask).sum() / (
                                t_nmask * self.each_dim[2])
            face_loss = exp_loss = 0
        else:
            raise RuntimeError

        weight = len(self.type)
        rec_loss = (face_loss + body_loss + hand_loss + exp_loss) / weight
        loss_dict['face_loss'] = face_loss
        loss_dict['body_loss'] = body_loss
        # loss_dict['hand_loss'] = hand_loss
        # loss_dict['exp_loss'] = exp_loss

        if self.type != 'bh':
            v_pr = pred_poses[:, :-self.each_dim[3], 1:] - pred_poses[:, :-self.each_dim[3], :-1]
            v_gt = gt_poses[:, :-self.each_dim[3], 1:] - gt_poses[:, :-self.each_dim[3], :-1]
        else:
            v_pr = pred_poses[:, :, 1:] - pred_poses[:, :, :-1]
            v_gt = gt_poses[:, :, 1:] - gt_poses[:, :, :-1]
        velocity_loss = torch.mean(torch.abs(v_pr - v_gt))

        gen_loss = 0 + rec_loss + velocity_loss

        loss_dict['rec_loss'] = rec_loss
        loss_dict['velocity_loss'] = velocity_loss

        return gen_loss, loss_dict

    def infer(self, aud_fn, pred_poses, gt_poses, mask, B=1, id=0, am=None, am_sr=None, audio_model=None):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        self.RTransformer.eval()
        self.AudEnc.eval()
        self.VQ.eval()

        aud_feat = get_mfcc_ta(aud_fn, sr=22000, fps=30, smlpx=True, type='mfcc', am=am, audio_model=audio_model,
                               encoder_choice=self.config.Model.encoder_choice)
        aud_feat = aud_feat.transpose(1, 0)
        aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)

        if id is None:
            id = torch.tensor([0]).to(self.device)
        else:
            id = id.repeat(B)

        with torch.no_grad():

            if self.vqtype == 'bh':
                blank_jaw = torch.zeros([B, self.each_dim[0], aud_feat.shape[2]], device=self.device)
                blank_exp = torch.zeros([B, self.each_dim[3], aud_feat.shape[2]], device=self.device)
                pred_poses = torch.cat([blank_jaw, pred_poses, blank_exp], dim=1)

            aud_feat = self.AudEnc(aud_feat)
            # pred_poses = self.VQ(gt_poses=gt_poses)
            input_poses = gt_poses * mask + pred_poses * (1 - mask)

            pred_poses = self.RTransformer(input_poses, aud_feat, mask)

            pred_poses = gt_poses * mask + pred_poses * (1 - mask)
        output = pred_poses

        # _, gt_e, gt_latent, _ = self.VQ.encode(gt_poses=gt_poses)
        #
        # pred_poses = self.VQ.decode(pred_state, None)

        # output = pred_poses.transpose(1, 2)

        return output

    def continuity(self, aud_fn, pred_poses, pre_poses, sec1_frames, B=1, id=0, am=None, am_sr=None, audio_model=None):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        self.RTransformer.eval()
        self.AudEnc.eval()
        self.VQ.eval()

        num_pre = 24

        aud_feat = get_mfcc_ta(aud_fn, sr=22000, fps=30, smlpx=True, type='mfcc', am=am, audio_model=audio_model,
                               encoder_choice=self.config.Model.encoder_choice)
        aud_feat = aud_feat.transpose(1, 0)
        aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)

        if pre_poses is None:
            num_frames = sec1_frames
            gt_poses = torch.zeros([1, num_frames, self.in_dim]).to('cuda')
            mask = torch.zeros([1, num_frames, 1]).to('cuda')
            aud_feat = aud_feat[..., :num_frames]
        else:
            num_frames = num_pre + aud_feat.shape[2] - sec1_frames
            gt_poses = torch.cat([pre_poses[:, -num_pre:, :],
                                  torch.zeros([pre_poses.shape[0], num_frames - num_pre, pre_poses.shape[2]],
                                              device='cuda')], dim=1)
            mask_head = torch.ones(gt_poses.size()[:1], device='cuda') * 1
            mask_head = torch.bernoulli(mask_head).reshape(-1, 1, 1).repeat(1, num_pre, 1)
            mask_body = torch.ones([gt_poses.shape[0], num_frames - num_pre], device='cuda') * 0.00
            mask_body = torch.bernoulli(mask_body).unsqueeze(dim=-1).repeat(1, 1, 1)
            mask = torch.cat([mask_head, mask_body], dim=1)
            aud_feat = aud_feat[..., (sec1_frames - num_pre):]

        if id is None:
            id = torch.tensor([0]).to(self.device)
        else:
            id = id.repeat(B)

        with torch.no_grad():

            if self.vqtype == 'bh':
                blank_jaw = torch.zeros([B, aud_feat.shape[2], self.each_dim[0]], device=self.device)
                blank_exp = torch.zeros([B, aud_feat.shape[2], self.each_dim[3]], device=self.device)
                pred_poses = torch.cat([blank_jaw, pred_poses, blank_exp], dim=-1)

            aud_feat = self.AudEnc(aud_feat)
            # pred_poses = self.VQ(gt_poses=gt_poses)
            input_poses = gt_poses * mask + pred_poses * (1 - mask)
            pred_poses = input_poses.transpose(1, 2)

            pred_poses = self.RTransformer(pred_poses, aud_feat, mask.transpose(1, 2))

            pred_poses = gt_poses.transpose(1, 2) * mask.transpose(1, 2) + pred_poses * (1 - mask.transpose(1, 2))

        smooth = False
        if smooth:
            lamda = 0.8
            smooth_f = 10
            frame = num_pre
            for i in range(smooth_f):
                f = frame + i
                l = lamda * (i + 1) / smooth_f
                pred_poses[..., f] = (1 - l) * pred_poses[..., f - 1] + l * pred_poses[..., f]

        output = pred_poses

        return output

    def infer_on_batch(self, aud, B, id, gt_poses, mask, pred_poses=None, **kwargs):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        self.RTransformer.eval()
        self.AudEnc.eval()
        self.VQ.eval()

        aud = aud.repeat(B, 1, 1)

        blank = torch.zeros([B, self.full_dim, aud.shape[2]], device=self.device)
        if self.two_stage:
            pred_poses = F.interpolate(pred_poses, size=aud.shape[2], mode='linear')
            blank = gt_poses * mask + blank + pred_poses * (1 - mask)
        else:
            blank = gt_poses * mask + blank

        if self.type == 'fbhe':
            blank = blank
        elif self.type == 'bh':
            blank = blank[:, self.each_dim[0]:(-self.each_dim[3])]
        elif self.type == 'fe':
            blank = torch.cat([blank[:, :self.each_dim[0]], blank[:, -self.each_dim[3]:]], dim=1)
        input_poses = blank

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            aud_feat = self.AudEnc(aud)
            pred_poses = self.RTransformer(input_poses, aud_feat, mask, id)

            # pred_list = []
            # for i in range(math.ceil(aud.shape[-1] / 180)):
            #     aud_feat = self.AudEnc(aud[:, :, i * 180:(i + 1) * 180])
            #     pred_list.append(self.HFTransformer(input_poses[:, :, i * 180:(i + 1) * 180], aud_feat, mask, id))
            # pred_poses = torch.cat(pred_list, dim=-1)

            pred_poses = gt_poses * mask + pred_poses * (1 - mask)  # if not self.type=='fe' else pred_poses
        end = time.time()

        # if mask.sum() != 0:
        #     pred_poses = gaussian_smoothing_around_timepoint(pred_poses, self.each_dim[0], self.each_dim[0]+self.each_dim[1]+self.each_dim[2])
        return pred_poses, end - start

    def infer_on_audio(self, aud_fn, fm_dict, id, gt_poses, mask, B, pred_poses=None, slice=None, aud=None,
                       text_fn=None, **kwargs):
        output = []

        assert self.args.infer, "train mode"
        self.RTransformer.eval()
        self.AudEnc.eval()
        self.VQ.eval()
        # if aud is None:
        # from filelock import SoftFileLock
        # lock = SoftFileLock('lockfile')
        # with lock:
        aud = get_mfcc_ta(aud_fn,
                          fps=30,
                          sr=fm_dict['sr'],
                          fm_dict=fm_dict,
                          encoder_choice=self.config.Model.encoder_choice,
                          )
        aud = torch.from_numpy(aud).unsqueeze(0).transpose(1, 2).to('cuda')

        # if fm_dict['text_m'] is not None and text_fn is not None:
        #     text = get_textfeat(aud_fn, renaming_suffix(aud_fn, '.txt'), fm_dict)
        # else:
        #     text = np.zeros([aud.shape[0], 1])
        # text = torch.from_numpy(text).unsqueeze(0).transpose(1, 2).to('cuda')

        aud = aud.repeat(B, 1, 1)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():

            blank = torch.zeros([B, self.full_dim, aud.shape[2]], device=self.device)
            if self.two_stage:
                pred_poses = F.interpolate(pred_poses, size=aud.shape[2], mode='linear')
                blank = gt_poses * mask + blank + pred_poses * (1 - mask)
            else:
                blank = gt_poses * mask + blank

            if self.type == 'fbhe':
                blank = blank
            elif self.type == 'bh':
                blank = blank[:, self.each_dim[0]:(-self.each_dim[3])]
            elif self.type == 'fe':
                blank = torch.cat([blank[:, :self.each_dim[0]], blank[:, -self.each_dim[3]:]], dim=1)
            input_poses = blank

            # aud_feat = self.AudEnc(aud)
            # pred_poses = self.HFTransformer(input_poses, aud_feat, mask, id)

            pred_list = []
            for i in range(math.ceil(aud.shape[-1] / 180)):
                aud_feat = self.AudEnc(aud[:, :, i * 180:(i + 1) * 180])
                pred_list.append(self.RTransformer(input_poses[:, :, i * 180:(i + 1) * 180], aud_feat, mask, id))
            pred_poses = torch.cat(pred_list, dim=-1)

            pred_poses = input_poses * mask + pred_poses * (1 - mask) # if not self.type=='fe' else pred_poses
        end = time.time()

        return pred_poses, end - start


def gaussian_kernel(window_size, sigma):
    """
    Generates a Gaussian kernel.
    :param window_size: The size of the window.
    :param sigma: The standard deviation of the Gaussian distribution.
    :return: A 1D tensor representing the Gaussian kernel.
    """
    kernel_range = torch.arange(window_size) - window_size // 2
    kernel = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_smoothing_around_timepoint(data, s, e, time_point=30, window_size=5, sigma=2):
    """
    Applies Gaussian smoothing to the specified time point region of the data.

    :param data: Tensor of shape [B, N, T], where B is the batch size, N is the number of joints, and T is the time length.
    :param time_point: The central time point around which smoothing is applied.
    :param window_size: The size of the Gaussian kernel window.
    :param sigma: The standard deviation of the Gaussian distribution.
    :return: Smoothed data.
    """
    B, N, T = data.shape
    smoothed_data = data.clone()
    kernel = gaussian_kernel(window_size, sigma).to(data.device)

    # Calculate the start and end points for applying the kernel
    start = time_point - 10
    end = time_point + 10

    # Applying the kernel around the specified time point
    for b in range(B):
        for n in range(s, e):
            smoothed_data[b, n, start:end] = torch.nn.functional.conv1d(
                data[b, n, start-5:end+5].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=window_size // 2
            ).squeeze()[..., 5:-5]

    return smoothed_data

