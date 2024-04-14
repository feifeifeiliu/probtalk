import os
import sys
import time

import torch
from torch.optim.lr_scheduler import StepLR
import random

sys.path.append(os.getcwd())

from data_utils.consts import get_speaker_id
from data_utils.foundation_models import get_textfeat
from data_utils.mesh_dataset import renaming_suffix
from losses import CrossEntropyLabelSmooth
from nets.inpainting.predictornet import PredictorNet
from nets.layers import *
from nets.base import TrainWrapperBaseClass
from nets.inpainting.vqvae_1d_sc import VQVAE_SC as s2g_body, ConditionEncoder
from data_utils import get_mfcc_ta
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from data_utils.lower_body import c_index_3d, c_index_6d


class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''

    def __init__(self, args, config):
        self.args = args
        self.config = config
        # self.device = torch.device(self.args.gpu)
        self.device = 'cuda'
        self.global_step = 0

        self.convert_to_6d = self.config.Data.pose.convert_to_6d
        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.type = self.config.Model.vq_type
        self.init_params()
        self.encoder_choice = self.config.Model.encoder_choice
        self.max_epoch = self.config.Train.epochs
        self.audio = self.config.Model.p_audio
        self.text = self.config.Model.p_text
        self.num_classes = get_speaker_id(config.Data.data_root).__len__()
        self.motion_context = config.Model.motion_context

        motion_dim = self.each_dim[1] + self.each_dim[2]

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

        aud_dim, text_dim = self.getFM_dim(self.audio, self.text)
        self.AudEnc = ConditionEncoder(in_dim=aud_dim, num_hiddens=256, num_residual_layers=2,
                                       num_residual_hiddens=256).to(self.device)
        self.TextEnc = ConditionEncoder(in_dim=text_dim, num_hiddens=256, num_residual_layers=2,
                                        num_residual_hiddens=256).to(self.device)

        self.Predictor = PredictorNet(self.knn, in_dim, state_dim, 512, 10, 6, self.num_classes,
                                      groups=config.Model.groups, identity=config.Model.identity,
                                      maskgit=config.Model.maskgit, maskgit_T=config.Model.maskgit_T,
                                      transformer=config.Model.transformer,
                                      text=self.text, audio=self.audio, motion_context=config.Model.motion_context).to(self.device)
        self.VQ = s2g_body(in_dim, embedding_dim=emb_dim,
                           num_embeddings=config.Model.code_num,
                           num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=512,
                           groups=config.Model.groups, q_type=config.Model.q_type, share_code=config.Model.share_code).to(self.device)
        model_path = self.config.Model.vq_path
        model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.VQ.load_state_dict(model_ckpt['generator']['VQ'])
        self.in_dim = in_dim

        self.discriminator = None
        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d

        self.crossentropy = CrossEntropyLabelSmooth(state_dim)

        self.random_mask = torch.zeros([self.config.Data.pose.generate_length-29,1,self.config.Data.pose.generate_length])
        for i in range(self.random_mask.shape[0]):
            self.random_mask[i, 0, i:(i+30)] = 1

        super().__init__(args, config)

    def to_parallel(self):
        if torch.cuda.device_count() > 1:
            self.AudEnc = torch.nn.DataParallel(self.AudEnc)
            self.TextEnc = torch.nn.DataParallel(self.TextEnc)
            self.Predictor = torch.nn.DataParallel(self.Predictor)
            self.VQ = torch.nn.DataParallel(self.VQ)

    # def parameters(self):
    #     return self.parameters()
    def init_optimizer(self):

        print('using AdamW')
        self.generator_optimizer = optim.AdamW(
            [{'params': self.AudEnc.parameters()},
             {'params': self.Predictor.parameters()},
             {'params': self.TextEnc.parameters()},],
            lr=self.config.Train.learning_rate.generator_learning_rate,
            betas=[0.9, 0.99]
        )

        # print('using SGD')
        # self.generator_optimizer = optim.SGD(
        #     [{'params': self.AudEnc.parameters()},
        #      {'params': self.Predictor.parameters()}, ],
        #     lr=0.01,
        #     momentum=0.9,
        # )

    def state_dict(self):
        if isinstance(self.Predictor, torch.nn.DataParallel):
            model_state = {
                'AudEnc': self.AudEnc.module.state_dict(),
                'Predictor': self.Predictor.module.state_dict(),
                'TextEnc': self.TextEnc.module.state_dict(),
                'generator_optim': self.generator_optimizer.state_dict(),
            }
        else:
            model_state = {
                'AudEnc': self.AudEnc.state_dict(),
                'Predictor': self.Predictor.state_dict(),
                'TextEnc': self.TextEnc.state_dict(),
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

        if 'AudEnc' in state_dict:
            self.AudEnc.load_state_dict(state_dict['AudEnc'])
        if 'Predictor' in state_dict:
            self.Predictor.load_state_dict(state_dict['Predictor'])
        if 'TextEnc' in state_dict:
            self.TextEnc.load_state_dict(state_dict['TextEnc'])

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
        # mask_ratio = ((10 - epoch) / 10) * 0.5 + 0.01
        # mask_all = True
        # if epoch < 10:
        #     mask = torch.ones([bs, 1, t], device=self.device) * mask_ratio
        #     mask = torch.bernoulli(mask)
        # else:
        #     mask_head = torch.ones([bs, 1, 1], device=self.device) * 0.5
        #     mask_head = torch.bernoulli(mask_head).repeat(1, 1, 8)
        #     mask_body = torch.ones([bs, 1, t - 8], device=self.device) * 0.05
        #     mask_body = torch.bernoulli(mask_body)
        #     mask = torch.cat([mask_head, mask_body], dim=-1)
        if self.motion_context:
            # version 1
            # mask = torch.zeros(bs, poses.shape[-1])
            # for i in range(bs):
            #     start = random.randint(0, poses.shape[-1] - 30)
            #     mask[i, start:start + 30] = 1
            # version 2
            select = torch.randint(0, self.random_mask.shape[0], (bs,))
            mask = self.random_mask[select]
            # version 3
            mask_ratio = ((5 - epoch) / 5) * 0.5 + 0.01
            # if epoch < 5:
            #     mask = torch.ones([bs, 1, t]) * mask_ratio
            #     mask = torch.bernoulli(mask)
            # else:
            # select = torch.randint(0, self.random_mask.shape[0], (bs,))
            # mask_0 = self.random_mask[select]
            # mask_1 = torch.ones([bs, 1, t]) * 0.05
            # mask_1 = torch.bernoulli(mask_1)
            # mask = (mask_0.bool() | mask_1.bool()).float()
            mask[:64] = 0
            mask = mask.to(self.device)
        else:
            mask = torch.zeros([bs, 1, t], device=self.device)
        input_poses = gt_poses * mask

        with torch.no_grad():
            self.VQ.eval()
            if isinstance(self.VQ, torch.nn.DataParallel):
                _, qo_gt, _ = self.VQ.module.encode(gt_poses=gt_poses)
                # _, qo_in, _ = self.VQ.module.encode(gt_poses=input_poses)
            else:
                _, qo_gt, _ = self.VQ.encode(gt_poses=gt_poses)
                # _, qo_in, _ = self.VQ.encode(gt_poses=input_poses)
            state = qo_gt.loss
            # input_codes = qo_in.loss

        if self.audio:
            audio = self.AudEnc(aud)
        else:
            audio = None

        if self.text:
            text = bat['text_feat'].to(self.device).to(torch.float32)
            text = self.TextEnc(text)
        else:
            text = None
        epoch_ratio = min(max(epoch / 99, 0.5), 1)
        pred_state = self.Predictor(input_poses, state, mask, id, audio, text, epoch_ratio)

        self.generator_optimizer.zero_grad()
        state = state
        if isinstance(pred_state, dict):
            loss = torch.tensor([0.], device=self.device)
            for key in pred_state:
                loss_dict[key] = F.cross_entropy(pred_state[key].reshape(-1, pred_state[key].shape[-1]), state.reshape(-1))
                # loss_dict[key] = self.crossentropy(pred_state[key].reshape(-1, pred_state[key].shape[-1]), state)
                loss = loss + loss_dict[key]
        else:
            if len(pred_state.shape) == 3:
                pred_state = pred_state.permute(0, 2, 1).contiguous()
            loss = F.cross_entropy(pred_state.reshape(-1, pred_state.shape[-1]), state.reshape(-1))
            # loss = self.crossentropy(pred_state.reshape(-1, pred_state.shape[-1]), state.reshape(-1))
        loss.backward()
        grad_ae = torch.nn.utils.clip_grad_norm(self.AudEnc.parameters(), self.config.Train.max_gradient_norm)
        grad_p = torch.nn.utils.clip_grad_norm(self.Predictor.parameters(), self.config.Train.max_gradient_norm)
        grad_te = torch.nn.utils.clip_grad_norm(self.TextEnc.parameters(), self.config.Train.max_gradient_norm)
        loss_dict['loss'] = loss.item()
        loss_dict['grad_ae'] = grad_ae.item()
        loss_dict['grad_p'] = grad_p.item()
        # loss_dict['grad_te'] = grad_te.item()

        self.generator_optimizer.step()

        return total_loss, loss_dict

    def infer(self, aud_fn, text, gt_poses, mask, B=1, id=0, am=None, am_sr=None, audio_model=None):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        self.AudEnc.eval()
        self.TextEnc.eval()
        self.Predictor.eval()
        self.VQ.eval()

        aud_feat = get_mfcc_ta(aud_fn, sr=22000, fps=30, smlpx=True, type='mfcc', am=am, audio_model=audio_model,
                               encoder_choice=self.config.Model.encoder_choice)
        aud_feat = aud_feat.transpose(1, 0)
        aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)

        if self.type == 'bh':
            gt_poses = gt_poses[:, 6:-self.each_dim[3], :]

        if self.text:
            text = self.TextEnc(text) if self.text else None
        else:
            text = None

        if id is None:
            id = torch.tensor([0]).to(self.device)
        else:
            id = id.repeat(B)

        with torch.no_grad():
            input_poses = gt_poses * mask
            audio = self.AudEnc(aud_feat)
            state = torch.ones([audio.shape[0], audio.shape[2], 4])
            pred_state = self.Predictor(input_poses, state, mask, id, audio, text, 0)
            # pred_state = self.Predictor.predict(input_poses, mask, id, audio, B, self.VQ.vq_layer.embeddings)
            if not self.knn:
                pred_state = self.VQ.vq_layer.quantize_all(pred_state)
                pred_state = pred_state.transpose(1, 2)

        pred_poses = self.VQ.decode(pred_state, None)

        return pred_poses

    def continuity(self, aud_fn, text, pre_poses, sec1_frames, B=1, id=0, am=None, am_sr=None, audio_model=None):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        self.AudEnc.eval()
        self.TextEnc.eval()
        self.Predictor.eval()
        self.VQ.eval()

        aud_feat = get_mfcc_ta(aud_fn, sr=22000, fps=30, smlpx=True, type='mfcc', am=am, audio_model=audio_model,
                               encoder_choice=self.config.Model.encoder_choice)
        aud_feat = aud_feat.transpose(1, 0)
        aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)

        num_pre = 24

        if pre_poses is None:
            num_frames = sec1_frames
            gt_poses = torch.zeros([1, sec1_frames, self.in_dim]).to('cuda')
            mask = torch.zeros([1, sec1_frames, 1]).to('cuda')
            aud_feat = aud_feat[..., :sec1_frames]
            text = text[..., :sec1_frames]
        else:
            num_frames = num_pre + aud_feat.shape[2] - sec1_frames
            pre_poses = pre_poses[:, :, 6:-self.each_dim[3]] if self.type == 'bh' else pre_poses
            gt_poses = torch.cat([pre_poses[:, -num_pre:, :],
                                  torch.zeros([pre_poses.shape[0], num_frames - num_pre, pre_poses.shape[2]],
                                              device='cuda')], dim=1)
            mask_head = torch.ones(gt_poses.size()[:1], device='cuda') * 1
            mask_head = torch.bernoulli(mask_head).reshape(-1, 1, 1).repeat(1, num_pre, 1)
            mask_body = torch.ones([gt_poses.shape[0], num_frames - num_pre], device='cuda') * 0.00
            mask_body = torch.bernoulli(mask_body).unsqueeze(dim=-1).repeat(1, 1, 1)
            mask = torch.cat([mask_head, mask_body], dim=1).to('cuda')
            aud_feat = aud_feat[..., (sec1_frames - num_pre):]
            text = text[..., (sec1_frames - num_pre):]

        if self.text:
            text = self.TextEnc(text) if self.text else None
        else:
            text = None

        if id is None:
            id = torch.tensor([0]).to(self.device)
        else:
            id = id.repeat(B)

        with torch.no_grad():
            input_poses = gt_poses * mask
            audio = self.AudEnc(aud_feat[:, :])
            state = torch.zeros([audio.shape[0], audio.shape[2], 4])
            pred_state = self.Predictor(input_poses, state, mask, id, audio, text, 0)
            # pred_state = self.Predictor.predict(input_poses, mask, id, audio, B, self.VQ.vq_layer.embeddings)
            if not self.knn:
                pred_state = self.VQ.vq_layer.quantize_all(pred_state)
                pred_state = pred_state.transpose(1, 2)
        # output = pred_state

        # _, gt_e, gt_latent, _ = self.VQ.encode(gt_poses=gt_poses)
        #
        pred_poses = self.VQ.decode(pred_state, None)

        output = pred_poses.transpose(1, 2)

        return output

    def infer_on_batch(self, aud, text, id, B, gt_poses, mask, **kwargs):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        self.AudEnc.eval()
        self.TextEnc.eval()
        self.Predictor.eval()
        self.VQ.eval()

        if self.type == 'bh':
            gt_poses = gt_poses[:, self.each_dim[0]:-self.each_dim[3]]
        elif self.type == 'fe':
            gt_poses = torch.cat([gt_poses[:, :self.each_dim[0]], gt_poses[:, -self.each_dim[3]:]], 1)
        aud = aud.repeat(B, 1, 1)
        text = text.repeat(B, 1, 1)
        text = F.interpolate(text, size=aud.shape[2], align_corners=False, mode='linear')
        id = torch.tensor(id).to('cuda').repeat(B)

        # aud = aud[:, :, :180]
        # text = text[:, :, :180]

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            input_poses = gt_poses * mask
            if self.audio is not None:
                audio = self.AudEnc(aud)
                state = torch.zeros([audio.shape[0], audio.shape[2], 4])
            else:
                audio = None
            if self.text is not None:
                text = self.TextEnc(text)
                state = torch.zeros([text.shape[0], text.shape[2], 4])
            else:
                text = None
            pred_state = self.Predictor(input_poses, state, mask, id, audio, text, 1)

            pred_state = self.VQ.vq_layer.quantize_all(pred_state)
            pred_state = pred_state.transpose(1, 2)

            pred_poses = self.VQ.decode(pred_state, None)
        end = time.time()

        pred_poses = F.interpolate(pred_poses, size=aud.shape[2], mode='linear')
        pred_poses = input_poses * mask + pred_poses * (1 - mask)

        return pred_poses, end-start

    def infer_on_vq(self, aud, text, id, gt_poses, **kwargs):
        B = 1
        assert self.args.infer, "train mode"
        self.AudEnc.eval()
        self.TextEnc.eval()
        self.Predictor.eval()
        self.VQ.eval()

        gt_poses = gt_poses[:, self.c_index]
        mask = torch.zeros([B, 1, aud.shape[2]]).to('cuda')
        aud = aud.repeat(B, 1, 1)
        text = text.repeat(B, 1, 1)
        id = id.repeat(B)

        with torch.no_grad():
            _, gt_e, gt_code, _ = self.VQ.encode(gt_poses=gt_poses)
            # gt_code = gt_code.reshape(1, -1)

            # input_poses = (gt_poses * mask).repeat(gt_code.shape[1]-1, 1, 1)
            # mask = mask.repeat(gt_code.shape[1]-1, 1, 1)
            # if self.audio:
            #     audio = self.AudEnc(aud).repeat(gt_code.shape[1]-1, 1, 1)
            # else:
            #     audio = None
            # if self.text:
            #     text = self.TextEnc(text).repeat(gt_code.shape[1]-1, 1, 1)
            # else:
            #     text = None
            # state = self.mask_sequence(gt_code)
            # logits = self.Predictor(input_poses, state, mask, id, audio, text, 1, True)
            # logits = self.seqs_merge(logits.reshape(gt_code.shape[1]-1, -1, self.config.Model.code_num))
            #
            # loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt_code[0, 1:])

            # input_poses = (gt_poses * mask)
            # mask = mask
            # if self.audio:
            #     audio = self.AudEnc(aud)
            # else:
            #     audio = None
            # if self.text:
            #     text = self.TextEnc(text)
            # else:
            #     text = None
            # # state = self.mask_sequence(gt_code)
            # state = torch.ones([audio.shape[0], audio.shape[2], 4], device=self.device, dtype=torch.int) * self.config.Model.code_num
            # logits = self.Predictor(input_poses, state, mask, id, audio, text, 1, True)
            # # logits = self.seqs_merge(logits.reshape(gt_code.shape[1]-1, -1, self.config.Model.code_num))
            # loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt_code[0, :])

            # input_poses = (gt_poses * mask)
            # mask = mask
            # if self.audio:
            #     audio = self.AudEnc(aud)
            # else:
            #     audio = None
            # if self.text:
            #     text = self.TextEnc(text)
            # else:
            #     text = None
            # # state = self.mask_sequence(gt_code)
            # state = torch.ones([audio.shape[0], audio.shape[2], 4], device=self.device, dtype=torch.int) * self.config.Model.code_num
            # logits = self.Predictor(input_poses, state, mask, id, audio, text, 1, True)
            # # logits = self.seqs_merge(logits.reshape(gt_code.shape[1]-1, -1, self.config.Model.code_num))
            # loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt_code[0, :])

            code_mask = torch.rand_like(gt_code.to(torch.float32)) < 0.85
            input_poses = (gt_poses * mask)
            mask = mask
            if self.audio:
                audio = self.AudEnc(aud)
            else:
                audio = None
            if self.text:
                text = self.TextEnc(text)
            else:
                text = None
            # state = self.mask_sequence(gt_code)
            state = gt_code.clone()
            state[code_mask] = self.config.Model.code_num
            logits = self.Predictor(input_poses, state, mask, id, audio, text, 1, True)
            logits = logits[code_mask].reshape(-1, logits.shape[-1])
            gt_code = gt_code[code_mask].reshape(-1)
            # logits = self.seqs_merge(logits.reshape(gt_code.shape[1]-1, -1, self.config.Model.code_num))
            loss = F.cross_entropy(logits, gt_code)

        return loss

    def infer_on_audio(self, aud_fn, fm_dict, id, B, gt_poses, mask, text_fn=None, slice=None, **kwargs):
        output = []

        assert self.args.infer, "train mode"
        self.AudEnc.eval()
        self.TextEnc.eval()
        self.Predictor.eval()
        self.VQ.eval()

        aud = get_mfcc_ta(aud_fn,
                          fps=30,
                          sr=fm_dict['sr'],
                          fm_dict=fm_dict,
                          encoder_choice=self.config.Model.encoder_choice,
                          )

        if self.text is not None:
            text = get_textfeat(aud_fn, renaming_suffix(aud_fn, '.txt'), fm_dict)
        else:
            text = np.zeros([aud.shape[0], 1])
        aud = torch.from_numpy(aud).unsqueeze(0).transpose(1, 2).to('cuda')
        text = torch.from_numpy(text).unsqueeze(0).transpose(1, 2).to('cuda').to(torch.float32)

        if self.type == 'bh':
            gt_poses = gt_poses[:, self.each_dim[0]:-self.each_dim[3]]
        elif self.type == 'fe':
            gt_poses = torch.cat([gt_poses[:, :self.each_dim[0]], gt_poses[:, -self.each_dim[3]:]], 1)
        aud = aud.repeat(B, 1, 1)
        text = text.repeat(B, 1, 1)
        text = F.interpolate(text, size=aud.shape[2], align_corners=False, mode='linear')
        id = torch.tensor(id).to('cuda').repeat(B)

        # aud = aud[:, :, :180]
        # text = text[:, :, :180]

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            input_poses = gt_poses * mask
            if self.audio is not None:
                audio = self.AudEnc(aud)
                state = torch.zeros([audio.shape[0], audio.shape[2], 4])
            else:
                audio = None
            if self.text is not None:
                text = self.TextEnc(text)
                state = torch.zeros([text.shape[0], text.shape[2], 4])
            else:
                text = None
            pred_state = self.Predictor(input_poses, state, mask, id, audio, text, 1)

            pred_state = self.VQ.vq_layer.quantize_all(pred_state)
            pred_state = pred_state.transpose(1, 2)

            pred_poses = self.VQ.decode(pred_state, None)
        end = time.time()

        return pred_poses, end-start

    def infer_to_code(self, aud, text, id, B, gt_poses, mask, **kwargs):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        self.AudEnc.eval()
        self.TextEnc.eval()
        self.Predictor.eval()
        self.VQ.eval()

        if self.type == 'bh':
            gt_poses = gt_poses[:, self.each_dim[0]:-self.each_dim[3]]
        elif self.type == 'fe':
            gt_poses = torch.cat([gt_poses[:, :self.each_dim[0]], gt_poses[:, -self.each_dim[3]:]], 1)
        aud = aud.repeat(B, 1, 1)
        text = text.repeat(B, 1, 1)
        text = F.interpolate(text, size=aud.shape[2], align_corners=False, mode='linear')
        id = torch.tensor(id).to('cuda').repeat(B)

        # aud = aud[:, :, :180]
        # text = text[:, :, :180]

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            input_poses = gt_poses * mask
            if self.audio is not None:
                audio = self.AudEnc(aud)
                state = torch.zeros([audio.shape[0], audio.shape[2], 4])
            else:
                audio = None
            if self.text is not None:
                text = self.TextEnc(text)
                state = torch.zeros([text.shape[0], text.shape[2], 4])
            else:
                text = None
            codes = self.Predictor(input_poses, state, mask, id, audio, text, 1)

            pred_state = self.VQ.vq_layer.quantize_all(codes)
            pred_state = pred_state.transpose(1, 2)

            pred_poses = self.VQ.decode(pred_state, None)
        end = time.time()

        pred_poses = F.interpolate(pred_poses, size=aud.shape[2], mode='linear')
        pred_poses = input_poses * mask + pred_poses * (1 - mask)

        return pred_poses, codes, end - start


    def reconstruct(self, gt_poses):

        assert self.args.infer, "train mode"
        self.VQ.eval()

        with torch.no_grad():
            z, e, eql_or_lat, enc_feats = self.VQ.encode(gt_poses)
            enc_feats[1] = 0
            enc_feats[2] = 0
            enc_feats[3] = 0
            enc_feats[4] = 0
            pred_poses = self.VQ.decode(e, enc_feats)

        output = pred_poses.transpose(1, 2)

        return output

    def mask_sequence(self, input_seq):
        """
        input_seq: a PyTorch tensor of shape (1, seq_len)
        """
        _, seq_len = input_seq.size()
        output = torch.ones(seq_len - 1, seq_len, device=input_seq.device, dtype=torch.int) * self.config.Model.code_num
        for i in range(seq_len - 1):
            output[i, :i + 1] = input_seq[0, :i + 1]
        return output

    def seqs_merge(self, input_seq):
        """
        input_seq: a PyTorch tensor of shape (seq_len-1, seq_len, code_num)
        """
        _, seq_len, code_num = input_seq.size()
        output = torch.zeros(1, seq_len - 1, code_num, device=input_seq.device)
        for i in range(seq_len - 1):
            output[0, i] = input_seq[i, i + 1]
        return output
