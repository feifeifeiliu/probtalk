import os
import sys

from torch.optim.lr_scheduler import StepLR

sys.path.append(os.getcwd())

from nets.layers import *
from nets.base import TrainWrapperBaseClass
from nets.spg.s2glayers import Generator as G_S2G, Discriminator as D_S2G
from nets.spg.vqvae_1d import VQVAE as s2g_body
# from nets.inpainting.vqvae_1d_sc import VQVAE_SC as s2g_body
from nets.utils import parse_audio, denormalize
from data_utils import get_mfcc_ta
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from data_utils.lower_body import c_index_3d, c_index_6d


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
        self.composition = self.config.Model.composition

        if self.type == 'fbhe':
            in_dim = self.full_dim
        elif self.type == 'bh':
            in_dim = self.each_dim[1] + self.each_dim[2]
        elif self.type == 'fe':
            in_dim = self.each_dim[0] + self.each_dim[3]
        if self.composition:
            self.g_body = s2g_body(self.each_dim[1], embedding_dim=64, num_embeddings=config.Model.code_num, num_hiddens=1024,
                                   num_residual_layers=2, num_residual_hiddens=512).to(self.device)
            self.g_hand = s2g_body(self.each_dim[2], embedding_dim=64, num_embeddings=config.Model.code_num, num_hiddens=1024,
                                   num_residual_layers=2, num_residual_hiddens=512).to(self.device)
        else:
            self.g = s2g_body(in_dim, embedding_dim=64, num_embeddings=config.Model.code_num,
                              num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=512,
                              ).to(self.device)

        self.discriminator = None

        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d

        if torch.cuda.device_count() > 1:
            self.g_body = torch.nn.DataParallel(self.g_body)
            self.g_hand = torch.nn.DataParallel(self.g_hand)

        super().__init__(args, config)

    def init_optimizer(self):
        print('using Adam')
        if self.composition:
            self.g_body_optimizer = optim.AdamW(
                self.g_body.parameters(),
                lr=self.config.Train.learning_rate.generator_learning_rate,
                betas=[0.9, 0.999]
            )
            self.g_hand_optimizer = optim.AdamW(
                self.g_hand.parameters(),
                lr=self.config.Train.learning_rate.generator_learning_rate,
                betas=[0.9, 0.999]
            )
        else:
            self.g_optimizer = optim.AdamW(
                self.g.parameters(),
                lr=self.config.Train.learning_rate.generator_learning_rate,
                betas=[0.9, 0.999]
            )

    def state_dict(self):
        if self.composition:
            if isinstance(self.g_body, torch.nn.DataParallel):
                model_state = {
                    'g_body': self.g_body.module.state_dict(),
                    'g_body_optim': self.g_body_optimizer.state_dict(),
                    'g_hand': self.g_hand.module.state_dict(),
                    'g_hand_optim': self.g_hand_optimizer.state_dict(),
                    'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
                    'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
                }
            else:
                model_state = {
                    'g_body': self.g_body.state_dict(),
                    'g_body_optim': self.g_body_optimizer.state_dict(),
                    'g_hand': self.g_hand.state_dict(),
                    'g_hand_optim': self.g_hand_optimizer.state_dict(),
                    'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
                    'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
                }
        else:
            model_state = {
                'g': self.g.state_dict(),
                'g_optim': self.g_optimizer.state_dict(),
                'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
                'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
            }
        return model_state

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)

        # id = bat['speaker'].to(self.device)
        # id = F.one_hot(id, self.num_classes)

        gt_poses = poses[:, self.c_index, :]
        # gt_poses = poses.permute(0, 2, 1)
        b_poses = gt_poses[:, :self.each_dim[1]]
        h_poses = gt_poses[:, self.each_dim[1]:]

        if self.composition:
            loss = 0
            loss_dict, loss = self.vq_train(b_poses[:, :], 'b', self.g_body, loss_dict, loss)
            loss_dict, loss = self.vq_train(h_poses[:, :], 'h', self.g_hand, loss_dict, loss)
        else:
            loss = 0
            loss_dict, loss = self.vq_train(gt_poses[:, :], 'g', self.g, loss_dict, loss)

        return total_loss, loss_dict

    def vq_train(self, gt, name, model, dict, total_loss, pre=None):
        e_q_loss, x_recon = model(gt_poses=gt, result_form='part')
        e_q_loss = e_q_loss.mean()
        # print(e_q_loss)
        loss, loss_dict = self.get_loss(pred_poses=x_recon, gt_poses=gt, e_q_loss=e_q_loss, pre=pre)
        # total_loss = total_loss + loss

        if name == 'b':
            optimizer_name = 'g_body_optimizer'
        elif name == 'h':
            optimizer_name = 'g_hand_optimizer'
        elif name == 'g':
            optimizer_name = 'g_optimizer'
        else:
            raise ValueError("model's name must be b or h")
        optimizer = getattr(self, optimizer_name)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key in list(loss_dict.keys()):
            dict[name + key] = loss_dict.get(key, 0).item()
        return dict, total_loss

    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 e_q_loss,
                 pre=None
                 ):
        loss_dict = {}


        rec_loss = torch.mean(torch.abs(pred_poses - gt_poses))
        v_pr = pred_poses[:, :, 1:] - pred_poses[:, :, :-1]
        v_gt = gt_poses[:, :, 1:] - gt_poses[:, :, :-1]
        velocity_loss = torch.mean(torch.abs(v_pr - v_gt))

        gen_loss = rec_loss + e_q_loss + velocity_loss

        loss_dict['rec_loss'] = rec_loss
        loss_dict['velocity_loss'] = velocity_loss
        loss_dict['e_q_loss'] = e_q_loss

        return gen_loss, loss_dict

    def infer_on_audio(self, aud_fn, initial_pose=None, norm_stats=None, exp=None, var=None, w_pre=False, continuity=False,
                       id=None, fps=15, sr=22000, smooth=False, **kwargs):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        if self.composition:
            self.g_body.eval()
            self.g_hand.eval()
        else:
            self.g.eval()

        if self.config.Data.pose.normalization:
            assert norm_stats is not None
            data_mean = norm_stats[0]
            data_std = norm_stats[1]

        # assert initial_pose.shape[-1] == pre_length
        if initial_pose is not None:
            gt = initial_pose[:, :, :].to(self.device).to(torch.float32)
            pre_poses = initial_pose[:, :, :15].permute(0, 2, 1).to(self.device).to(torch.float32)
            poses = initial_pose.permute(0, 2, 1).to(self.device).to(torch.float32)
            B = pre_poses.shape[0]
        else:
            gt = None
            pre_poses = None
            B = 1

        if type(aud_fn) == torch.Tensor:
            aud_feat = torch.tensor(aud_fn, dtype=torch.float32).to(self.device)
            num_poses_to_generate = aud_feat.shape[-1]
        else:
            aud_feat = get_mfcc_ta(aud_fn, sr=sr, fps=fps, smlpx=True, type='mfcc').transpose(1, 0)
            aud_feat = aud_feat[:, :]
            num_poses_to_generate = aud_feat.shape[-1]
            aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
            aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)

        # pre_poses = torch.randn(pre_poses.shape).to(self.device).to(torch.float32)
        if id is None:
            id = F.one_hot(torch.tensor([[0]]), self.num_classes).to(self.device)

        with torch.no_grad():
            aud_feat = aud_feat.permute(0, 2, 1)
            gt_poses = gt[:, self.c_index].permute(0, 2, 1)
            if self.composition:
                if continuity:
                    pred_poses_body = []
                    pred_poses_hand = []
                    pre_b = None
                    pre_h = None
                    for i in range(5):
                        _, pred_body = self.g_body(gt_poses=gt_poses[:, i*60:(i+1)*60, :self.each_dim[1]], pre_state=pre_b)
                        pre_b = pred_body[..., -1:].transpose(1,2)
                        pred_poses_body.append(pred_body)
                        _, pred_hand = self.g_hand(gt_poses=gt_poses[:, i*60:(i+1)*60, self.each_dim[1]:], pre_state=pre_h)
                        pre_h = pred_hand[..., -1:].transpose(1,2)
                        pred_poses_hand.append(pred_hand)

                    pred_poses_body = torch.cat(pred_poses_body, dim=2)
                    pred_poses_hand = torch.cat(pred_poses_hand, dim=2)
                else:
                    _, pred_poses_body = self.g_body(gt_poses=gt_poses[..., :self.each_dim[1]], id=id)
                    _, pred_poses_hand = self.g_hand(gt_poses=gt_poses[..., self.each_dim[1]:], id=id)
                pred_poses = torch.cat([pred_poses_body, pred_poses_hand], dim=1)
            else:
                _, pred_poses = self.g(gt_poses=gt_poses, id=id)
            pred_poses = pred_poses.transpose(1, 2).cpu().numpy()
        output = pred_poses

        if self.config.Data.pose.normalization:
            output = denormalize(output, data_mean, data_std)

        if smooth:
            lamda = 0.8
            smooth_f = 10
            frame = 149
            for i in range(smooth_f):
                f = frame + i
                l = lamda * (i + 1) / smooth_f
                output[0, f] = (1 - l) * output[0, f - 1] + l * output[0, f]

        output = np.concatenate(output, axis=1)

        return output

    def infer_on_batch(self, gt_poses=None, **kwargs):

        assert self.args.infer, "train mode"
        if self.composition:
            self.g_body.eval()
            self.g_hand.eval()
        else:
            self.g.eval()
        gt_poses = gt_poses[:, self.each_dim[0]:-self.each_dim[3]]
        gt_poses = gt_poses[:, self.c_index]

        with torch.no_grad():
            if self.composition:
                pred_poses_body = self.g_body(gt_poses=gt_poses[:, :self.each_dim[1]])
                pred_poses_hand = self.g_hand(gt_poses=gt_poses[:, self.each_dim[1]:])
                pred_poses = torch.cat([pred_poses_body, pred_poses_hand], dim=1)
            else:
                pred_poses = self.g(gt_poses=gt_poses, id=id)

        return pred_poses

    def load_state_dict(self, state_dict):
        if self.composition:
            self.g_body.load_state_dict(state_dict['g_body'])
            self.g_hand.load_state_dict(state_dict['g_hand'])
        else:
            self.g.load_state_dict(state_dict['g'])
