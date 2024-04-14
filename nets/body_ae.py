import os
import sys

sys.path.append(os.getcwd())

from nets.base import TrainWrapperBaseClass
from nets.spg.s2glayers import Discriminator as D_S2G
from nets.spg.vqvae_1d import AE as s2g_body
import torch
import torch.optim as optim
import torch.nn.functional as F

from data_utils.lower_body import c_index_3d, c_index_6d


def separate_aa(aa):
    aa = aa[:, :, :].reshape(aa.shape[0], aa.shape[1], -1, 5)
    axis = F.normalize(aa[:, :, :, :3], dim=-1)
    angle = F.normalize(aa[:, :, :, 3:5], dim=-1)
    return axis, angle


class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0

        self.gan = False
        self.convert_to_6d = False
        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.type = 'bh'
        self.init_params()
        self.num_classes = 4
        self.g = s2g_body(self.full_dim, embedding_dim=64, num_embeddings=0,
                          num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=512).to(self.device)
        self.discriminator = None

        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d

        if torch.cuda.device_count() > 1:
            self.g = torch.nn.DataParallel(self.g)

        super().__init__(args, config)

    def init_optimizer(self):

        self.g_optimizer = optim.Adam(
            self.g.parameters(),
            lr=self.config.Train.learning_rate.generator_learning_rate,
            betas=[0.9, 0.999]
        )

    def state_dict(self):
        if isinstance(self.g, torch.nn.DataParallel):
            model_state = {
                'g': self.g.module.state_dict(),
                'generator_optim': self.g_optimizer.state_dict(),
            }
        else:
            model_state = {
                'g': self.g.state_dict(),
                'generator_optim': self.g_optimizer.state_dict(),
            }
        return model_state


    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        expression = bat['expression'].to(self.device).to(torch.float32)

        poses = torch.cat([poses[:, :self.each_dim[0]],poses[:, self.c_index], expression], 1)
        gt_poses = poses

        loss = 0
        loss_dict, loss = self.vq_train(gt_poses, 'g', self.g, loss_dict, loss)

        return total_loss, loss_dict

    def vq_train(self, gt, name, model, dict, total_loss):
        x_recon = model(gt_poses=gt)
        loss, loss_dict = self.get_loss(pred_poses=x_recon, gt_poses=gt)
        # total_loss = total_loss + loss

        if name == 'g':
            optimizer_name = 'g_optimizer'

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
                 ):
        loss_dict = {}


        rec_loss = torch.mean(torch.abs(pred_poses - gt_poses))
        v_pr = pred_poses[:, :self.dim_list[-1], 1:] - pred_poses[:, :self.dim_list[-1], :-1]
        v_gt = gt_poses[:, :self.dim_list[-1], 1:] - gt_poses[:, :self.dim_list[-1], :-1]
        velocity_loss = torch.mean(torch.abs(v_pr - v_gt))

        gen_loss = rec_loss + velocity_loss

        loss_dict['rec_loss'] = rec_loss
        loss_dict['velocity_loss'] = velocity_loss

        return gen_loss, loss_dict

    def load_state_dict(self, state_dict):
        # try:
        self.g.load_state_dict(state_dict['g'], strict=False)
        # except:
        #     self.g.load_state_dict(state_dict['VQ'])

    def extract(self, x):
        self.g.eval()
        # if x.shape[2] > self.full_dim:
        #     if x.shape[2] == 239:
        #         x = x[:, :, 102:]
        #     x = x[:, :, self.c_index]
        feat = self.g.encode(x)
        return feat.transpose(1, 2), x

    def reconstruct(self, x):
        self.g.eval()
        feat = self.g.encode(x)
        x_recon = self.g.decode(feat)
        x_recon = F.interpolate(x_recon, size=x.shape[2], align_corners=False, mode='linear')
        return x_recon
