import os
import sys
import time

sys.path.append(os.getcwd())

from nets.layers import *
from nets.base import TrainWrapperBaseClass
# from nets.spg.faceformer import Faceformer
from nets.spg.s2g_face import Generator as s2g_face
from losses import KeypointLoss
from nets.utils import denormalize
from data_utils import get_mfcc_psf, get_mfcc_psf_min, get_mfcc_ta
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import normalize
# import smplx
from data_utils.consts import smplx_hyperparams, get_speaker_id

betas_dim = smplx_hyperparams['betas_dim']
exp_dim = smplx_hyperparams['expression_dim']


class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = 'cuda'
        self.global_step = 0

        self.convert_to_6d = self.config.Data.pose.convert_to_6d
        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.type = 'fbhe'
        self.init_params()
        self.num_classes = get_speaker_id(config.Data.data_root).__len__()


        self.generator = s2g_face(
            n_poses=self.config.Data.pose.generate_length,
            each_dim=self.each_dim,
            dim_list=self.dim_list,
            training=not self.args.infer,
            device=self.device,
            identity=False if self.convert_to_6d else True,
            num_classes=self.num_classes,
        ).to(self.device)

        # self.generator = Faceformer().to(self.device)

        self.discriminator = None
        self.am = None

        self.MSELoss = KeypointLoss().to(self.device)
        super().__init__(args, config)

    def init_optimizer(self):
        self.generator_optimizer = optim.SGD(
            filter(lambda p: p.requires_grad,self.generator.parameters()),
            lr=0.001,
            momentum=0.9,
            nesterov=False,
        )

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        id = bat['speaker'].to(self.device)
        id = F.one_hot(id, self.num_classes)

        if self.expression:
            expression = bat['expression'].to(self.device).to(torch.float32)
            poses = torch.cat([poses, expression], dim=1)

        pred_poses, _ = self.generator(
            aud,
            poses,
            id,
        )

        G_loss, G_loss_dict = self.get_loss(
            pred_poses=pred_poses,
            gt_poses=poses,
            pre_poses=None,
            mode='training_G',
            gt_conf=None,
            aud=aud,
        )

        self.generator_optimizer.zero_grad()
        G_loss.backward()
        grad = torch.nn.utils.clip_grad_norm(self.generator.parameters(), self.config.Train.max_gradient_norm)
        loss_dict['grad'] = grad.item()
        self.generator_optimizer.step()

        for key in list(G_loss_dict.keys()):
            loss_dict[key] = G_loss_dict.get(key, 0).item()

        return total_loss, loss_dict

    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 pre_poses,
                 aud,
                 mode='training_G',
                 gt_conf=None,
                 exp=1,
                 gt_nzero=None,
                 pre_nzero=None,
                 ):
        loss_dict = {}


        [b_j, b_e, b_b, b_h, b_f] = self.dim_list

        MSELoss = torch.mean(torch.abs(pred_poses[:, :self.each_dim[0]] - gt_poses[:, :self.each_dim[0]]))
        if self.expression:
            expl = torch.mean((pred_poses[:, -self.each_dim[3]:] - gt_poses[:, -self.each_dim[3]:])**2)
        else:
            expl = 0

        gen_loss = expl + MSELoss

        loss_dict['MSELoss'] = MSELoss
        if self.expression:
            loss_dict['exp_loss'] = expl

        return gen_loss, loss_dict

    def infer_on_audio(self, aud_fn, B, frame, fm_dict, id=None, **kwargs):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        # assert self.args.infer, "train mode"
        self.generator.eval()

        if type(aud_fn) == torch.Tensor:
            aud_feat = torch.tensor(aud_fn, dtype=torch.float32).to(self.generator.device)
            num_poses_to_generate = aud_feat.shape[-1]
        else:
            aud_feat = get_mfcc_ta(aud_fn, fps=30, fm_dict=fm_dict, encoder_choice='faceformer')
            aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
            aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.generator.device).transpose(1, 2)
        if frame is None:
            frame = aud_feat.shape[2]*30//16000
        #
        if id is None:
            id = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=self.generator.device)
        elif type(id) == torch.Tensor:
            id = id.repeat(B)
            id = F.one_hot(id, self.num_classes).to(self.generator.device)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            pred_poses = self.generator(aud_feat, None, id, time_steps=frame)[0]
        output = pred_poses
        end = time.time()
        return output, end-start


    def generate(self, wv2_feat, frame):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        # assert self.args.infer, "train mode"
        self.generator.eval()

        B = 1

        id = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=self.generator.device)
        id = id.repeat(wv2_feat.shape[0], 1)

        with torch.no_grad():
            pred_poses = self.generator(wv2_feat, None, id, time_steps=frame)[0]
        return pred_poses
