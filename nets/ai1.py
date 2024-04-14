import torch
import torch.nn.functional as F
import time

import nets
from data_utils.consts import smplx_hyperparams

exp_dim = smplx_hyperparams['expression_dim']

def init_model(model_name, model_path, args, config):
    generator = getattr(nets, model_name)(args, config)
    print(model_path)
    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    generator.load_state_dict(model_ckpt['generator'])

    return generator


class All_In_One_Model:

    def __init__(self, face_name, face_path, body_name, body_path, device, args, config):
        self.args = args
        self.config = config
        self.convert_to_6d = config.Data.pose.convert_to_6d
        self.running_time = 0
        self.frame = 0
        self.init_params()
        if face_name is not None:
            self.face_model = init_model(face_name, face_path, args, config)
        else:
            self.face_model = None

        # config.Model.vq_type = 'bh'
        # config.Model.vq_path = 'experiments/2023-10-19-smplx_S2G-new_vqt_bh1024_size8_newpq128_d128_group4/ckpt-99.pth'

        if isinstance(body_name, list):
            # load hierarchical model
            self.body_model = []
            for name, path in zip(body_name, body_path):
                self.body_model.append(init_model(name, path, args, config))
        else:
            self.body_model = init_model(body_name, body_path, args, config)

    def __call__(self, forward_type='infer_on_audio', target_frame=0, result_format='one', **kwargs):
        cost_time = 0
        if forward_type == 'infer_on_vq':
            for model in self.body_model:
                ce_loss = getattr(model, forward_type)(**kwargs)
                return ce_loss

        if isinstance(self.body_model, list):
            body_list = []
            body = None
            for model in self.body_model:
                body, time_0 = getattr(model, forward_type)(**kwargs, target_frame=target_frame, pred_poses=body)
                body_list.append(body)
                cost_time = cost_time + time_0
        else:
            body, time_0 = getattr(self.body_model, forward_type)(**kwargs)
            cost_time = cost_time + time_0

        if self.face_model is not None:
            face, time_0 = getattr(self.face_model, forward_type)(**kwargs)
            cost_time = cost_time + time_0
        else:
            face = None

        if face is not None:
            jaw = face[:, :self.each_dim[0]]
            exp = face[:, -self.each_dim[3]:]
            if body.shape[2] < face.shape[2]:
                body = F.interpolate(body, size=face.shape[2], align_corners=False, mode='linear')

            if body.shape[1] == self.full_dim:
                holistic_body = torch.cat([jaw, body[:, self.each_dim[0]:-self.each_dim[3]], exp], dim=1)
            else:
                holistic_body = torch.cat([jaw, body, exp], dim=1)
        else:
            holistic_body = body

        if body.shape[2] < target_frame:
            holistic_body = F.interpolate(holistic_body, size=target_frame, align_corners=False, mode='linear')

        if result_format == 'one':
            return holistic_body, cost_time
        else:
            if body_list[0].shape[2] < target_frame:
                body_list[0] = F.interpolate(body_list[0], size=target_frame, align_corners=False, mode='linear')
            return body_list, cost_time

    def init_params(self):
        if self.convert_to_6d:
            scale = 2
        else:
            scale = 1

        global_orient = round(0 * scale)
        leye_pose = reye_pose = round(0 * scale)
        jaw_pose = round((3) * scale)
        body_pose = round((45) * scale)
        left_hand_pose = right_hand_pose = round((45) * scale)
        expression = exp_dim

        b_j = 0
        jaw_dim = jaw_pose
        b_e = b_j + jaw_dim
        eye_dim = leye_pose + reye_pose
        b_b = b_e + eye_dim
        body_dim = global_orient + body_pose
        b_h = b_b + body_dim
        hand_dim = left_hand_pose + right_hand_pose
        b_f = b_h + hand_dim
        face_dim = expression
        # f = 'f' in self.type
        # b = 'b' in self.type
        # h = 'h' in self.type
        # e = 'e' in self.type

        self.dim_list = [b_j, b_e, b_b, b_h, b_f]
        self.full_dim = jaw_dim * 1 + (eye_dim + body_dim) * 1 + hand_dim * 1 + face_dim * 1
        self.pose = int(self.full_dim / round(3 * scale))
        self.each_dim = [jaw_dim, eye_dim + body_dim, hand_dim, face_dim]
