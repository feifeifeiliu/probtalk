import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.consts import get_speaker_id, smplx_hyperparams


exp_dim = smplx_hyperparams['expression_dim']


class TrainWrapperBaseClass():
    def __init__(self, args, config) -> None:
        self.init_optimizer()
        self.num_classes = get_speaker_id(config.Data.data_root).__len__()

    def init_optimizer(self) -> None:
        print('using Adam')
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr = self.config.Train.learning_rate.generator_learning_rate,
            betas=[0.9, 0.999]
        )
        if self.discriminator is not None:
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr = self.config.Train.learning_rate.discriminator_learning_rate,
                betas=[0.9, 0.999]
            )

    def to_parallel(self):
        pass

    def __call__(self, bat):
        raise NotImplementedError

    def get_loss(self, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        model_state = {
            'generator': self.generator.state_dict(),
            'generator_optim': self.generator_optimizer.state_dict(),
            'discriminator': self.discriminator.state_dict() if self.discriminator is not None else None,
            'discriminator_optim': self.discriminator_optimizer.state_dict() if self.discriminator is not None else None
        }
        return model_state

    # def parameters(self):
    #     return self.generator.parameters()

    def load_state_dict(self, state_dict):
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

    def infer_on_audio(self, aud_fn, initial_pose=None, norm_stats=None, **kwargs):
        raise NotImplementedError

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

    def getFM_dim(self, audio_name, text_name):
        if audio_name == 'hubert':
            aud_d = 1024
        elif audio_name == 'wav2vec':
            aud_d = 768
        elif audio_name == 'speech2text':
            aud_d = 768
        elif audio_name == None:
            aud_d = 768
        else:
            raise NameError("The audio model name is incorrect.")

        if text_name == 'gpt2':
            text_d = 768
        elif text_name == 'clip':
            text_d = 512
        elif text_name == 'bert':
            text_d = 768
        elif text_name == 't5':
            text_d = 768
        elif text_name == 'ton':
            text_d = 3
        elif text_name == 'fasttext':
            raise NotImplementedError
        elif text_name == None:
            text_d = 768
        else:
            raise NameError("The text model name is incorrect.")
        return aud_d, text_d
