import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
import math
from data_utils.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore warnings

# change_angle = torch.tensor([5.4062e-05, 4.6029e-05, 1.7887e-04, 1.8914e-04])

# change_angle = torch.tensor([1.7895e-04, 1.8926e-04, 8.1730e-05, 7.3042e-05])


class EmbeddingSpaceEvaluator:
    def __init__(self, face_ae, body_ae, full_ae, vae, device):

        # init embed net
        self.face_ae = face_ae
        self.body_ae = body_ae
        self.full_ae = full_ae
        # self.vae = vae

        # storage
        self.face_real_feat = []
        self.face_gene_feat = []
        self.body_real_feat = []
        self.body_gene_feat = []
        self.full_real_feat = []
        self.full_gene_feat = []

        self.real_joints_list = []
        self.generated_joints_list = []
        self.real_6d_list = []
        self.generated_6d_list = []
        self.audio_beat_list = []

    def reset(self):
        self.real_feat_list = []
        self.generated_feat_list = []

    def get_no_of_samples(self):
        return len(self.real_feat_list)

    def push_samples(self, generated_poses, real_poses):
        # convert poses to latent features
        generated_poses = generated_poses.float()
        real_poses = real_poses.float()

        real_feat, _ = self.face_ae.extract(real_poses)
        generated_feat, _ = self.face_ae.extract(generated_poses)
        real_feat = real_feat.squeeze()
        generated_feat = generated_feat.reshape(-1, generated_feat.shape[1])
        self.face_real_feat.append(real_feat.data.cpu().numpy())
        self.face_gene_feat.append(generated_feat.data.cpu().numpy())

        real_feat, _ = self.body_ae.extract(real_poses)
        generated_feat, _ = self.body_ae.extract(generated_poses)
        real_feat = real_feat.squeeze()
        generated_feat = generated_feat.reshape(-1, generated_feat.shape[1])
        self.body_real_feat.append(real_feat.data.cpu().numpy())
        self.body_gene_feat.append(generated_feat.data.cpu().numpy())

        real_feat, _ = self.full_ae.extract(real_poses)
        generated_feat, _ = self.full_ae.extract(generated_poses)
        real_feat = real_feat.squeeze()
        generated_feat = generated_feat.reshape(-1, generated_feat.shape[1])
        self.full_real_feat.append(real_feat.data.cpu().numpy())
        self.full_gene_feat.append(generated_feat.data.cpu().numpy())

    def push_joints(self, generated_poses, real_poses):
        self.real_joints_list.append(real_poses.data.cpu())
        self.generated_joints_list.append(generated_poses.squeeze().data.cpu())

    def push_aud(self, aud):
        self.audio_beat_list.append(aud.squeeze().data.cpu())

    def get_MAAC(self):
        ang_vel_list = []
        for real_joints in self.real_joints_list:
            # real_joints[:, 15:21] = real_joints[:, 16:22]
            # vec = real_joints[:, 15:21] - real_joints[:, 13:19]
            # inner_product = torch.einsum('kij,kij->ki', [vec[:, 2:], vec[:, :-2]])
            vec = real_joints[:, [16, 17, 20, 21, 9, 9, 18, 19]] - real_joints[:, [18, 19, 18, 19, 16, 17, 16, 17]]
            vec = F.normalize(vec, dim=-1)
            inner_product = torch.einsum('kij,kij->ki', [vec[:, [0, 1, 4, 5]], vec[:, [2, 3, 6, 7]]])
            inner_product = torch.clamp(inner_product, -1, 1, out=None)
            angle = torch.acos(inner_product) / math.pi
            ang_vel = (angle[1:] - angle[:-1]).abs().mean(dim=0)
            ang_vel_list.append(ang_vel.unsqueeze(dim=0))
        all_vel = torch.cat(ang_vel_list, dim=0)
        MAAC = all_vel.mean(dim=0)
        return MAAC

    def get_BCscore(self, thres=0.1, change_angle=torch.tensor([1.7895e-04, 1.8926e-04, 8.1730e-05, 7.3042e-05]), joints_list=None):
        thres = thres
        sigma = 0.1
        sum_1 = 0
        total_beat = 0
        for joints, audio_beat_time in zip(joints_list, self.audio_beat_list):
            joints = joints.clone()
            motion_beat_time = []
            if joints.dim() == 4:
                joints = joints[0]
            # joints[:, 15:21] = joints[:, 16:22]
            # vec = joints[:, 15:21] - joints[:, 13:19]
            vec = joints[:, [16,17,20,21,9,9,18,19]] - joints[:, [18,19,18,19,16,17,16,17]]
            vec = F.normalize(vec, dim=-1)
            inner_product = torch.einsum('kij,kij->ki', [vec[:, [0,1,4,5]], vec[:, [2,3,6,7]]])
            inner_product = torch.clamp(inner_product, -1, 1, out=None)
            angle = torch.acos(inner_product) / math.pi
            ang_vel = (angle[1:] - angle[:-1]).abs() / change_angle / len(change_angle)

            ang_vel = ang_vel.sum(-1)
            angle_diff = torch.cat((torch.zeros(1), ang_vel), dim=0)

            motion_beat_time = []
            for t in range(1, joints.shape[0] - 1):
                if angle_diff[t] < angle_diff[t - 1] and angle_diff[t] < angle_diff[t + 1]:
                    if angle_diff[t - 1] - angle_diff[t] >= thres or angle_diff[t + 1] - angle_diff[t] >= thres:
                        motion_beat_time.append(float(t) / 30.0)
            if len(motion_beat_time) != 0:
                motion_beat_time = torch.tensor(motion_beat_time)
                sum = 0
                for audio in audio_beat_time:
                    sum += np.power(math.e,
                                    -(np.power((audio.item() - motion_beat_time), 2)).min() / (2 * sigma * sigma))
                sum_1 = sum_1 + sum
            total_beat = total_beat + len(audio_beat_time)

            # angle_diff = torch.cat((torch.zeros(1, 4), ang_vel), dim=0)
            #
            # sum_2 = 0
            # for i in range(angle_diff.shape[1]):
            #     motion_beat_time = []
            #     for t in range(1, joints.shape[0] - 1):
            #         if angle_diff[t][i] < angle_diff[t - 1][i] and angle_diff[t][i] < angle_diff[t + 1][i]:
            #             if angle_diff[t - 1][i] - angle_diff[t][i] >= thres or angle_diff[t + 1][i] - angle_diff[t][i] >= thres:
            #                 motion_beat_time.append(float(t) / 30.0)
            #     if len(motion_beat_time) == 0:
            #         continue
            #     motion_beat_time = torch.tensor(motion_beat_time)
            #     sum = 0
            #     for audio in audio_beat_time:
            #         sum += np.power(math.e,
            #                         -(np.power((audio.item() - motion_beat_time), 2)).min() / (2 * sigma * sigma))
            #     sum_2 = sum_2 + sum
            #     total_beat = total_beat + len(audio_beat_time)
            # sum_1 = sum_1 + sum_2
        return sum_1 / total_beat

    def get_scores(self, type):

        if type == 'fe':
            gene_list = self.face_gene_feat
            real_list = self.face_real_feat
        elif type == 'bh':
            gene_list = self.body_gene_feat
            real_list = self.body_real_feat
        elif type == 'fbhe':
            gene_list = self.full_gene_feat
            real_list = self.full_real_feat
        else:
            raise TypeError


        generated_feats = np.vstack(gene_list)
        real_feats = np.vstack(real_list)

        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)
            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10
            return frechet_dist

        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        a = diff.dot(diff)
        b = np.trace(sigma1)
        c = np.trace(sigma2)
        d = tr_covmean
        print('diff=', a, '    trace_1=', b, '    trace_2=', c, '    tr_covmean=', d)

        return (a + b + c - 2 * d)
