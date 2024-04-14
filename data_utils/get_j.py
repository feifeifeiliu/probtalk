import torch

from data_utils.rotation_conversion import matrix_to_axis_angle, rotation_6d_to_matrix
from data_utils.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d
from data_utils.consts import smplx_hyperparams

exp_dim = smplx_hyperparams['expression_dim']

def to6d(poses, config):
    bs, t, _ = poses.shape

    poses_exp = poses[..., -exp_dim:] if config.Data.pose.expression else None
    poses = poses[..., :-exp_dim] if config.Data.pose.expression else poses

    poses = poses.reshape(-1, 3)
    poses = matrix_to_rotation_6d(axis_angle_to_matrix(poses)).reshape(bs, t, -1)

    poses = torch.cat([poses, poses_exp], -1) if config.Data.pose.expression else poses
    return poses


def to3d(poses, config):
    bs, t, _ = poses.shape

    poses_exp = poses[..., -exp_dim:] if config.Data.pose.expression else None
    poses = poses[..., :-exp_dim] if config.Data.pose.expression else poses

    poses = poses.reshape(-1, 6)
    poses = matrix_to_axis_angle(rotation_6d_to_matrix(poses)).reshape(bs, t, -1)

    poses = torch.cat([poses, poses_exp], -1) if config.Data.pose.expression else poses
    return poses


def get_joint(smplx_model, betas, pred):
    joint = smplx_model(betas=betas.repeat(pred.shape[0], 1),
                        expression=pred[:, 165:(165+exp_dim)],
                        jaw_pose=pred[:, 0:3],
                        leye_pose=pred[:, 3:6],
                        reye_pose=pred[:, 6:9],
                        global_orient=pred[:, 9:12],
                        body_pose=pred[:, 12:75],
                        left_hand_pose=pred[:, 75:120],
                        right_hand_pose=pred[:, 120:165],
                        return_verts=True)['joints']
    return joint


def get_joints(smplx_model, betas, pred, bat=4):
    if len(pred.shape) == 3:
        B = pred.shape[0]
        x = bat if B>= bat else B
        T = pred.shape[1]
        pred = pred.reshape(-1, 165+exp_dim)
        smplx_model.batch_size = L = T * x

        times = pred.shape[0] // smplx_model.batch_size
        joints = []
        for i in range(times):
            joints.append(get_joint(smplx_model, betas, pred[i*L:(i+1)*L]))
        joints = torch.cat(joints, dim=0)
        joints = joints.reshape(B, T, -1, 3)
    else:
        smplx_model.batch_size = pred.shape[0]
        joints = get_joint(smplx_model, betas, pred)
    return joints