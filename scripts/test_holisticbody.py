import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())

from tqdm import tqdm
import torch
from torch.utils import data
import numpy as np
import smplx as smpl
import math
from transformers import Wav2Vec2Processor
import time

from data_utils.lower_body import part2full, poses2pred, c_index_6d, c_index_3d
from data_utils.get_j import to3d, get_joints, to6d
from data_utils import torch_data, get_mfcc_ta
from nets.ai1 import All_In_One_Model
from nets import denormalize
from nets.init_model import init_model
from nets.utils import tofbhe
from trainer.options import parse_args
from trainer.config import load_JsonConfig, load_YmlConfig
from evaluation.FGD import EmbeddingSpaceEvaluator
from evaluation.metrics import LVD
from data_utils.consts import smplx_hyperparams
from data_utils.foundation_models import getFM

betas_dim = smplx_hyperparams['betas_dim']
exp_dim = smplx_hyperparams['expression_dim']

face_tvar = torch.tensor(0.00070697901537641883)
face_tsum = torch.tensor(0.00109510007314383984)
body_tvar = torch.tensor(0.98894238471984863281)
body_tsum = torch.tensor(1.60230898857116699219)

def init_dataloader(data_root, speakers, args, config):
    data_base = torch_data(
        data_root=data_root,
        speakers=speakers,
        split='test',
        limbscaling=False,
        normalization=config.Data.pose.normalization,
        norm_method=config.Data.pose.norm_method,
        split_trans_zero=False,
        num_pre_frames=config.Data.pose.pre_pose_length,
        num_generate_length=config.Data.pose.generate_length,
        num_frames=30,
        aud_feat_win_size=config.Data.aud.aud_feat_win_size,
        aud_feat_dim=config.Data.aud.aud_feat_dim,
        feat_method=config.Data.aud.feat_method,
        smplx=True,
        audio_sr=22000,
        convert_to_6d=config.Data.pose.convert_to_6d,
        expression=config.Data.pose.expression,
        config=config
    )

    if config.Data.pose.normalization:
        # norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats_fn = 'data_utils/norm_stats.npy'
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
        data_base.data_mean = norm_stats[0]
        data_base.data_std = norm_stats[1]
    else:
        norm_stats = [0, 0]

    data_base.get_dataset(norm_stats)
    test_set = data_base.all_dataset
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False)

    return test_set, test_loader, norm_stats


face_joint_ind = torch.cat([torch.arange(22, 23, 1), torch.arange(74, 127, 1)])


def face_loss(gt, prs, loss_dict):
    jaw_xyz = gt[:, 22:23, :] - prs[:, 22:23, :]
    jaw_dist = jaw_xyz.norm(p=2, dim=-1)
    jaw_dist = jaw_dist.sum(dim=-1).mean()
    landmark_xyz = gt[:, 74:] - prs[:, 74:]
    landmark_dist = landmark_xyz.norm(p=2, dim=-1)
    landmark_dist = landmark_dist.sum(dim=-1).mean()
    loss_dict['face_L2'] = landmark_dist + jaw_dist

    face_gt = torch.cat([gt[:, 22:25], gt[:, 74:]], dim=1)
    face_pr = torch.cat([prs[:, 22:25], prs[:, 74:]], dim=1)
    loss_dict['face_LVD'] = LVD(face_gt, face_pr)

    # var = prs[:, :, face_joint_ind].var(dim=0).norm(p=2, dim=-1).sum(dim=-1).mean()
    # loss_dict['face_diverse'] = var

    return loss_dict


def face_loss_diverse(gt, prs, loss_dict):
    jaw_dist = (gt[:, 22:23, :] - prs[:, :, 22:23]).norm(p=2, dim=-1).sum(dim=-1).mean()
    landmark_dist = (gt[:, 74:] - prs[:, :, 74:]).norm(p=2, dim=-1).sum(dim=-1).mean()
    loss_dict['face_L2'] = landmark_dist + jaw_dist

    face_gt = torch.cat([gt[:, 22:25], gt[:, 74:]], dim=1)
    face_pr = torch.cat([prs[:, :, 22:25], prs[:, :, 74:]], dim=2)
    loss_dict['face_LVD'] = LVD(face_gt, face_pr)

    var = prs[:, :, face_joint_ind].var(dim=0).norm(p=2, dim=-1).sum(dim=-1).mean()
    loss_dict['face_diverse'] = var/face_tvar.item()

    return loss_dict


joint_ind = torch.cat([torch.arange(0, 22, 1), torch.arange(23, 74, 1)])


def body_loss(gt, prs, poses, pred):
    loss_dict = {}

    # LVD
    v_diff = LVD(gt[:, :22], prs[:, :, :22])
    loss_dict['MAD'] = v_diff
    # Accuracy
    error = (gt[:, :22] - prs[:, :, :22]).norm(p=2, dim=-1).sum(dim=-1).mean()
    loss_dict['MAJE'] = error
    # Diversity
    var = prs[:, :, :22].var(dim=0).norm(p=2, dim=-1).sum(dim=-1).mean()
    loss_dict['diverse'] = var

    # LVD
    v_diff = LVD(gt[:, joint_ind], prs[:, :, joint_ind])
    loss_dict['MAD_full'] = v_diff
    # Accuracy
    error = (gt[:, joint_ind] - prs[:, :, joint_ind]).norm(p=2, dim=-1).sum(dim=-1).mean()
    loss_dict['MAJE_full'] = error
    # Diversity
    var = prs[:, :, joint_ind].var(dim=0).norm(p=2, dim=-1).sum(dim=-1).mean()
    loss_dict['diverse_full'] = var/body_tvar.item()

    return loss_dict

    # # LVD 6d
    # gt_velocity = (poses[:, 1:] - poses[:, :-1]).abs()
    # pr_velocity = (pred[:, 1:] - pred[:, :-1]).abs()
    # loss_dict['MAD_6d'] = ((pr_velocity - gt_velocity).abs().mean())
    # # Accuracy 6d
    # error = (poses - pred).abs().mean()
    # loss_dict['MAJE_6d'] = error
    # # Diversity 6d
    # var = pred.var(dim=0).abs().mean()
    # loss_dict['diverse_6d'] = var
    #
    # # LVD 6d
    # gt_velocity = (poses[:, 1:, :6] - poses[:, :-1, :6]).abs()
    # pr_velocity = (pred[:, 1:, :6] - pred[:, :-1, :6]).abs()
    # loss_dict['MAD_face_6d'] = ((pr_velocity - gt_velocity).abs().mean())
    # gt_velocity = (poses[:, 1:, -exp_dim:] - poses[:, :-1, -exp_dim:]).abs()
    # pr_velocity = (pred[:, 1:, -exp_dim:] - pred[:, :-1, -exp_dim:]).abs()
    # loss_dict['MAD_face_6d'] = ((pr_velocity - gt_velocity).abs().mean()) + loss_dict['MAD_face_6d']
    # # Accuracy jaw
    # error_jaw = (poses[..., :6] - pred[..., :6]).abs().mean()
    # error_exp = (poses[..., -exp_dim:] - pred[..., -exp_dim:]).abs().mean()
    # loss_dict['MAJE_face_6d'] = error_jaw + error_exp
    # return loss_dict


def test(test_loader, norm_stats, generator, FGD_handler, smplx_model, config):
    print('start testing')

    fm_dict = getFM(config.Data.audio, config.Data.text)
    fm_dict['sr'] = 16000
    # fm_dict = None

    body3d_list = []
    face3d_list = []

    if norm_stats is not None:
        norm_stats_torch = []
        norm_stats_torch.append(torch.from_numpy(norm_stats[0]).to('cuda'))
        norm_stats_torch.append(torch.from_numpy(norm_stats[1]).to('cuda'))
        norm_stats = norm_stats_torch
    c_index = c_index_6d if config.Data.pose.convert_to_6d else c_index_3d
    betas = torch.zeros([1, betas_dim]).to('cuda').to(torch.float32)

    inpaint = False
    test_body = False
    test_face = False
    jd = 6 if config.Data.pose.convert_to_6d else 3
    ed = exp_dim

    loss_dict = {}
    B = 16
    total_time = 0
    total_frame = 0

    # import pickle
    # file = open('g_list.pickle', 'rb')
    # FGD_handler.generated_joints_list = pickle.load(file)
    # file.close()
    # file = open('a_beat.pickle', 'rb')
    # FGD_handler.audio_beat_list = pickle.load(file)
    # file.close()

    with torch.no_grad():
        count = 0
        for bat in tqdm(test_loader, desc="Testing......"):
            # if count == 1000:
            #     break
            count = count + 1

            ### get data
            _, poses, exp = bat['aud_feat'].to('cuda').to(torch.float32), bat['poses'].to('cuda').to(torch.float32), \
                bat['expression'].to('cuda').to(torch.float32)
            aud = bat['aud_feat'].to('cuda').to(torch.float32)
            text = bat['text_feat'].to('cuda').to(torch.float32)
            id = bat['speaker'].to('cuda')
            cur_wav_file = bat['aud_file'][0]
            jaw = poses[:, :int(c_index.size / 43)]
            gt_poses = torch.cat([jaw, poses[:, c_index], exp], dim=1)
            poses = torch.cat([poses, exp], dim=1)

            if inpaint:
                mask_probability = torch.rand([poses.shape[-1] - 29])
                mask_indice = torch.argmax(mask_probability)
                mask = torch.zeros_like(mask_probability)
                mask[mask_indice] = 1
                mask = torch.cat([mask[:mask_indice], torch.ones([29]), mask[mask_indice:]])
            else:
                mask = torch.zeros(poses.shape[-1])
            mask = mask.reshape(1, 1, -1).repeat(B, 1, 1).to(poses.device)

            ### predict poses
            # pred, cost_time = generator(forward_type='infer_on_audio',
            #                             aud=aud, text=text, gt_poses=gt_poses, id=id, B=B, mask=mask,
            #                             target_frame=poses.shape[2], fm_dict=fm_dict,
            #                             aud_fn=bat['aud_file'][0], fps=30, frame=poses.shape[-1])



            pred = None
            num_slices = 1 + math.ceil((aud.shape[-1] - 180) / 150)
            cost_time = 0
            input_gt = gt_poses.clone().repeat(B, 1, 1)
            input_mask = mask.clone()
            for i in range(num_slices):
                slice_start = 0 if i == 0 else 150 + 180 * (i - 1)
                slice_end = 180 if i == 0 else 150 + 180 * i

                pred_0, time_x = generator(forward_type='infer_on_batch',
                                           aud=aud[..., slice_start:slice_end],
                                           text=text[..., slice_start:slice_end],
                                           gt_poses=input_gt[..., slice_start:slice_end],
                                           mask=input_mask[..., slice_start:slice_end],
                                           id=id, B=B)
                if pred is None:
                    pred = pred_0
                else:
                    pred = torch.cat([pred, pred_0[..., 30:]], -1)
                input_gt[..., slice_start:slice_end] = pred_0
                input_mask[..., slice_start:slice_end] = 1
                cost_time = cost_time + time_x

            pred = gt_poses * mask + pred * (1 - mask)

            total_time = total_time + cost_time
            total_frame = total_frame + poses.shape[-1]

            # poses = poses[..., 30:]
            # pred = pred[..., 30:]
            if test_body:
                if pred.shape[1] == 270:
                    pred = torch.cat([poses[:, :jd].repeat(B, 1, 1), pred, poses[:, -ed:].repeat(B, 1, 1)], dim=1)
                else:
                    pred = torch.cat([poses[:, :jd].repeat(B, 1, 1), pred[:, jd:-ed], poses[:, -ed:].repeat(B, 1, 1)],
                                     dim=1)
            if test_face:
                if pred.shape[1] == 106:
                    pred = torch.cat([pred[:, :jd], poses[:, c_index].repeat(B, 1, 1), pred[:, -ed:]], dim=1)

            FGD_handler.push_samples(pred[:].transpose(1, 2).unfold(1, 90, 90).flatten(0, 1),
                                     gt_poses[0:1].transpose(1, 2).unfold(1, 90, 90).flatten(0, 1))

            if config.Data.pose.normalization:
                poses = denormalize(poses, norm_stats[0], norm_stats[1], 'all', c_index)
                pred = denormalize(pred, norm_stats[0], norm_stats[1], 'fbhe', c_index)
            if pred.shape[1] == 270:
                pad = torch.zeros([B, 1, poses.shape[2]], device=pred.device)
                pred = torch.cat([pad.repeat(1, 6, 1), pred, pad.repeat(1, exp_dim, 1)], dim=1)
            if not config.Data.pose.convert_to_6d:
                poses = to6d(poses.transpose(1, 2), config).transpose(1, 2)
                pred = to6d(pred.transpose(1, 2), config).transpose(1, 2)
            poses_3d = to3d(poses.transpose(1, 2), config)
            pred_3d = to3d(pred.transpose(1, 2), config)

            # FGD_handler.push_samples(pred_3d[0:1, :, 3:-100].transpose(1, 2), tofbhe(poses_3d.transpose(1, 2), c_index_3d)[:, 3:-100])

            ### process data to get 3D joints
            full_pred = []
            for j in range(B):
                f_pred = part2full(pred_3d[j])
                full_pred.append(f_pred)
            for i in range(full_pred.__len__()):
                full_pred[i] = full_pred[i].unsqueeze(dim=0)
            full_pred = torch.cat(full_pred, dim=0)
            poses_3d = poses2pred(poses_3d.squeeze())
            pred_joints = get_joints(smplx_model, betas, full_pred)
            gt_joints = get_joints(smplx_model, betas, poses_3d)

            body3d_list.append(gt_joints.cpu()[:, joint_ind])

            # FGD_handler.push_samples(pred_joints[0:1].flatten(2).unfold(1, 90, 90).flatten(0, 1),
            #                          gt_joints.unsqueeze(0).flatten(2).unfold(1, 90, 90).flatten(0, 1))

            # a = gt_joints.unsqueeze(0).flatten(2).unfold(1, 90, 90).flatten(0, 1)
            # FGD_handler.push_samples(a+0.1, a)

            ### calculate losses

            FGD_handler.push_joints(pred_joints, gt_joints)
            aud = get_mfcc_ta(cur_wav_file, fps=30, sr=16000, fm_dict=None, encoder_choice='onset')
            FGD_handler.push_aud(torch.from_numpy(aud))

            bat_loss_dict = body_loss(gt_joints, pred_joints, tofbhe(poses, c_index_6d).transpose(1, 2),
                                      pred.transpose(1, 2))

            # zero_poses = torch.zeros([poses_3d.shape[0], 162], device='cuda')
            #
            # full_pred = full_pred[0]
            # pred_face_param = torch.cat([full_pred[:, :3], zero_poses, full_pred[:, -exp_dim:]], dim=-1)
            # poses_3d[:, 3:165] = pred_face_param[:, 3:165]
            # gt_joints = get_joints(smplx_model, betas, poses_3d)
            # pred_joints = get_joints(smplx_model, betas, pred_face_param)
            # bat_loss_dict = face_loss(gt_joints, pred_joints, bat_loss_dict)

            zero_poses = torch.zeros([B, poses_3d.shape[0], 162], device='cuda')
            pred_face_param = torch.cat([full_pred[..., :3], zero_poses, full_pred[..., -exp_dim:]], dim=-1)
            poses_3d[..., 3:165] = zero_poses[0]
            gt_joints = get_joints(smplx_model, betas, poses_3d)
            pred_joints = get_joints(smplx_model, betas, pred_face_param)
            bat_loss_dict = face_loss_diverse(gt_joints, pred_joints, bat_loss_dict)

            face3d_list.append(gt_joints.cpu()[:, face_joint_ind])

            if loss_dict:  # 非空
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] += bat_loss_dict[key]
            else:
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] = bat_loss_dict[key]
        print('maskgit')
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] / count
            print(key + '=' + str(loss_dict[key].item()))

        MAAC = FGD_handler.get_MAAC()
        print(MAAC)
        # for thres in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
        #     BC_score = FGD_handler.get_BCscore(thres, MAAC, FGD_handler.real_joints_list)
        #     print('GT BC_score with thres {} = {}'.format(thres, BC_score))
        for thres in [0.01]:
            BC_score = FGD_handler.get_BCscore(thres, MAAC, FGD_handler.generated_joints_list)
            print('Generated BC_score with thres {} = {}'.format(thres, BC_score))
        fgd_dist, feat_dist = FGD_handler.get_scores('fe')
        print('face_fgd=', fgd_dist.item())
        fgd_dist, feat_dist = FGD_handler.get_scores('bh')
        print('body_fgd=', fgd_dist.item())
        fgd_dist, feat_dist = FGD_handler.get_scores('fbhe')
        print('full_fgd=', fgd_dist.item())
        # print('feat_dist=', feat_dist.item())
        print('fps=', total_frame / total_time)


def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    file_type = args.config_file.split('.')[-1]
    if file_type == 'json':
        config = load_JsonConfig(args.config_file)
    elif file_type == 'yml':
        config = load_YmlConfig(args.config_file)

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    print('init dataloader...')
    test_set, test_loader, norm_stats = init_dataloader(config.Data.data_root, args.speakers, args, config)
    print('init model...')
    generator = All_In_One_Model(args.face_model_name, args.face_model_path,
                                 args.body_model_name, args.body_model_path,
                                 device, args, config)

    config_emb = config
    config_emb.Model.vq_type = 'fe'
    face_ae = init_model('emb_net', args, config_emb, True,
                         './experiments/val_models/val_face.pth')
    config_emb.Model.vq_type = 'bh'
    body_ae = init_model('emb_net', args, config_emb, True,
                         './experiments/val_models/val_body.pth')
    config_emb.Model.vq_type = 'fbhe'
    full_ae = init_model('emb_net', args, config_emb, True, './experiments/val_models/val.pth')
    FGD_handler = EmbeddingSpaceEvaluator(face_ae, body_ae, full_ae, None, 'cuda')
    # FGD_handler = None

    print('init smlpx model...')
    dtype = torch.float32
    smplx_path = './visualise/'
    model_params = dict(model_path=smplx_path,
                        model_type='smplx',
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        num_betas=betas_dim,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        use_pca=False,
                        flat_hand_mean=False,
                        create_expression=True,
                        num_expression_coeffs=exp_dim,
                        num_pca_comps=12,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype, )

    smplx_model = smpl.create(**model_params).to('cuda')

    test(test_loader, norm_stats, generator, FGD_handler, smplx_model, config)


if __name__ == '__main__':
    main()
