import os
import sys
import math

from data_utils.foundation_models import getFM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
sys.path.append(os.getcwd())

from transformers import Wav2Vec2Processor
from glob import glob

import numpy as np
import json
import smplx as smpl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import time
from transformers import AutoProcessor

from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
from data_utils.lower_body import part2full, pred2poses, poses2pred, poses2poses
from visualise.rendering import RenderTool
from data_utils.lower_body import c_index_3d, c_index_6d
from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig
from nets.spg.wav2vec import Wav2Vec2Model
from nets.ai1 import All_In_One_Model
from nets.utils import tofbhe
from data_utils.consts import smplx_hyperparams

betas_dim = smplx_hyperparams['betas_dim']
exp_dim = smplx_hyperparams['expression_dim']


def init_dataloader(data_root, speakers, args, config):
    if data_root.endswith('.csv'):
        raise NotImplementedError
    else:
        data_class = torch_data
    if 'smplx' in config.Model.model_name or 's2g' in config.Model.model_name:
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
    else:
        data_base = torch_data(
            data_root=data_root,
            speakers=speakers,
            split='val',
            limbscaling=False,
            normalization=config.Data.pose.normalization,
            norm_method=config.Data.pose.norm_method,
            split_trans_zero=False,
            num_pre_frames=config.Data.pose.pre_pose_length,
            aud_feat_win_size=config.Data.aud.aud_feat_win_size,
            aud_feat_dim=config.Data.aud.aud_feat_dim,
            feat_method=config.Data.aud.feat_method
        )
    if config.Data.pose.normalization:
        norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
        data_base.data_mean = norm_stats[0]
        data_base.data_std = norm_stats[1]
    else:
        norm_stats = None

    data_base.get_dataset(norm_stats)
    infer_set = data_base.all_dataset
    infer_loader = data.DataLoader(data_base.all_dataset, batch_size=1, shuffle=False)

    return infer_set, infer_loader, norm_stats


def get_vertices(smplx_model, betas, result_list, exp, require_pose=False):
    vertices_list = []
    poses_list = []
    expression = torch.zeros([1, 50])

    for i in result_list:
        vertices = []
        poses = []
        for j in range(i.shape[0]):
            output = smplx_model(betas=betas,
                                 expression=i[j][165:(165 + exp_dim)].unsqueeze_(dim=0) if exp else expression,
                                 jaw_pose=i[j][0:3].unsqueeze_(dim=0),
                                 leye_pose=i[j][3:6].unsqueeze_(dim=0),
                                 reye_pose=i[j][6:9].unsqueeze_(dim=0),
                                 global_orient=i[j][9:12].unsqueeze_(dim=0),
                                 body_pose=i[j][12:75].unsqueeze_(dim=0),
                                 left_hand_pose=i[j][75:120].unsqueeze_(dim=0),
                                 right_hand_pose=i[j][120:165].unsqueeze_(dim=0),
                                 return_verts=True)
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
            # pose = torch.cat([output.body_pose, output.left_hand_pose, output.right_hand_pose], dim=1)
            pose = output.joints
            poses.append(pose.detach().cpu())
        vertices = np.asarray(vertices)
        vertices_list.append(vertices)
        poses = torch.cat(poses, dim=0)
        poses_list.append(poses)
    if require_pose:
        return vertices_list, poses_list
    else:
        return vertices_list, None


global_orient = torch.tensor([3.0747, -0.0158, -0.0152])


def infer(data_root, generator, infer_loader, device, norm_stats, smplx_model, rendertool, args=None, config=None):
    num_sample = 32
    face = False
    stand = False
    shape = 'fbhe'
    cover_length = 30
    j = 0
    gt_0 = None
    if norm_stats is not None:
        norm_stats_torch = []
        norm_stats_torch.append(torch.from_numpy(norm_stats[0]).to('cuda'))
        norm_stats_torch.append(torch.from_numpy(norm_stats[1]).to('cuda'))
        norm_stats = norm_stats_torch

    fm_dict = getFM(config.Data.audio, config.Data.text)
    fm_dict['sr'] = 16000

    all_list = {}
    for bat in infer_loader:
        with (torch.no_grad()):
            poses_ = bat['poses'].to(torch.float32).to(device)
            if poses_.shape[-1] == 300:
                j = j + 1
                if j > 5:
                    continue
                if config.Data.pose.expression:
                    expression = bat['expression'].to(device).to(torch.float32)
                    poses = torch.cat([poses_, expression], dim=1)
                else:
                    poses = poses_
                cur_wav_file = bat['aud_file'][0]

                # if cur_wav_file.split('\\')[-1].split('.')[0] != '214307-00_05_48-00_05_58':
                #     continue
                betas = bat['betas'][0].to(torch.float32).to('cuda')
                aud = bat['aud_feat'].to('cuda').to(torch.float32)
                text = bat['text_feat'].to('cuda').to(torch.float32)
                id = bat['speaker'].to('cuda')

                # betas = torch.zeros([1, 300], dtype=torch.float64).to('cuda')
                gt = poses.to('cuda')
                c_index = c_index_6d if config.Data.pose.convert_to_6d else c_index_3d
                if shape == 'fbhe':
                    gt_poses = tofbhe(gt, c_index)
                else:
                    gt_poses = gt[:, c_index, :]
                bs, n, t = gt_poses.size()
                mask_head = torch.ones([bs, 1, 1], device='cuda') * 0
                mask_head = torch.bernoulli(mask_head).repeat(1, 1, 30)
                mask_body = torch.ones([bs, 1, t - 60], device='cuda') * 0
                mask_body = torch.bernoulli(mask_body)
                mask = torch.cat([mask_head, mask_body, mask_head], dim=2)

                if config.Data.pose.normalization:
                    gt = denormalize(gt, norm_stats[0], norm_stats[1])

                gt = gt.squeeze(0).transpose(0, 1)
                if config.Data.pose.convert_to_6d:
                    if config.Data.pose.expression:
                        gt_exp = gt[:, -exp_dim:]
                        gt = gt[:, :-exp_dim]

                    gt = gt.reshape(gt.shape[0], -1, 6)

                    gt = matrix_to_axis_angle(rotation_6d_to_matrix(gt)).reshape(gt.shape[0], -1)
                    gt = torch.cat([gt, gt_exp], -1)

                result_list = [gt.to('cuda')]

                # for i in range(num_sample):
                #     # if i == 0:
                #     # torch.cuda.synchronize()
                #     # start = time.time()
                #     pred = None
                #     l_pslice = 180 - cover_length
                #     num_slices = 1 + math.ceil((aud.shape[-1] - 180) / l_pslice)
                #     cost_time = 0
                #     input_gt = gt_poses.clone().repeat(num_sample, 1, 1)
                #     input_mask = mask.clone().repeat(num_sample, 1, 1)
                #
                #     # pred, cost_time = generator(forward_type='infer_on_audio',
                #     #                             aud=aud, text=text, gt_poses=gt_poses, id=id, B=1, mask=mask,
                #     #                             target_frame=poses.shape[2], fm_dict=fm_dict,
                #     #                             aud_fn=bat['aud_file'][0], fps=30, frame=poses.shape[-1],
                #     #                             result_format='one')
                #
                #     for i in range(num_slices):
                #         slice_start = 0 if i == 0 else l_pslice + 180 * (i - 1)
                #         slice_end = 180 if i == 0 else l_pslice + 180 * i
                #
                #         pred_0, time_x = generator(forward_type='infer_on_batch',
                #                                    aud=aud[..., slice_start:slice_end],
                #                                    text=text[..., slice_start:slice_end],
                #                                    gt_poses=input_gt[..., slice_start:slice_end],
                #                                    mask=input_mask[..., slice_start:slice_end],
                #                                    id=id, B=num_sample)
                #         if pred is None:
                #             pred = pred_0
                #         else:
                #             pred = torch.cat([pred, pred_0[..., cover_length:]], -1)
                #         input_gt[..., slice_start:slice_end] = pred_0
                #         input_mask[..., slice_start:slice_end] = 1
                #         cost_time = cost_time + time_x
                #     # torch.cuda.synchronize()
                #     # end = time.time()
                #     print(cost_time)
                #     # else:
                #     #     pred = pred_poses
                #     if config.Data.pose.normalization:
                #         pred = denormalize(pred, norm_stats[0], norm_stats[1], shape, c_index).squeeze(dim=0)
                #     pred = pred.squeeze().to('cpu')
                #
                #     pred = pred.transpose(0, 1)
                #     if pred.shape[1] == 376:
                #         pred_face = pred[:, -exp_dim:]
                #         pred = pred[:, :-exp_dim]
                #     if config.Data.pose.convert_to_6d:
                #         pred = pred.reshape(pred.shape[0], -1, 6)
                #         pred = matrix_to_axis_angle(rotation_6d_to_matrix(pred))
                #         pred = pred.reshape(pred.shape[0], -1)
                #     if pred.shape[1] == 138:
                #         pred = torch.cat([pred, pred_face[:pred.shape[0]]], dim=-1)
                #     # else:
                #     #     pad = torch.zeros([pred.shape[0], 1], device=pred.device)
                #     #     pred = torch.cat([pad.repeat(1, 3), pred, pad.repeat(1, exp_dim)], dim=1)
                #
                #     # pred[:, 9:12] = global_orient
                #     pred = part2full(pred, stand)
                #     result_list[0] = poses2poses(result_list[0], pred.to('cuda'))
                #     # pred = pred2poses(pred, gt.to('cpu'))
                #     # result_list[0] = poses2pred(result_list[0], stand)
                #     # if gt_0 is None:
                #     #     gt_0 = gt
                #     # pred = pred2poses(pred, gt_0)
                #     # result_list[0] = poses2poses(result_list[0], gt_0)
                #
                #     if face:
                #         pad = torch.zeros([pred.shape[0], 162], device=pred.device)
                #         pred = torch.cat([pred[:, :3], pad, pred[:, -exp_dim:]], dim=1)
                #         pred = poses2poses(pred, result_list[0].cpu())
                #
                #     result_list.append(pred.to('cuda'))
                #     # all_list[cur_wav_file] = [res.to('cpu').numpy() for res in result_list]

                t = aud.shape[-1]

                pred_all = None
                l_pslice = 180
                num_slices = math.ceil((t - 30) / (l_pslice - cover_length))
                cost_time = 0
                cost_time_all = 0
                input_gt = gt_poses.clone().repeat(num_sample, 1, 1)
                input_mask = mask.clone().repeat(num_sample, 1, 1)
                code_list = []

                # pred_all, cost_time = generator(forward_type='infer_on_audio',
                #                             aud=aud, text=text, gt_poses=gt_poses, id=id, B=num_sample, mask=mask,
                #                             target_frame=poses.shape[2], fm_dict=fm_dict,
                #                             aud_fn=bat['aud_file'][0], fps=30, frame=poses.shape[-1],
                #                             result_format='one')

                torch.cuda.synchronize()
                start_time = time.time()
                for i in range(num_slices):
                    slice_start = (l_pslice - cover_length) * i
                    slice_end = slice_start + l_pslice
                    pred_0, codes, time_x = generator.body_model[0].infer_to_code(
                                               aud=aud[..., slice_start:slice_end],
                                               text=text[..., slice_start:slice_end],
                                               gt_poses=input_gt[..., slice_start:slice_end],
                                               mask=input_mask[..., slice_start:slice_end],
                                               id=id, B=num_sample)
                    code_list.append(codes)
                    if pred_all is None:
                        pred_all = pred_0
                    else:
                        pred_all = torch.cat([pred_all, pred_0[..., cover_length:]], -1)
                    input_gt[..., slice_start:slice_end] = pred_0
                    input_mask[..., slice_start:slice_end] = 1
                    cost_time = cost_time + time_x

                codes = torch.cat([code_list[0], code_list[1][:,3:]], dim=1)
                latents = generator.body_model[0].VQ.vq_layer.quantize_all(codes).transpose(1, 2)
                preliminary_motion = generator.body_model[0].VQ.decode(latents, None)

                preliminary_motion = F.interpolate(preliminary_motion, size=input_gt.shape[-1], align_corners=False, mode='linear')

                input_mask = mask.clone().repeat(num_sample, 1, 1)
                pred_all = preliminary_motion
                pred_all = None
                l_pslice = 176
                num_slices = math.ceil((t - 30) / (l_pslice - cover_length))
                for i in range(num_slices):
                    slice_start = (l_pslice - cover_length) * i
                    slice_end = slice_start + l_pslice

                    pred_0, time_x = generator.body_model[1].infer_on_batch(
                                               aud=aud[..., slice_start:slice_end],
                                               text=text[..., slice_start:slice_end],
                                               pred_poses=preliminary_motion[..., slice_start:slice_end],
                                               gt_poses=input_gt[..., slice_start:slice_end],
                                               mask=input_mask[..., slice_start:slice_end],
                                               id=id, B=num_sample)
                    if pred_all is None:
                        pred_all = pred_0
                    else:
                        pred_all = torch.cat([pred_all, pred_0[..., cover_length:]], -1)
                    input_gt[..., slice_start:slice_end] = pred_0
                    input_mask[..., slice_start:slice_end] = 1
                    cost_time = cost_time + time_x
                torch.cuda.synchronize()
                end_time = time.time()
                print('total_time is ', end_time - start_time)

                print(cost_time)

                for i in range(num_sample):
                    pred = pred_all[i]
                    if config.Data.pose.normalization:
                        pred = denormalize(pred, norm_stats[0], norm_stats[1], shape, c_index).squeeze(dim=0)
                    pred = pred.squeeze().to('cpu')

                    pred = pred.transpose(0, 1)
                    if pred.shape[1] == 376:
                        pred_face = pred[:, -exp_dim:]
                        pred = pred[:, :-exp_dim]
                    if config.Data.pose.convert_to_6d:
                        pred = pred.reshape(pred.shape[0], -1, 6)
                        pred = matrix_to_axis_angle(rotation_6d_to_matrix(pred))
                        pred = pred.reshape(pred.shape[0], -1)
                    if pred.shape[1] == 138:
                        pred = torch.cat([pred, pred_face[:pred.shape[0]]], dim=-1)
                    # else:
                    #     pad = torch.zeros([pred.shape[0], 1], device=pred.device)
                    #     pred = torch.cat([pad.repeat(1, 3), pred, pad.repeat(1, exp_dim)], dim=1)

                    # pred[:, 9:12] = global_orient
                    pred = part2full(pred, stand)
                    result_list[0] = poses2poses(result_list[0], pred.to('cuda'))

                    if face:
                        pad = torch.zeros([pred.shape[0], 162], device=pred.device)
                        pred = torch.cat([pred[:, :3], pad, pred[:, -exp_dim:]], dim=1)
                        pred = poses2poses(pred, result_list[0].cpu())

                    result_list.append(pred.to('cuda'))


                vertices_list, joints_list = get_vertices(smplx_model, betas, result_list, config.Data.pose.expression, require_pose=True)

                results_tensor = []
                for i in range(num_sample + 1):
                    results_tensor.append(joints_list[i].unsqueeze(0))
                results_tensor = torch.cat(results_tensor, dim=0)

                err = (results_tensor[0:1,:,:22] - results_tensor[1:,:,:22]).norm(p=2, dim=-1).sum(-1).sum(-1)
                min_index = err.argmin(0)

                rendertool._render_sequences(cur_wav_file, [vertices_list[0], vertices_list[min_index.item()+1]], mask=mask, stand=stand, face=face, multi_view=False)

                # result_list = [res.to('cpu') for res in result_list]
                result_list = [res.to('cpu') for res in [result_list[0], result_list[min_index.item()+1]]]
                dict = np.concatenate(result_list[:], axis=0)
                file_name = 'visualise/video/' + config.Log.name + '/' + \
                            cur_wav_file.split('\\')[-1].split('.')[-2].split('/')[-1]
                np.save(file_name, dict)

    # np.save('visualise/video/' + config.Log.name, all_list)


def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    config = load_JsonConfig(args.config_file)

    face_model_name = args.face_model_name
    face_model_path = args.face_model_path
    smplx_path = './visualise/'

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    print('init model...')
    generator = All_In_One_Model(args.face_model_name, args.face_model_path,
                                 args.body_model_name, args.body_model_path,
                                 device, args, config)
    print('init dataloader...')
    infer_set, infer_loader, norm_stats = init_dataloader(config.Data.data_root, args.speakers, args, config)

    print('init smlpx model...')
    dtype = torch.float32
    model_params = dict(model_path=smplx_path,
                        model_type='smplx',
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        num_betas=smplx_hyperparams['betas_dim'],
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        use_pca=False,
                        flat_hand_mean=False,
                        create_expression=True,
                        num_expression_coeffs=smplx_hyperparams['expression_dim'],
                        num_pca_comps=12,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        # gender='ne',
                        dtype=dtype, )
    smplx_model = smpl.create(**model_params).to('cuda')
    print('init rendertool...')
    rendertool = RenderTool('visualise/video/' + config.Log.name)
    # rendertool = None

    infer(config.Data.data_root, generator, infer_loader, device, norm_stats, smplx_model, rendertool, args, config)


if __name__ == '__main__':
    main()
