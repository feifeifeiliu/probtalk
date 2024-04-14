import os
import sys

import librosa

from data_utils.consts import smplx_hyperparams
from data_utils.utils import linear_interpolation
from nets.ai1 import All_In_One_Model
from scripts.visualise_inpaint import get_vertices

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())

from transformers import Wav2Vec2Processor, AutoProcessor, Wav2Vec2Model
from glob import glob

import numpy as np
import json
import smplx as smpl
import time
import math
import pickle

from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
from data_utils.lower_body import part2full, pred2poses, poses2pred, poses2poses
from visualise.rendering import RenderTool
from data_utils.lower_body import c_index_3d, c_index_6d
from data_utils.consts import speaker_id

betas_dim = smplx_hyperparams['betas_dim']
exp_dim = smplx_hyperparams['expression_dim']


global_orient = torch.tensor([3.0747, -0.0158, -0.0152])


def get_betas(speaker):
    if speaker == 'oliver':
        motion_fn = "./demo_audio/oliver/214542-00_01_17-00_01_27/214542-00_01_17-00_01_27.pkl"
    elif speaker == 'seth':
        motion_fn = "./demo_audio/seth/98201-00_04_32-00_04_42/98201-00_04_32-00_04_42.pkl"
    elif speaker == 'chemistry':
        motion_fn = "./demo_audio/chemistry/68991-00_00_23-00_00_33/68991-00_00_23-00_00_33.pkl"
    elif speaker == 'conan':
        motion_fn = "./demo_audio/conan/115563-00_01_35-00_01_45/115563-00_01_35-00_01_45.pkl"

    f = open(motion_fn, 'rb+')
    data = pickle.load(f)

    try:
        jaw_pose = np.array(data['jaw_pose'])
    except:
        data = data[0]

    betas = np.array(data['betas'])
    return torch.from_numpy(betas).to('cuda')



def infer(generator, smplx_model, rendertool, config, args):
    # betas = torch.zeros([1, 300], dtype=torch.float64).to('cuda')
    proxies = {
        # 'http': '127.0.0.1:7890',
        # 'https': '127.0.0.1:7890',
    }
    aud_p = AutoProcessor.from_pretrained("D:\Downloads\wav2vec", proxies=proxies)
    aud_m = Wav2Vec2Model.from_pretrained("D:\Downloads\wav2vec", proxies=proxies)
    aud_m = aud_m.to('cuda')
    num_sample = args.num_sample
    cur_wav_file = args.audio_file
    speech_array, sampling_rate = librosa.load(cur_wav_file, sr=16000)
    audio = torch.from_numpy(speech_array)
    speaker = args.speaker
    betas = get_betas(speaker)
    face = args.only_face
    stand = args.stand
    c_index = c_index_6d if config.Data.pose.convert_to_6d else c_index_3d
    norm_stats = np.load("./data_utils/norm_stats.npy", allow_pickle=True)
    norm_stats_torch = []
    norm_stats_torch.append(torch.from_numpy(norm_stats[0]).to('cuda'))
    norm_stats_torch.append(torch.from_numpy(norm_stats[1]).to('cuda'))
    norm_stats = norm_stats_torch
    shape = 'fbhe'

    if face:
        body_static = torch.zeros([1, 162], device='cuda')
        body_static[:, 6:9] = torch.tensor([3.0747, -0.0158, -0.0152]).reshape(1, 3).repeat(body_static.shape[0], 1)

    result_list = []

    id = speaker_id[args.speaker]
    id = torch.tensor([id], device='cuda')

    inputs = aud_p(audio, sampling_rate=sampling_rate, return_tensors="pt")
    for key in inputs.data:
        inputs.data[key] = inputs.data[key].to('cuda')
    with torch.no_grad():
        outputs = aud_m(**inputs)
    audio_ft = outputs.last_hidden_state
    aud = linear_interpolation(audio_ft.transpose(1, 2), int(speech_array.shape[0] / sampling_rate * 30))

    t = aud.shape[-1]

    pred_all = None
    l_pslice = 180
    cover_length = 30
    num_slices = math.ceil((t - 30) / (l_pslice - cover_length))
    cost_time = 0
    cost_time_all = 0
    input_gt = torch.zeros([num_sample, 376, t], device='cuda')
    input_mask = torch.zeros([num_sample, 1, t], device='cuda')
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
            text=aud[..., slice_start:slice_end],
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

    codes = torch.cat([code_list[0], code_list[1][:, 3:]], dim=1)
    latents = generator.body_model[0].VQ.vq_layer.quantize_all(codes).transpose(1, 2)
    preliminary_motion = generator.body_model[0].VQ.decode(latents, None)

    preliminary_motion = F.interpolate(preliminary_motion, size=input_gt.shape[-1], align_corners=False, mode='linear')

    input_mask = torch.zeros([num_sample, 1, t], device='cuda')
    pred_all = None
    l_pslice = 176
    num_slices = math.ceil((t - 30) / (l_pslice - cover_length))
    for i in range(num_slices):
        slice_start = (l_pslice - cover_length) * i
        slice_end = slice_start + l_pslice

        pred_0, time_x = generator.body_model[1].infer_on_batch(
            aud=aud[..., slice_start:slice_end],
            text=aud[..., slice_start:slice_end],
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

        if face:
            pad = torch.zeros([pred.shape[0], 162], device=pred.device)
            pred = torch.cat([pred[:, :3], pad, pred[:, -exp_dim:]], dim=1)
            pred = poses2poses(pred, result_list[0].cpu())

        result_list.append(pred.to('cuda'))

    vertices_list, _ = get_vertices(smplx_model, betas, result_list, config.Data.pose.expression)

    result_list = [res.to('cpu') for res in result_list]
    dict = np.concatenate(result_list[:], axis=0)
    file_name = 'visualise/video/' + config.Log.name + '/' + \
                cur_wav_file.split('\\')[-1].split('.')[-2].split('/')[-1]
    np.save(file_name, dict)

    rendertool._render_sequences(cur_wav_file, vertices_list, stand=stand, face=face, whole_body=args.whole_body)


def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    config = load_JsonConfig(args.config_file)

    smplx_path = './visualise/'

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    print('init model...')
    generator = All_In_One_Model(args.face_model_name, args.face_model_path,
                                 args.body_model_name, args.body_model_path,
                                 device, args, config)

    print('init smlpx model...')
    dtype = torch.float32
    model_params = dict(model_path=smplx_path,
                        model_type='smplx',
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        num_betas=300,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        use_pca=False,
                        flat_hand_mean=False,
                        create_expression=True,
                        num_expression_coeffs=100,
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

    infer(generator, smplx_model, rendertool, config, args)


if __name__ == '__main__':
    main()
