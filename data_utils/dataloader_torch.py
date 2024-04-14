import sys
import os

from data_utils.consts import get_speaker_id, speaker_id
from data_utils.foundation_models import getFM

sys.path.append(os.getcwd())
import os
from tqdm import tqdm
from data_utils.utils import *
import torch.utils.data as data
from data_utils.mesh_dataset import SmplxDataset
from transformers import Wav2Vec2Processor
from data_utils.consts import smplx_hyperparams


exp_dim = smplx_hyperparams['expression_dim']


class MultiVidData():
    def __init__(self, 
                data_root, 
                speakers, 
                split='train', 
                limbscaling=False, 
                normalization=False,
                norm_method='new',
                split_trans_zero=False,
                num_frames=25,
                num_pre_frames=25,
                num_generate_length=None,
                aud_feat_win_size=None,
                aud_feat_dim=64,
                feat_method='mel_spec',
                context_info=False,
                smplx=False,
                audio_sr=16000,
                convert_to_6d=False,
                expression=False,
                config=None
                ):
        self.data_root = data_root
        self.speakers = speakers
        self.split = split
        if split == 'pre':
            self.split = 'train'
        self.norm_method=norm_method
        self.normalization = normalization
        self.limbscaling = limbscaling
        self.convert_to_6d = convert_to_6d
        self.num_frames=num_frames
        self.num_pre_frames=num_pre_frames
        if num_generate_length is None:
            self.num_generate_length = num_frames
        else:
            self.num_generate_length = num_generate_length
        self.split_trans_zero=split_trans_zero

        dataset = SmplxDataset
        
        if self.split_trans_zero:
            self.trans_dataset_list = []
            self.zero_dataset_list = []
        else:
            self.all_dataset_list = []
        self.dataset={}
        self.complete_data=[]
        self.config=config
        load_mode=self.config.dataset_load_mode

        self.fm_dict = getFM(config.Data.audio, config.Data.text)
        self.fm_dict['sr'] = 16000
        
        ######################load with pickle file
        if load_mode=='pickle':
            import pickle
            import subprocess
            
            # store_file_path='/tmp/store.pkl'
            # cp /is/cluster/scratch/hyi/ExpressiveBody/SMPLifyX4/scripts/store.pkl /tmp/store.pkl
            # subprocess.run(f'cp /is/cluster/scratch/hyi/ExpressiveBody/SMPLifyX4/scripts/store.pkl {store_file_path}',shell=True)
            
            # f = open(self.config.store_file_path, 'rb+')
            f = open(self.split+config.Data.pklname, 'rb+')
            self.dataset=pickle.load(f)
            f.close()
            for key in self.dataset:
                self.complete_data.append(self.dataset[key].complete_data)
        ######################load with pickle file
                
        ######################load with a csv file
        elif load_mode=='csv':

            # 这里从我的一个code文件夹导入的，后续再完善进来
            try:
                sys.path.append(self.config.config_root_path)
                from config import config_path
                from csv_parser import csv_parse
                
            except ImportError as e:
                print(f'err: {e}')
                raise ImportError('config root path error...')


            for speaker_name in self.speakers:
                # df_intervals=pd.read_csv(self.config.voca_csv_file_path)
                df_intervals=None
                df_intervals=df_intervals[df_intervals['speaker']==speaker_name]
                df_intervals = df_intervals[df_intervals['dataset'] == self.split]

                print(f'speaker {speaker_name} train interval length: {len(df_intervals)}')
                for iter_index, (_, interval) in tqdm(
                        (enumerate(df_intervals.iterrows())),desc=f'load {speaker_name}'
                ):
                    
                    (
                        interval_index,
                        interval_speaker,
                        interval_video_fn,
                        interval_id,
                        
                        start_time,
                        end_time,
                        duration_time,
                        start_time_10,
                        over_flow_flag,
                        short_dur_flag,
                        
                        big_video_dir,
                        small_video_dir_name,
                        speaker_video_path,
                        
                        voca_basename,
                        json_basename,
                        wav_basename,
                        voca_top_clip_path,
                        voca_json_clip_path,
                        voca_wav_clip_path,
                        
                        audio_output_fn,
                        image_output_path,
                        pifpaf_output_path,
                        mp_output_path,
                        op_output_path,
                        deca_output_path,
                        pixie_output_path,
                        cam_output_path, 
                        ours_output_path,
                        merge_output_path,
                        multi_output_path,
                        gt_output_path,
                        ours_images_path,
                        pkl_fil_path,
                    )=csv_parse(interval)
                    
                    if not os.path.exists(pkl_fil_path) or not os.path.exists(audio_output_fn):
                        continue

                    key=f'{interval_video_fn}/{small_video_dir_name}'
                    self.dataset[key] = dataset(
                        data_root=pkl_fil_path,
                        speaker=speaker_name,
                        audio_fn=audio_output_fn,
                        audio_sr=audio_sr,
                        fps=num_frames,
                        feat_method=feat_method,
                        audio_feat_dim=aud_feat_dim,
                        train=(self.split == 'train'),
                        load_all=True,
                        split_trans_zero=self.split_trans_zero,
                        limbscaling=self.limbscaling,
                        num_frames=self.num_frames,
                        num_pre_frames=self.num_pre_frames,
                        num_generate_length=self.num_generate_length,
                        audio_feat_win_size=aud_feat_win_size,
                        context_info=context_info,
                        convert_to_6d=convert_to_6d,
                        expression=expression,
                        config=self.config
                    )
                    self.complete_data.append(self.dataset[key].complete_data)
        ######################load with a csv file
                
        ######################origin load method
        elif load_mode=='json':



            # if wav2:
            #     am_sr = 16000
            #     from transformers import AutoProcessor
            #     from nets.spg.wav2vec import Wav2Vec2Model
            #     am = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            #     audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to('cuda')
            # else:
            #     am = None
            #     am_sr = None
            #     audio_model = None

            if 'expressive_body-V2.1' in data_root.split('/')[-1]:
                id = get_speaker_id(data_root)
                self.speakers = os.listdir(data_root)
            else:
                id = speaker_id

            for speaker_name in self.speakers:
                speaker_root = os.path.join(self.data_root, speaker_name)

                videos=[v for v in os.listdir(speaker_root) ]
                print(videos)

                haode = huaide = 0

                for vid in tqdm(videos, desc="Processing training data of {}......".format(speaker_name)):
                    source_vid=vid
                    # vid_pth=os.path.join(speaker_root, source_vid, 'images/half', self.split)
                    vid_pth = os.path.join(speaker_root, source_vid, self.split)
                    if smplx == 'pose':
                        seqs = [s for s in os.listdir(vid_pth) if (s.startswith('clip'))]
                    else:
                        try:
                            seqs = [s for s in os.listdir(vid_pth)]
                        except:
                            continue

                    for s in seqs:
                        seq_root=os.path.join(vid_pth, s)
                        key = seq_root # correspond to clip******
                        audio_fname = os.path.join(speaker_root, source_vid, self.split, s, '%s.wav' % (s))
                        motion_fname = os.path.join(speaker_root, source_vid, self.split, s, '%s.pkl' % (s))
                        if not os.path.isfile(audio_fname) or not os.path.isfile(motion_fname):
                            huaide = huaide + 1
                            continue

                        # if haode >= 200:
                        #     break

                        self.dataset[key]=dataset(
                            data_root=seq_root,
                            speaker=id[speaker_name],
                            motion_fn=motion_fname,
                            audio_fn=audio_fname,
                            audio_sr=audio_sr,
                            fps=num_frames,
                            feat_method=feat_method,
                            audio_feat_dim=aud_feat_dim,
                            train=(self.split=='train'),
                            load_all=True,
                            split_trans_zero=self.split_trans_zero,
                            limbscaling=self.limbscaling,
                            num_frames=self.num_frames,
                            num_pre_frames=self.num_pre_frames,
                            num_generate_length=self.num_generate_length,
                            audio_feat_win_size=aud_feat_win_size,
                            context_info=context_info,
                            convert_to_6d=convert_to_6d,
                            expression=expression,
                            config=self.config,
                            fm_dict=self.fm_dict,
                            whole_video=config.Data.whole_video,
                        )
                        self.complete_data.append(self.dataset[key].complete_data)
                        haode = haode + 1
                print("huaide:{}, haode:{}".format(huaide, haode))
            import pickle

            f = open(self.split+config.Data.pklname, 'wb')
            pickle.dump(self.dataset, f)
            f.close()

        ######################origin load method

        self.complete_data=np.concatenate(self.complete_data, axis=0, dtype='f')
        self.normalize_stats = {}

        # assert self.complete_data.shape[-1] == (12+21+21)*2
        if self.normalization:
            data_mean, data_std = self._normalization_stats(self.complete_data)
            self.data_mean = data_mean.reshape(1, 1, -1)
            self.data_std = data_std.reshape(1, 1, -1)
        else:
            self.data_mean = None
            self.data_std = None

        self.complete_data = []

        # if fm_dict['aud_m'] is not None:
        #     fm_dict['aud_m'].to('cpu')
        # if fm_dict['text_m'] is not None:
        #     fm_dict['text_m'].to('cpu')
        # del fm_dict
        torch.cuda.empty_cache()
    
    def get_dataset(self, norm_stats):
        self.normalize_stats['mean'] = norm_stats[0]
        self.normalize_stats['std'] = norm_stats[1]

        for key in list(self.dataset.keys()):
            if self.dataset[key].complete_data.shape[0] < self.num_generate_length:
                continue
            self.dataset[key].num_generate_length = self.num_generate_length
            self.dataset[key].get_dataset(self.normalization, self.normalize_stats, self.split)
            self.all_dataset_list.append(self.dataset[key].all_dataset)
        
        if self.split_trans_zero:
            self.trans_dataset = data.ConcatDataset(self.trans_dataset_list)
            self.zero_dataset = data.ConcatDataset(self.zero_dataset_list)
        else:
            self.all_dataset = data.ConcatDataset(self.all_dataset_list)

    def _normalization_stats(self, complete_data):
        face_dim = exp_dim
        face_data = complete_data[:, -face_dim:] # [n, face_dim]
        face_mean = np.mean(face_data, axis=0)
        face_std = np.std(face_data, axis=0)
        face_std[np.where(face_std == 0)] = 1e-9

        if self.convert_to_6d:
            print('warning: using new data normalization')
            complete_data = complete_data[:, :330].reshape(complete_data.shape[0], -1, 6).reshape(-1, 6)
            data_mean = np.mean(complete_data, axis=0)
            data_std = np.std(complete_data, axis=0)
            data_mean = data_mean.squeeze().reshape(1, 6)
            data_mean = np.repeat(data_mean, 55, 0)
            assert data_mean.shape[0] == 55 and data_mean.shape[1] == 6
            data_mean = data_mean.reshape(-1)

            data_std = data_std.squeeze().reshape(1, 6)
            data_std = np.repeat(data_std, 55, 0)
            assert data_std.shape[0] == 55 and data_std.shape[1] == 6
            data_std = data_std.reshape(-1)
        else:
            data_mean = np.mean(complete_data[:, :165], axis=0)
            data_std = np.std(complete_data[:, :165], axis=0)
            data_std[np.where(data_std == 0)] = 1e-9

        data_mean = np.concatenate([data_mean, face_mean])
        data_std = np.concatenate([data_std, face_std])

        return data_mean, data_std



