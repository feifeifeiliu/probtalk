# ProbTalk: Towards Variable and Coordinated Holistic Co-Speech Motion Generation [CVPR2024]

The official PyTorch implementation of the **CVPR2024** paper [**"Towards Variable and Coordinated Holistic Co-Speech Motion Generation"**](https://arxiv.org/abs/2404.00368).

Please visit our [**webpage**](https://feifeifeiliu.github.io/probtalk/) for more details.

[//]: # (![teaser]&#40;visualise/teaser_01.png&#41;)


## TODO

- [x] Update training code.
- [x] Update testing code.
- [ ] Update baseline methods.
- [ ] Update visualization code for linux.


## Getting started

The training code was tested on `Ubuntu 18.04.5 LTS` and the visualization code was test on `Windows 11`, and it requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (12GB+ GPU memory)


### 1. Setup environment

Clone the repo:
  ```bash
  git clone https://github.com/feifeifeiliu/probtalk.git
  cd probtalk
  ```  
Create conda environment:
```bash
conda create --name probtalk python=3.8
conda activate probtalk
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
pip install -r requirements.txt
```
    
Please install [**MPI-Mesh**](https://github.com/MPI-IS/mesh).

### 2. Get data

Download [**SHOW dataset**](https://talkshow.is.tue.mpg.de/download.php) and unzip it.
The [split files](data_utils/split) provide split information.

### 3. Download the pretrained models

Download [**pretrained models**](https://www.dropbox.com/scl/fo/4mdq1em6arysz1cxmkhtf/ACFenfjSPzFcswh_PIvtDz4?rlkey=p2wnbtcd81ko4y3tw5hdxhez1&e=1&st=swh97z2a&dl=0),
unzip and place it in the ProbTalk folder, i.e. ``path-to-ProbTalk/experiments``.

### 4. Training

The training procedure of ProbTalk includes 3 steps.

**(a) Train a PQ-VAE**

Modify the value of 'Data.data_root' in [vq.json](config/vq.json). 
```bash
bash train_vq.sh
```

**(b) Train a Predictor**

Modify the value of 'Model.model_name' in [transformer.json](config/transformer.json) to 's2g_body_predictor'. 

Modify the value of 'Model.vq_path' in [transformer.json](config/transformer.json).
```bash
bash train_transformer.sh
```

**(c) Train a Refiner**

Modify the value of 'Model.model_name' in [transformer.json](config/transformer.json) to 's2g_body_refiner'.
```bash
bash train_transformer.sh
```

### 5. Testing

We have identified a difference between the data we used and the SHOW dataset V1.0. We are working on correcting this issue and will update the model, code, and evaluation results soon.
In the meantime, to reproduce the evaluation results presented in our paper, please download the test data from  [**this page**](https://www.dropbox.com/scl/fo/4mdq1em6arysz1cxmkhtf/ACFenfjSPzFcswh_PIvtDz4?rlkey=p2wnbtcd81ko4y3tw5hdxhez1&e=1&st=swh97z2a&dl=0).
Then, set 'dataset_load_mode' to 'pickle' in the ['transformer.json'](config/transformer.json) configuration file, and run the following command:
```bash
bash test_holistic.sh
```

### 5. Visualization

If you ssh into the linux machine, NotImplementedError might occur. In this case, please refer to [**issue**](https://github.com/MPI-IS/mesh/issues/66) for solving the error.

Download [**smplx model**](https://drive.google.com/file/d/1Ly_hQNLQcZ89KG0Nj4jYZwccQiimSUVn/view?usp=share_link) (Please register in the official [**SMPLX webpage**](https://smpl-x.is.tue.mpg.de) before you use it.) and place it in ``path-to-ProbTalk/visualise/smplx_model``.
To visualise the demo videos, run:
    
    bash demo.sh

The videos and generated motion data are saved in ``./visualise/video/demo``.

If you ssh into the linux machine, there might be an error about OffscreenRenderer. In this case, please refer to [**issue**](https://github.com/MPI-IS/mesh/issues/66) for solving the error.

## Citation
If you find our work useful to your research, please consider citing:
```
@article{liu2024towards,
    title={Towards Variable and Coordinated Holistic Co-Speech Motion Generation},
    author={Liu, Yifei and Cao, Qiong and Wen, Yandong and Jiang, Huaiguang and Ding, Changxing},
    journal={arXiv preprint arXiv:2404.00368},
    year={2024}
}

@inproceedings{yi2023generating,
    title={Generating Holistic 3D Human Motion from Speech},
    author={Yi, Hongwei and Liang, Hualin and Liu, Yifei and Cao, Qiong and Wen, Yandong and Bolkart, Timo and Tao, Dacheng and Black, Michael J},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    pages={469-480},
    month={June}, 
    year={2023} 
}
```

## Acknowledgements
We thank Hongwei Yi for the insightful discussions, Hualin Liang for helping us conduct user study.

For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.  
Here are some great resources we benefit:  
- [Freeform](https://github.com/TheTempAccount/Co-Speech-Motion-Generation), [TalkShow](https://github.com/yhw-yhw/TalkSHOW) for training pipeline
- [MPI-Mesh](https://github.com/MPI-IS/mesh), [Pyrender](https://github.com/mmatl/pyrender), [Smplx](https://github.com/vchoutas/smplx), [VOCA](https://github.com/TimoBolkart/voca) for rendering  
- [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) and [Faceformer](https://github.com/EvelynFan/FaceFormer) for audio encoder

## License
This code and model are available for non-commercial and commercial purposes as defined in the LICENSE (i.e., MIT LICENSE). Note that, using ProbTalk, you have to register SMPL-X and agree with the LICENSE of it, and it's not MIT LICENSE, you can check the LICENSE of SMPL-X from https://github.com/vchoutas/smplx/blob/main/LICENSE; Enjoy your journey of exploring more beautiful avatars in your own application.
