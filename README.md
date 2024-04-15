# ProbTalk: Towards Variable and Coordinated Holistic Co-Speech Motion Generation [CVPR2024]

The official PyTorch implementation of the **CVPR2024** paper [**"Towards Variable and Coordinated Holistic Co-Speech Motion Generation"**](https://arxiv.org/abs/2404.00368).

Please visit our [**webpage**](https://feifeifeiliu.github.io/ProbTalk/) for more details.

[//]: # (![teaser]&#40;visualise/teaser_01.png&#41;)


## TODO

- [ ] Training code.
- [ ] Testing code.


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

### 2. Get data (to do)

### 3. Download the pretrained models

Download [**pretrained models**](https://drive.google.com/drive/folders/1hm_6s8HuToQz4Fa8PO3WrcJ7DZV7hWwq?usp=sharing),
unzip and place it in the ProbTalk folder, i.e. ``path-to-ProbTalk/experiments``.

### 4. Training (to do)

### 5. Testing (to do)

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
For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.  
Here are some great resources we benefit:  
- [Freeform](https://github.com/TheTempAccount/Co-Speech-Motion-Generation), [TalkShow](https://github.com/yhw-yhw/TalkSHOW) for training pipeline
- [MPI-Mesh](https://github.com/MPI-IS/mesh), [Pyrender](https://github.com/mmatl/pyrender), [Smplx](https://github.com/vchoutas/smplx), [VOCA](https://github.com/TimoBolkart/voca) for rendering  
- [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) and [Faceformer](https://github.com/EvelynFan/FaceFormer) for audio encoder

## License
This code and model are available for non-commercial and commercial purposes as defined in the LICENSE (i.e., MIT LICENSE). Note that, using ProbTalk, you have to register SMPL-X and agree with the LICENSE of it, and it's not MIT LICENSE, you can check the LICENSE of SMPL-X from https://github.com/vchoutas/smplx/blob/main/LICENSE; Enjoy your journey of exploring more beautiful avatars in your own application.
