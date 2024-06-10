export sine=1
CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/train.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth chemistry conan \
--config_file ./config/transformer.json \
#--resume \
#--pretrained_pth ./experiments/2023-09-20-smplx_S2G-hf-single-all/ckpt-59.pth

