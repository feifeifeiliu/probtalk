CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/train.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/vq.json