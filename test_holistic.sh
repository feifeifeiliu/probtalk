CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/test_holisticbody.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file \
./config/transformer.json \
--body_model_name \
s2g_body_predictor \
s2g_body_refiner \
--body_model_path \
experiments/2023-11-06-smplx_S2G-predictor-audio-mocon_v2/ckpt-99.pth \
experiments/2024-03-31-smplx_S2G-final-hf/ckpt-99.pth \
--infer

#--config_file ./config/LS3DCG.json \
#--body_model_name s2g_LS3DCG \
#--body_model_path experiments/2023-11-10-smplx_S2G-LS3DCG-6d-0.1g/ckpt-99.pth \

#--config_file ./config_yml/248_server/body_vqt.yml \
#--body_model_name s2g_body_vqt \
#--body_model_path experiments/2024-01-26-smplx_S2G-rq4096_d128_group4/ckpt-99.pth \

#--body_model_name s2g_body_predictor \
#--body_model_path experiments/2023-08-28-smplx_S2G-predictor_linear-vq_res_fbhe/ckpt-99.pth \
#--model_path experiments/2023-08-26-smplx_S2G-new_vqt_fbhe1024_size8_newourq4096_d128_group4/ckpt-99.pth \


#--config_file ./config/body_pixel.json \
#--face_model_name s2g_face \
#--face_model_path experiments/2023-10-28-smplx_S2G-face-6d/ckpt-99.pth \
#--body_model_name s2g_body_pixel \
#--body_model_path experiments/2023-10-30-smplx_S2G-body-pixel-6d/ckpt-99.pth \


#--config_file \
#./config/body_inpaint.json \
#--body_model_name \
#s2g_body_predictor \
#s2g_body_hf \
#--body_model_path \
#experiments/2023-11-06-smplx_S2G-predictor-audio-mocon_v2/ckpt-99.pth \
#experiments/2023-11-25-smplx_S2G-hf_vqaug_win15_r0.5_smooth/ckpt-139.pth \