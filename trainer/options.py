from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='experiments', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--speakers', nargs='+')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--model_name', type=str)
    
    #for Tmpt and S2G
    parser.add_argument('--use_template', action='store_true')
    parser.add_argument('--template_length', default=0, type=int)

    #for training from a ckpt
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrained_pth', default=None, type=str)
    parser.add_argument('--style_layer_norm', action='store_true')
    
    #required
    parser.add_argument('--config_file', default='./config/style_gestures.json', type=str)

    # for visualization and test
    parser.add_argument('--audio_file', default=None, type=str)
    parser.add_argument('--speaker', default='oliver', type=str, help='oliver, chemistry, seth, conan')
    parser.add_argument('--only_face', action='store_true')
    parser.add_argument('--stand', action='store_true')
    parser.add_argument('--whole_body', action='store_true')
    parser.add_argument('--num_sample', default=1, type=int)
    parser.add_argument('--model_path', default='experiments/2023-05-09-smplx_S2G-hf-vqbh-nomrec_real/ckpt-99.pth', type=str)
    parser.add_argument('--face_model_name', default=None, type=str)
    parser.add_argument('--face_model_path', default=None, type=str)
    parser.add_argument('--body_model_name', nargs='+', default='s2g_body_pixel')
    parser.add_argument('--body_model_path', nargs='+', default='./experiments/2023-05-12-smplx_S2G-body-pixel/ckpt-99.pth')
    parser.add_argument('--infer', action='store_true')

    return parser