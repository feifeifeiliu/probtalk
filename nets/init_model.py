import torch

import nets


def init_model(model_name, args, config, pretrained=False, model_path=None):

    generator = getattr(nets, model_name)(args, config)
    if pretrained:
        model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        generator.load_state_dict(model_ckpt['generator'])
    return generator


