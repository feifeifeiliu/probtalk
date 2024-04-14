import os

expression_dim = os.environ.setdefault('expression_dim', '100')
betas_dim = os.environ.setdefault('betas_dim', '300')

speaker_id = {
    'oliver': 0,
    'chemistry': 1,
    'seth': 2,
    'conan': 3,
}

def get_speaker_id(data_root):
    keys = os.listdir(data_root)
    speakers = {}
    for id in range(keys.__len__()):
        speakers[keys[id]] = id
    return speakers


smplx_hyperparams = {'betas_dim': int(betas_dim), 'expression_dim': int(expression_dim)}