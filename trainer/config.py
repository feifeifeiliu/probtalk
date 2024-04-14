'''
load config from json file
'''
import json
import os

import configparser

import yaml


class Object():
    def __init__(self, config:dict) -> None:
        for key in list(config.keys()):
            if isinstance(config[key], dict):
                setattr(self, key, Object(config[key]))
            else:
                setattr(self, key, config[key])


def dict_merge(old, new):
    for key in list(new.keys()):
        if isinstance(new[key], dict):
            old[key] = dict_merge(old[key], new[key])
        else:
            old[key] = new[key]
    return old

def get_full_ymlconfig(config):
    empty_dict = {}
    for key in list(config.keys()):
        if key == '_BASE_':
            with open(config[key], 'r') as f:
                base = yaml.load(f, Loader=yaml.FullLoader)
            empty_dict = get_full_ymlconfig(base)
        elif isinstance(config[key], dict) and key in list(empty_dict.keys()):
            empty_dict[key] = dict_merge(empty_dict[key], config[key])
        else:
            empty_dict[key] = config[key]

    return empty_dict

def load_JsonConfig(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    
    return Object(config)


def load_YmlConfig(yml_file):
    with open(yml_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = get_full_ymlconfig(config)

    return Object(config)



if __name__ == '__main__':
    config = load_JsonConfig('config/style_gestures.json')
    print(dir(config))