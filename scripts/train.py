import os
import sys
# os.chdir('/home/jovyan/Co-Speech-Motion-Generation/src')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append(os.getcwd())

from trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()