import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pandas_path  # Path style access for pandas
from pandas_path import path
from tqdm import tqdm

import torch                    
import torchvision
import fasttext

from PIL import Image
from support.UniVisModel import UniVisionModel

import os


if __name__ == '__main__':

    #ARB
    torch.set_float32_matmul_precision('medium')
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print("You are using device: %s" % device)

    #data_dir = "./hateful_memes/"
    data_dir = "./HatefulMemes/"

    #img_tar_path = data_dir / "img.tar.gz"
    train_path = data_dir + "train.jsonl"
    dev_path = data_dir + "dev.jsonl"
    test_path = data_dir + "test.jsonl"

    
    hparams = {

        # Required hparams
        "train_path": train_path,
        "dev_path": dev_path,
        "img_dir": data_dir,

        # Optional hparams
        "embedding_dim": 150,
        "language_feature_dim": 300,
        "vision_feature_dim": 300,
        "fusion_output_size": 256,
        "output_path": "model-outputs",
        "dev_limit": None,
        "lr": 0.00005,
        #"max_epochs": 10,
        "max_epochs": 10,
        "n_gpu": 1,
        #"batch_size": 4,
        "batch_size": 16,
        # allows us to "simulate" having larger batches 
        #"accumulate_grad_batches": 16,
        "accumulate_grad_batches": 4,
        "early_stop_patience": 3,
        #alex: fix windows fasttext bug:
        "num_workers": 0,
    }

    univision_model = UniVisionModel(hparams=hparams)
    univision_model.fit()

    # ARB
    univision_model.get_visuals()


