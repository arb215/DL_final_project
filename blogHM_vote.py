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
from support.ModelHM_vote import HatefulMemesModel

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

    #train_samples_frame = pd.read_json(train_path, lines=True)
    ##print(train_samples_frame.head())
#
    #print(train_samples_frame.label.value_counts())
#
    #desc = train_samples_frame.text.map(
    #    lambda text: len(text.split(" "))
    #).describe()
#
    ##print(desc)
#
    ## define a callable image_transform with Compose
    #image_transform = torchvision.transforms.Compose(
    #    [
    #        torchvision.transforms.Resize(size=(224, 224)),
    #        torchvision.transforms.ToTensor()
    #    ]
    #)
#
    #images = [
    #    Image.open(
    #        data_dir + train_samples_frame.loc[i, "img"]
    #    ).convert("RGB")
    #    for i in range(5)
    #]
#
    #for image in images:
    #    print(image.size)
#
    ## convert the images and prepare for visualization.
    #tensor_img = torch.stack(
    #    [image_transform(image) for image in images]
    #)
    #grid = torchvision.utils.make_grid(tensor_img)
#
    ## plot
    #plt.rcParams["figure.figsize"] = (20, 5)
    #plt.axis('off')
    #_ = plt.imshow(grid.permute(1, 2, 0))
#
    #print(tensor_img.shape)

    hparams = {

        # Required hparams
        "train_path": train_path,
        "dev_path": dev_path,
        "img_dir": data_dir,

        # Optional hparams
        "embedding_dim": 150,
        "language_feature_dim": 768,
        "vision_feature_dim": 1000,
        "fusion_output_size": 256,
        "output_path": "model-outputs",
        "dev_limit": None,
        "lr": 0.00005,
        #"max_epochs": 10,
        "max_epochs": 10,
        "n_gpu": 1,
        #"batch_size": 2,
        #"batch_size": 4,
        "batch_size": 8,
        #"batch_size": 16,
        # allows us to "simulate" having larger batches 
        #"accumulate_grad_batches": 32,
        #"accumulate_grad_batches": 16,
        "accumulate_grad_batches": 8,
        #"accumulate_grad_batches": 4,
        "early_stop_patience": 3,
        #alex: fix windows fasttext bug:
        "num_workers": 0,
    }

    hateful_memes_model = HatefulMemesModel(hparams=hparams)
    hateful_memes_model.fit()

    # ARB
    hateful_memes_model.get_visuals()

    #hateful_memes_model = HatefulMemesModel.load_from_checkpoint(hparams=hparams, checkpoint_path='./model-outputs/myCheckpoint.ckpt')
    #
    #submission = hateful_memes_model.make_submission_frame(
    #    test_path
    #)
    #submission.head()
    #submission.groupby("label").proba.mean()
    #submission.label.value_counts()
    #submission.to_csv(("model-outputs/submission.csv"), index=True)
    #
