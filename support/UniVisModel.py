import json
import logging
from pathlib import Path
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch                    
import torchvision
import fasttext
from torchmetrics.classification import BinaryAUROC

import pytorch_lightning as pl

from support.DatasetHM import HatefulMemesDataset
from support.ConcatVisOnly import LanguageAndVisionConcat


# for the purposes of this post, we'll filter
# much of the lovely logging info from our LightningModule
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)


class UniVisionModel(pl.LightningModule):
    def __init__(self, hparams):
        for data_key in ["train_path", "dev_path", "img_dir",]:
            # ok, there's one for-loop but it doesn't count
            if data_key not in hparams.keys():
                raise KeyError(
                    f"{data_key} is a required hparam in this model"
                )
        
        super(UniVisionModel, self).__init__()

        #self.hparams = hparams
        for key in hparams.keys():
            self.hparams[key]=hparams[key]
        
        # assign some hparams that get used in multiple places
        self.embedding_dim = self.hparams.get("embedding_dim", 300)
        self.language_feature_dim = self.hparams.get(
            "language_feature_dim", 300
        )
        self.vision_feature_dim = self.hparams.get(
            # balance language and vision features by default
            "vision_feature_dim", self.language_feature_dim
        )
        self.output_path = Path(
            self.hparams.get("output_path", "model-outputs")
        )
        self.output_path.mkdir(exist_ok=True)
        
        # instantiate transforms, datasets
        self.text_transform = self._build_text_transform()
        self.image_transform = self._build_image_transform()
        self.train_dataset = self._build_dataset("train_path")
        self.dev_dataset = self._build_dataset("dev_path")
        
        # set up model and training
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

        # ARB
        self.val_loss = []
        self.train_loss = []
        self.val_acc = []
        self.train_acc = []
        self.val_auroc = []        
        self.train_auroc = []

    ## Required LightningModule Methods (when validating) ##
    
    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def training_step(self, batch, batch_nb):
        preds, loss = self.forward(
            text=batch["text"], 
            image=batch["image"], 
            label=batch["label"]
        )
        # ARB
        acc = preds[torch.arange(preds.shape[0]), batch["label"]]
        ave_acc = acc.mean()
        metric = BinaryAUROC(thresholds=5).to('cuda')
        acc_auroc = metric(acc, batch["label"])
        
        return {"loss": loss, "training_loss": loss, "ave_acc": ave_acc, "acc_auroc": acc_auroc}

    def validation_step(self, batch, batch_nb):
        preds, loss = self.eval().forward(
            text=batch["text"], 
            image=batch["image"], 
            label=batch["label"]
        )
        # ARB
        acc = preds[torch.arange(preds.shape[0]), batch["label"]]
        ave_acc = acc.mean()
        metric = BinaryAUROC(thresholds=5).to('cuda')
        acc_auroc = metric(acc, batch["label"])
        
        return {"batch_val_loss": loss, "ave_acc": ave_acc, "acc_auroc": acc_auroc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            tuple(
                output["batch_val_loss"] 
                for output in outputs
            )
        ).mean()
        avg_acc = torch.stack(
            tuple(
                output["ave_acc"] 
                for output in outputs
            )
        ).mean()
        avg_auroc = torch.stack(
            tuple(
                output["acc_auroc"] 
                for output in outputs
            )
        ).mean()

        self.val_loss.append(avg_loss.cpu())
        self.val_acc.append(avg_acc.cpu())
        self.val_auroc.append(avg_auroc.cpu())
        
        return {
            "val_loss": avg_loss,
            "progress_bar":{"val_loss": avg_loss, "val_acc": avg_acc, "val_auroc": avg_auroc}
        }
    
    def training_epoch_end(self, outputs):
        # ARB
        avg_loss = torch.stack(
            tuple(
                output["training_loss"] 
                for output in outputs
            )
        ).mean()
        avg_acc = torch.stack(
            tuple(
                output["ave_acc"] 
                for output in outputs
            )
        ).mean()
        avg_auroc = torch.stack(
            tuple(
                output["acc_auroc"] 
                for output in outputs
            )
        ).mean()

        self.train_loss.append(avg_loss.cpu())
        self.train_acc.append(avg_acc.cpu())
        self.train_auroc.append(avg_auroc.cpu())

        #torch.save({
        #    'model_state_dict': self.vision_module.state_dict()
        #}, "vision_model_" + str(self.current_epoch) + ".pt")
    
    @torch.no_grad()
    def get_visuals(self):
        # ARB

        plt.plot(self.train_loss, label='train')
        plt.plot(self.val_loss, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('myloss.png')
        plt.close()

        plt.plot(self.train_acc, label='train')
        plt.plot(self.val_acc, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        plt.savefig('myAcc.png')
        plt.close()

        plt.plot(self.train_auroc, label='train')
        plt.plot(self.val_auroc, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('auroc')
        plt.legend()
        plt.savefig('myAuroc.png')
        plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                        self.model.parameters(), 
                        lr=self.hparams.get("lr", 0.001)
                    )
        
        # ARB
        fcn = lambda epoch: 0.95 ** epoch

        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    #'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                    'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fcn),
                    #'monitor': 'batch_val_loss',          
                }
        }
    
    #def configure_optimizers(self):
    #    optimizers = [
    #        torch.optim.AdamW(
    #            self.model.parameters(), 
    #            lr=self.hparams.get("lr", 0.001)
    #        )
    #    ]
    #    #schedulers = [
    #    #    torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    #        optimizers[0]
    #    #    )
    #    #]
    #    return optimizers#, schedulers
    
    #@pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            shuffle=True, 
            batch_size=self.hparams.get("batch_size", 4), 
            num_workers=self.hparams.get("num_workers", 16)
        )

    #@pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset, 
            shuffle=False, 
            batch_size=self.hparams.get("batch_size", 4), 
            num_workers=self.hparams.get("num_workers", 16)
        )
    
    ## Convenience Methods ##
    
    def fit(self):
        self._set_seed(self.hparams.get("random_state", 42))
        #self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer = pl.Trainer(
            callbacks=[self.trainer_params['checkpoint_callback']],#, self.trainer_params['early_stop_callback']],
            default_root_dir=self.trainer_params['default_save_path'],
            accumulate_grad_batches=self.trainer_params['accumulate_grad_batches'],
            gpus=self.trainer_params['gpus'],
            max_epochs=self.trainer_params['max_epochs'],
            gradient_clip_val=self.trainer_params['gradient_clip_val']
        )

        self.trainer.fit(self)
        
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_text_transform(self):
        ##with tempfile.NamedTemporaryFile() as ft_training_data:
        #with tempfile.NamedTemporaryFile(mode='w', encoding='utf8') as ft:
        #    #ft_path = Path(ft_training_data.name)
        #    ft_path = Path(ft.name)
        #    #with ft_path.open("w") as ft:
        #    # had to tab the group below left one because of above commentout
        #    training_data = [
        #        json.loads(line)["text"] + "/n" 
        #        for line in open(
        #            self.hparams.get("train_path"), encoding='utf-8'
        #        ).read().splitlines()
        #    ]
        #    for line in training_data:
        #        ft.write(line + "\n")
        #    language_transform = fasttext.train_unsupervised(
        #        #str(ft_path),
        #        str(ft),
        #        model=self.hparams.get("fasttext_model", "cbow"),
        #        dim=self.embedding_dim
        #    )
        f = open('temp.txt', mode= 'w', encoding='utf-8')
        training_data = [
            json.loads(line)["text"] + "/n" 
            for line in open(
                self.hparams.get("train_path"), encoding='utf-8'
            ).read().splitlines()
        ]
        for line in training_data:
            f.write(line + "\n")
        language_transform = fasttext.train_unsupervised(
            'temp.txt',
            model=self.hparams.get("fasttext_model", "cbow"),
            dim=self.embedding_dim
        )
        f.close()

        return language_transform
    
    def _build_image_transform(self):
        image_dim = self.hparams.get("image_dim", 224)
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    size=(image_dim, image_dim)
                ),        
                torchvision.transforms.ToTensor(),
                # all torchvision models expect the same
                # normalization mean and std
                # https://pytorch.org/docs/stable/torchvision/models.html
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return image_transform

    def _build_dataset(self, dataset_key):
        return HatefulMemesDataset(
            data_path=self.hparams.get(dataset_key, dataset_key),
            img_dir=self.hparams.get("img_dir"),
            image_transform=self.image_transform,
            text_transform=self.text_transform,
            # limit training samples only
            dev_limit=(
                self.hparams.get("dev_limit", None) 
                if "train" in str(dataset_key) else None
            ),
            balance=True if "train" in str(dataset_key) else False,
        )
    
    def _build_model(self):
        
        # easiest way to get features rather than
        # classification is to overwrite last layer
        # with an identity transformation, we'll reduce
        # dimension using a Linear layer, resnet is 2048 out
        self.vision_module = torchvision.models.resnet152(
            pretrained=True
        )
        #vision_module.fc = torch.nn.Linear(
        #        in_features=2048,
        #        out_features=self.vision_feature_dim
        #)
        print("this is _build_model")

        return LanguageAndVisionConcat(
            num_classes=self.hparams.get("num_classes", 2),
            loss_fn=torch.nn.CrossEntropyLoss(),
            vision_module=self.vision_module,
            vision_feature_dim=self.vision_feature_dim,
            dropout_p=self.hparams.get("dropout_p", 0.1),
        )
    
    def _get_trainer_params(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            #filepath=self.output_path,
            dirpath=self.output_path,
            monitor=self.hparams.get(
                #"checkpoint_monitor", "avg_val_loss"
                "checkpoint_monitor", None
            ),
            mode=self.hparams.get(
                "checkpoint_monitor_mode", "min"
            ),
            verbose=self.hparams.get("verbose", True)
        )

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=self.hparams.get(
                "early_stop_monitor", "avg_val_loss"
            ),
            min_delta=self.hparams.get(
                "early_stop_min_delta", 0.001
            ),
            patience=self.hparams.get(
                "early_stop_patience", 3
            ),
            verbose=self.hparams.get("verbose", True),
        )

        trainer_params = {
            "checkpoint_callback": checkpoint_callback,
            "early_stop_callback": early_stop_callback,
            "default_save_path": self.output_path,
            "accumulate_grad_batches": self.hparams.get(
                "accumulate_grad_batches", 1
            ),
            "gpus": self.hparams.get("n_gpu", 1),
            "max_epochs": self.hparams.get("max_epochs", 100),
            "gradient_clip_val": self.hparams.get(
                "gradient_clip_value", 1
            ),
        }
        return trainer_params
            
    @torch.no_grad()
    def make_submission_frame(self, test_path):
        test_dataset = self._build_dataset(test_path)
        submission_frame = pd.DataFrame(
            index=test_dataset.samples_frame.id,
            columns=["proba", "label"]
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            shuffle=False, 
            batch_size=self.hparams.get("batch_size", 4), 
            num_workers=self.hparams.get("num_workers", 16))
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            preds, _ = self.model.eval().to("cpu")(
                batch["text"], batch["image"]
            )
            submission_frame.loc[batch["id"], "proba"] = preds[:, 1]
            submission_frame.loc[batch["id"], "label"] = preds.argmax(dim=1)
        submission_frame.proba = submission_frame.proba.astype(float)
        submission_frame.label = submission_frame.label.astype(int)
        return submission_frame