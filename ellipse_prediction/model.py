import os
import sys

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from ellcv.types import Ellipse
from ellcv.utils import compute_ellipses_iou
from ellcv.visu import draw_ellipse

sys.path.append("../config")
from config.config import _cfg
from ellipse_prediction.loss import SamplingBasedLoss



def build_ellipses_from_pred(pred):
    ellipses = []
    for i in range(pred.shape[0]):
        axes = pred[i, :2].numpy()
        center = pred[i, 2:4].numpy()
        angle = torch.asin(pred[i, 4]).numpy()
        ell = Ellipse.compose(axes, angle, center)
        ellipses.append(ell)
    return ellipses



class EllipsePredictor(pl.LightningModule):
    def __init__(self, lr=_cfg.LEARNING_RATE, crop_size=_cfg.CROP_SIZE, output_val_images=None):
        super(EllipsePredictor, self).__init__()

        self.lr = lr
        self.crop_size = crop_size
        self.output_val_images = output_val_images

        # Loss
        self.loss_fn = SamplingBasedLoss(_cfg.LOSS_SAMPLING_X_INTERVAL,
                                         _cfg.LOSS_SAMPLING_Y_INTERVAL,
                                         _cfg.LOSS_SAMPLING_X_NUM,
                                         _cfg.LOSS_SAMPLING_Y_NUM,
                                         _cfg.LOSS_SAMPLING_SCALE)

        # Mode architecture
        vgg = models.vgg19().features
        self.backbone = nn.Sequential(vgg, nn.AdaptiveAvgPool2d(output_size=(2, 2)))
        self.n_features = 512*2*2
        self.mlp = nn.Sequential(nn.Linear(self.n_features, 256),
                                  nn.BatchNorm1d(num_features=256),
                                  nn.ReLU(True),
                                  nn.Linear(256, 256),
                                  nn.BatchNorm1d(num_features=256),
                                  nn.ReLU(True),
                                  nn.Linear(256, 64),
                                  nn.BatchNorm1d(num_features=64),
                                  nn.ReLU(True))

        self.abxy = nn.Sequential(nn.Linear(64, 32),
                                  nn.BatchNorm1d(num_features=32),
                                  nn.ReLU(True),
                                  nn.Linear(32, 4),
                                  nn.Sigmoid())

        self.angle = nn.Sequential(nn.Linear(64, 32),
                                   nn.BatchNorm1d(num_features=32),
                                   nn.ReLU(True),
                                   nn.Linear(32, 1),
                                   nn.Tanh())

        # Validation
        self.val_data_index = 0


    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, self.n_features)
        x = self.mlp(x)
        return torch.cat([self.abxy(x), self.angle(x)], dim=1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        image, gt_params = train_batch
        pred_params = self.forward(image)
        pred_params = pred_params.cpu()
        gt_params = gt_params.cpu()
        loss = self.loss_fn(pred_params, gt_params)
        self.log("train_loss", loss, on_epoch=True, logger=True)
        return loss


    def _validation_step_internal(self, val_batch, batch_idx):
        image, gt_params= val_batch
        pred_params = self.forward(image)
        pred_params = pred_params.cpu()
        gt_params = gt_params.cpu()

        gt_params_2 = gt_params.clone()
        pred_params_2 = pred_params.clone()

        loss = self.loss_fn(pred_params, gt_params)

        pred_params_2[:, :4] *= self.crop_size
        gt_params_2[:, :4] *= self.crop_size
        pred_ellipses = build_ellipses_from_pred(pred_params_2)
        gt_ellipses = build_ellipses_from_pred(gt_params_2)

        ious = []
        for i, (pred, gt) in enumerate(zip(pred_ellipses, gt_ellipses)):
            img = image[i, ...].squeeze().permute(1, 2, 0).cpu().contiguous().numpy()
            img *= 255
            img = img.astype(np.uint8)

            if self.output_val_images is not None:
                draw_ellipse(img, gt, color=(255, 0, 0), thickness=3)
                draw_ellipse(img, pred, color=(0, 255, 0))
                cv2.imwrite(os.path.join(self.output_val_images,
                                         "img_%04d.png" % self.val_data_index),
                            img[:, :, ::-1])
                self.val_data_index += 1
            iou = compute_ellipses_iou(pred, gt)
            ious.append(iou)
        return {'test_loss': loss.item(), 'test_iou': ious}


    def validation_step(self, val_batch, batch_idx):
        res = self._validation_step_internal(val_batch ,batch_idx)
        self.log("val_loss", res["test_loss"], on_epoch=True, logger=True, prog_bar=True)
        self.log("val_ious", np.mean(res["test_iou"]), on_epoch=True, logger=True)

    def on_validation_start(self):
        # reinitialize the image index
        self.val_data_index = 0


    def test_step(self, batch, batch_idx):
        return self._validation_step_internal(batch, batch_idx)


    def test_epoch_end(self, outputs):
        losses = [x['test_loss'] for x in outputs]
        ious = np.hstack([x['test_iou'] for x in outputs])
        percentage_good = np.mean(np.asarray(ious) >= 0.8)
        self.log("mean iou", np.mean(ious))
        self.log("percent. iou >= 0.8",  percentage_good)

    def predict(self, img, device):
        """
            Do a prediction. It automatically rescales to the correct size but requires a square image as input.
            img: square PIL image
        """
        if type(img) is np.ndarray:
            img = Image.fromarray(img)

        if img.size[0] != img.size[1]:
            print("Image is not square")
            return None, None, None

        scale = self.crop_size / img.size[0]
        img_rescaled = img.resize((self.crop_size, self.crop_size))

        X = transforms.ToTensor()(img_rescaled)
        X = X.unsqueeze(0)
        Xdev = X.to(device)
        
        self.eval()
        with torch.no_grad():
            pred = self.forward(Xdev)
        pred = pred.cpu().numpy().flatten()
        pred[:4] *= self.crop_size
        angle = np.arcsin(pred[4])
        pred[:4] /= scale
        
        return pred[:2], angle, pred[2:4]

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if os.path.splitext(checkpoint_path)[1] == ".pth":
            self.load_state_dict(checkpoint)
        else:
            self.load_state_dict(checkpoint["state_dict"])
 


