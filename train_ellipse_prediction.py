#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:56:29 2021

@author: mzins
"""
import argparse
import glob
import json
import logging
import os
import time

import cv2
import numpy as np

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from config import _cfg
from dataset import EllipsesDataset
from model import EllipsePredictor
from loss import SamplingBasedLoss
from scene_loader import Scene_loader



def run_training(training_dataset_file, validation_dataset_file, obj,
                 nb_epochs, out_checkpoints_folder, out_logs_folder,
                 validation_interval, save_weights_only=False):
    """
        Run the training for a given object id
    """
    cat_id = obj["category_id"]
    obj_id = obj["object_id"]
    print("========================================")
    print("  Start training object: %d (class %d)" % (obj_id, cat_id))
    print("========================================")
    name = "obj_%02d_%02d" % (cat_id, obj_id)
    dataset_train = EllipsesDataset(training_dataset_file, obj_id,
                                    crop_size=_cfg.CROP_SIZE,
                                    transforms=transforms.Compose([
                                        transforms.ColorJitter(*_cfg.TRAIN_COLOR_JITTER),
                                        transforms.ToTensor()]),
                                    random_shift=_cfg.TRAIN_RANDOM_SHIFT,
                                    random_rotation=_cfg.TRAIN_RANDOM_ROTATION,
                                    random_perspective=_cfg.TRAIN_RANDOM_PERSPECTIVE,
                                    random_blur=_cfg.TRAIN_RANDOM_BLUR,
                                    probability_do_augm=_cfg.TRAIN_PROBA_DO_AUGM)

    dataset_valid = EllipsesDataset(validation_dataset_file, obj_id, 
                                    crop_size=_cfg.CROP_SIZE,
                                    transforms=transforms.Compose([
                                        transforms.ColorJitter(*_cfg.VALID_COLOR_JITTER),
                                        transforms.ToTensor()]),
                                    random_shift=_cfg.VALID_RANDOM_SHIFT,
                                    random_rotation=_cfg.VALID_RANDOM_ROTATION,
                                    random_perspective=_cfg.VALID_RANDOM_PERSPECTIVE,
                                    random_blur=_cfg.VALID_RANDOM_BLUR)


    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(out_logs_folder, name))

    training_loader = torch.utils.data.DataLoader(dataset_train, 
                                                  batch_size=_cfg.TRAIN_BATCH_SIZE,
                                                  shuffle=_cfg.TRAIN_BATCH_SHUFFLE,
                                                  num_workers=_cfg.TRAIN_LOADER_NUM_WORKERS,
                                                  drop_last=True)

    validation_loader = torch.utils.data.DataLoader(dataset_valid, 
                                                    batch_size=_cfg.VALID_BATCH_SIZE,
                                                    shuffle=_cfg.VALID_BATCH_SHUFFLE,
                                                    num_workers=_cfg.VALID_LOADER_NUM_WORKERS)
    # Model
    model = EllipsePredictor(lr=_cfg.LEARNING_RATE,
                             crop_size=_cfg.CROP_SIZE,
                             output_val_images=None)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(out_checkpoints_folder, name),
        save_last=True,
        filename="best",
        save_top_k=1,
        verbose=True,
        monitor='val_ious',
        mode='max',
        prefix='ckpt',
        save_weights_only=save_weights_only
    )


    # Training
    ta = time.time()

    trainer = pl.Trainer(gpus=1, logger=tb_logger, max_epochs=nb_epochs,
                         check_val_every_n_epoch=validation_interval,
                         log_every_n_steps=50,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model, training_loader, validation_loader)

    print("Finished in %.2fs" % (time.time()-ta))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("scene", help="<Required> Input Scene file containing the objects (.json)")
    parser.add_argument("training", help="<Required> Input training dataset file (.json)")
    parser.add_argument("validation", help="<Required> Input validation dataset file (.json)")
    parser.add_argument("checkpoints", help="<Required> Output folder where to save checkpoints.")
    parser.add_argument("--logs", help="Output folder where to output training logs.",
                        default="logs")
    parser.add_argument("--nb_epochs", help="Number of training epochs for each object.",
                        default=300, type=int)
    parser.add_argument("--valid_interval", help="Interval of epochs between validations.",
                        default=10, type=int)
    parser.add_argument("--object_id", help="<Optional> Specify a single object id for training.",
                        default=None, type=int)
    parser.add_argument("--save_all_parameters", action="store_true", 
                        help="Save all the hyper-parameters in checkpoints, "
                             "not only weights (default is False).",
                        default=False)


    args = parser.parse_args()


    scene_file = args.scene
    training_dataset_file = args.training
    validation_dataset_file = args.validation
    out_checkpoints_folder = args.checkpoints
    out_logs_folder = args.logs
    nb_epochs = args.nb_epochs
    validation_interval = args.valid_interval
    save_all_parameters = args.save_all_parameters
    save_weights_only = not save_all_parameters
    only_object_id = args.object_id


    scene = Scene_loader(scene_file)

    print("List of objects:")
    for obj in scene:
        print(" - obj_%02d_%02d" % (obj["category_id"], obj["object_id"]))
    print()
        
    if only_object_id is not None and only_object_id >= 0 and only_object_id < len(scene):
        print("Train only object", only_object_id)

        obj = scene.get_object_by_id(only_object_id)
        run_training(training_dataset_file, validation_dataset_file, obj,
                     nb_epochs=nb_epochs,
                     validation_interval=validation_interval,
                     out_checkpoints_folder=out_checkpoints_folder,
                     out_logs_folder=out_logs_folder,
                     save_weights_only=save_weights_only)
    else:
        for obj in scene:
            run_training(training_dataset_file, validation_dataset_file, obj,
                        nb_epochs=nb_epochs,
                        validation_interval=validation_interval,
                        out_checkpoints_folder=out_checkpoints_folder,
                        out_logs_folder=out_logs_folder,
                        save_weights_only=save_weights_only)


