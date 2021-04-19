#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 22:45:10 2021

@author: mzins
"""


class Config():
    def __init__(self):
        pass
    
    
_cfg = Config()


# Data
_cfg.CROP_SIZE = 256

# Data Transformations (use None to disable)
# ColorJitter: brightness, contrast, saturation, hue
# The values are chosen uniformly in the following intervals
# [1-BRIGHTNESS, 1+BRIGHTNESS]
# [1-CONTRAST, 1+CONTRAST]
# [1-SATURATION, 1+SATURATION]
# [-HUE, HUE]
_cfg.TRAIN_COLOR_JITTER = [0.1, 0.1, 0.1, 0.05]
_cfg.TRAIN_RANDOM_SHIFT = [-20, 20]
_cfg.TRAIN_RANDOM_ROTATION = [-30, 30]
_cfg.TRAIN_RANDOM_PERSPECTIVE = [-5, 5]
_cfg.TRAIN_RANDOM_BLUR = 3
_cfg.TRAIN_PROBA_DO_AUGM = 0.5   # probability of applyin each of the augmentations

_cfg.VALID_COLOR_JITTER = [0.0, 0.0, 0.0, 0.0]
_cfg.VALID_RANDOM_SHIFT = None
_cfg.VALID_RANDOM_ROTATION = None
_cfg.VALID_RANDOM_PERSPECTIVE = None
_cfg.VALID_RANDOM_BLUR = None
_cfg.VALID_PROBA_AUGM = 0.0  # probability of applyin each of the augmentations


# Training
_cfg.TRAIN_BATCH_SIZE = 16
_cfg.TRAIN_BATCH_SHUFFLE = True
_cfg.TRAIN_LOADER_NUM_WORKERS = 16

_cfg.VALID_BATCH_SIZE = 16
_cfg.VALID_BATCH_SHUFFLE = False
_cfg.VALID_LOADER_NUM_WORKERS = 16

_cfg.LEARNING_RATE = 5e-5


# Loss
_cfg.LOSS_SAMPLING_X_INTERVAL = [-2, 12]
_cfg.LOSS_SAMPLING_Y_INTERVAL = [-2, 12]
_cfg.LOSS_SAMPLING_X_NUM = 25
_cfg.LOSS_SAMPLING_Y_NUM = 25
_cfg.LOSS_SAMPLING_SCALE = 10


# Detector
_cfg.DETECTOR_ARCHITECTURE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
_cfg.DETECTOR_NUM_CLASSES = 7
_cfg.DETECTOR_SCORE_THRESH_TEST = 0.5
_cfg.DETECTOR_DATALOADER_NUM_WORKERS = 8
_cfg.DETECTOR_IMS_PER_BATCH = 2
_cfg.DETECTOR_BASE_LR = 0.00025
_cfg.DETECTOR_NB_EPOCHS = 2000
_cfg.DETECTOR_BATCH_SIZE_PER_IMAGE = 64
