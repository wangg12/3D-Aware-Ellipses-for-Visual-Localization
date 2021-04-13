#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:38:33 2021

@author: mzins
"""

import glob
import os
import shutil


folder = "/home/mzins/dev/3D-Aware_Ellipses_for_Visual_Localization/checkpoints_log_bn"
list_folders = glob.glob(os.path.join(folder, "*"))
os.mkdir(os.path.join(folder, "weights_best"))
os.mkdir(os.path.join(folder, "weights_last"))

for fold in sorted(list_folders):
    ckpts = sorted(glob.glob(os.path.join(fold, "*.ckpt")))
    ckpt_best = ckpts[0]
    ckpt_last = ckpts[1]
    name = os.path.basename(fold)
    
    name_best = name + "_" + "ckpt_best.pth"
    name_last = name + "_" + "ckpt_last.pth"
    shutil.move(ckpt_best, os.path.join(folder, "weights_best", name_best))
    shutil.move(ckpt_last, os.path.join(folder, "weights_last", name_last))
    print("moved", name_best, name_last)
    
    
    
    
                  
    