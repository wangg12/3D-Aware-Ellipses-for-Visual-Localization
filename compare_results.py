#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:01:42 2021

@author: mzins
"""

import numpy as np
import matplotlib.pyplot as plt


pos_err_ref = np.loadtxt("/home/mzins/dev/Learning_Uncertainties/ellipse_from_object/7-Scenes/analysis_new_training_test/seq-02/best/SAVE_errors.txt")
pos_err = np.loadtxt("/home/mzins/dev/3D-Aware_Ellipses_for_Visual_Localization/ERRORS/SAVE_errors.txt")[:1000]


plt.plot(pos_err_ref, ".", markersize=3, label="ref")
plt.plot(pos_err, ".", markersize=1, label="new")
plt.legend()