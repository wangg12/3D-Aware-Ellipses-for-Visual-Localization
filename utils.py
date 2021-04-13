#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:56:31 2021

@author: mzins
"""

import math
import os


def force_square_bbox(bbox, margin=0):
    """
        Transform a box into a square box.
        - bbox : 2x2 matrix with first corner on first row and second corner 
                 on the second row or 1D list [x_min, y_min, x_max, y_max]
        - margin: (optional) margin applied on the left, right top and bottom
    """
    x1, y1, x2, y2 = bbox.flatten().tolist()
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    d = math.ceil(max(w, h) / 2)
    d += margin

    cx = round((x1 + x2) / 2)
    cy = round((y1 + y2) / 2)

    return list(map(int, [cx-d, cy-d, cx+d, cy+d]))


def create_if_needed(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

