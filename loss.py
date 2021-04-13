#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 22:06:30 2021

@author: mzins
"""

import pytorch_lightning as pl
import torch


class SamplingBasedLoss(pl.LightningModule):
    def __init__(self, range_x, range_y, sampling_x, sampling_y, scale):
        super(SamplingBasedLoss, self).__init__()
        self.scale = scale
        self.generate_sampling_points(range_x, range_y, sampling_x, sampling_y)
        
    
    def generate_sampling_points(self, range_x, range_y, sampling_x, sampling_y):
        samples_x = torch.linspace(*range_x, sampling_x)
        samples_y = torch.linspace(*range_y, sampling_y)
        # samples_x = torch.linspace(xmin, xmax, sampling_x, device=self.device)
        # samples_y = torch.linspace(ymin, ymax, sampling_y, device=self.device)
        y, x = torch.meshgrid(samples_y, samples_x)
        self.sampling_pts = torch.cat((x.flatten().view((1, -1)), 
                                       y.flatten().view((1, -1))), 0)
        # self.sampling_pts = nn.Parameter(torch.cat((x.flatten().view((1, -1)), 
        #                                             y.flatten().view((1, -1))), 0))

    def sample_ellipse(self, axes, sin, center):
        A = torch.diag(axes)
        cos = torch.sqrt(1 - sin**2)
        R = torch.cat((torch.cat((cos, -sin), dim=1),
                       torch.cat((sin, cos), dim=1)), dim=0)
        
        pts_centered = self.sampling_pts.T - center
        M = R @ A @ R.T @ pts_centered.T
        return torch.einsum("ij,ji->i", pts_centered, M)
        
    def forward(self, X, Y):
        X[:, :4] *= self.scale
        Y[:, :4] *= self.scale
        var = X[:, 5]
        vals = []
        for i in range(X.shape[0]):
            pts_X = self.sample_ellipse(X[i, :2], X[i, 4].view((1, 1)), X[i, 2:4])
            pts_Y = self.sample_ellipse(Y[i, :2], Y[i, 4].view((1, 1)), Y[i, 2:4])
            vals.append((torch.sqrt(torch.sum((pts_X - pts_Y)**2)) / pts_X.shape[0]).unsqueeze(0) * torch.exp(-var[i]))
        return torch.sum(torch.cat(vals))