# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from torch_utils.misc import edm_schedule, save_image
import numpy as np
from training.samplers import backward_sde_sampler
from random import random

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, self_cond=False):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.self_cond = self_cond

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma


        if self.self_cond and random() < 0.5:
            with torch.no_grad():
                cat_input = torch.cat([y + n, torch.zeros_like(y)], axis=1)
                denoised = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :3]
                denoised = denoised.detach()
        else:
            denoised = torch.zeros_like(y + n)
        
        D_yn = net(torch.cat([y + n, denoised], axis=1), sigma, labels, augment_labels=augment_labels)[:, :3]

        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
                                # CDM (Martingale) Loss
#----------------------------------------------------------------------------


@persistence.persistent_class
class MartingaleLoss:
    """ensures that the expected generated image is not changing."""
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, epsilon_min=0.0, epsilon_max=0.05, num_steps=6, rho=7, 
        martingale_lambda=2., S_churn=10.0, S_min=0.01, S_max=1.0, S_noise=1.007):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.num_steps = num_steps
        self.rho = rho
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.martingale_lambda = martingale_lambda
        
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # save_image(images, "before_augmentations.png")
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # save_image(y, "after_augmentations.png")

        n = torch.randn_like(y) * sigma

        # Regular loss computation
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn_initial = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss1 = weight * ((D_yn_initial - y) ** 2)


        # repeat one time
        labels = labels.repeat([2, 1])
        augment_labels = augment_labels.repeat([2, 1])
        sigma_max = torch.clone(sigma)

        # get sigma deviations
        stretch = self.epsilon_max - self.epsilon_min
        epsilon = torch.rand_like(sigma) * stretch
        # TODO(@giannisdaras): fix this hardcoded value
        sigma_min = torch.maximum(sigma - epsilon, torch.ones_like(sigma) * 0.002)

        t_steps = edm_schedule(sigma_max=torch.unsqueeze(sigma_max, 1), sigma_min=torch.unsqueeze(sigma_min, 1), num_steps=self.num_steps)
        t_steps = t_steps.repeat([2, 1, 1, 1, 1])
        x_next = y.repeat([2, 1, 1, 1]) + n.repeat([2, 1, 1, 1])
        
        with torch.no_grad():
            for i in range(self.num_steps - 1):
                t_cur = t_steps[:, :, :, :, i]
                t_next = t_steps[:, :, :, :, i + 1]
                x_cur = x_next
                x_next, _ = backward_sde_sampler(net, x_cur, labels, self.num_steps, t_cur, t_next, i, second_order=False, augment_labels=augment_labels)
        
        D_yn = net(x_next, sigma_min.repeat([2, 1, 1, 1]), labels, augment_labels=augment_labels)
        x_hat_1 = D_yn[:D_yn.shape[0] // 2]
        x_hat_2 = D_yn[-D_yn.shape[0] // 2:]
        loss2 = (x_hat_1 - D_yn_initial) * (x_hat_2 - D_yn_initial)
        return loss1 + self.martingale_lambda * loss2
