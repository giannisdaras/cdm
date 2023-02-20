# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import matplotlib.pyplot as plt
import click



#----------------------------------------------------------------------------

def stochastic_sampler(net, x_cur, class_labels, S_churn, S_min, S_max, S_noise, num_steps, t_cur, t_next, i):
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    # Euler step.
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next, class_labels).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def det_sampler(net, x_cur, class_labels, num_steps, t_cur, t_next, i, second_order=False):
    x_hat = x_cur
    t_hat = t_cur

    # Euler step.
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1 and second_order:
        denoised = net(x_next, t_next, class_labels).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next




def generate_image_grid(
    network_pkl, dest_path,
    seed=0, gridw=8, gridh=8, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    same_latents=False, second_order=False,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
    if same_latents:
        latents = torch.randn([1, net.img_channels, net.img_resolution, net.img_resolution], device=device).repeat(batch_size, 1, 1, 1)
    else:
        latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        
    class_labels = None
    if net.label_dim:
        if same_latents:
            class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[1], device=device)]
            class_labels = class_labels.repeat([batch_size, 1])
        else:
            class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next
        x_next = det_sampler(net, x_cur, class_labels, num_steps, t_cur, t_next, i, second_order=second_order)

    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(dest_path)
    print('Done.')

#----------------------------------------------------------------------------


@click.command()

# Main options.
@click.option('--model_path',        help='Path to pre-trained model (pkl file)', metavar='DIR', type=str, required=True)
@click.option('--output_path',        help='Path for output file.', metavar='DIR', type=str, required=True)
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)

def main(model_path, output_path, num_steps):
    generate_image_grid(model_path,   output_path,  num_steps=num_steps)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
