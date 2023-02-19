import torch
import numpy as np


def churn_sampler(net, x_cur, class_labels, S_churn, S_min, S_max, S_noise, num_steps, t_cur, t_next, i, second_order=False):
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    # Euler step.
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1 and second_order:
        denoised = net(x_next, t_next, class_labels).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next, denoised


def backward_sde_sampler(net, x_hat, class_labels, num_steps, t_cur, t_next, i, second_order=False, augment_labels=None, **kwargs):
    t_hat = t_cur
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + 2 * (t_next - t_hat) * d_cur  + torch.sqrt(2 * (t_hat - t_next).abs() * t_cur) * torch.randn_like(x_hat)

    if i < num_steps - 1 and second_order:
        denoised = net(x_next, t_next, class_labels, augment_labels=augment_labels).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + 2 * (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)  + torch.sqrt(2 * (t_hat - t_next).abs() * t_cur) * torch.randn_like(x_hat)
    return x_next, denoised