import torch


def ca_loss(pred, target, alpha=0.1, beta=0.1):
    """
    CA loss that punishes alive cells outside the target, on alive cells use
    a MSE loss on the visible channels weighted by the living mask
    """
    pred_rgb = pred[:, 1:4]
    target_rgb = target[:, :3]
    alive = pred[:, 0:1]
    target_mask = (target_rgb.sum(dim=1, keepdim=True) > 0).float()
    mse = ((alive * (pred_rgb - target_rgb)) ** 2).sum() / (alive.sum() + 1e-8)
    outside = alive * (1 - target_mask)
    outside_penalty = outside.sum() / (outside.numel() + 1e-8)
    alive_in_target = (alive * target_mask).sum() / (target_mask.sum() + 1e-8)
    loss = mse + alpha * outside_penalty - beta * alive_in_target
    return loss
