import torch


def ca_loss(pred, target, alpha=0.2, beta=0.05, gamma=0.0):
    """
    CA loss that punishes alive cells outside the target, on alive cells use
    a MSE loss on the visible channels weighted by the living mask
    """
    pred_rgb = pred[:, 1:4]
    target_rgb = target[:, :3]
    alive = pred[:, 0:1]
    # threshold for anti-aliasing backgrounds
    target_mask = (target_rgb.sum(dim=1, keepdim=True) > 0.05).float()
    # Core loss: only penalize RGB error where alive
    mse = ((alive * (pred_rgb - target_rgb)) ** 2).sum() / (alive.sum() + 1e-8)
    # Penalty for being alive outside target
    outside = alive * (1 - target_mask)
    outside_penalty = outside.sum() / (outside.numel() + 1e-8)
    # Reward for being alive inside target
    alive_in_target = (alive * target_mask).sum() / (target_mask.sum() + 1e-8)
    # regularize alive mask to be close to 0 or 1
    alive_reg = ((alive * (1 - alive)) ** 2).mean()
    # Net loss
    loss = mse + alpha * outside_penalty - beta * alive_in_target + gamma * alive_reg
    return loss
