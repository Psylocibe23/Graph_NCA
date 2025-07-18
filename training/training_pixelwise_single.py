import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from utils.data_loader import SingleEmojiDataset
from utils.graph_utils import create_pixel_graph_edges
from modules.hybrid_nca import HybridPixelGraphNca
from modules.sobel_filters import SobelFilter  

# Config and Device
cfg = json.load(open('config.json'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data (now with batch!)
target_idx = 5
target_name = cfg['data']['targets'][target_idx]
dataset = SingleEmojiDataset('config.json', target_name)
seed, target_img = dataset[0]
seed = seed.unsqueeze(0).to(device)  # (1, C, H, W)
target_img = target_img.unsqueeze(0).to(device) # (1, 3, H, W)

C = cfg["model"]["channels"]["C"]
H = cfg["data"]["canvas"]["P"] * cfg["data"]["canvas"]["H"]
W = cfg["data"]["canvas"]["P"] * cfg["data"]["canvas"]["W"]
BATCH_SIZE = 8
MIN_STEPS = 32
MAX_STEPS = 54

# Model and Edges
edges = create_pixel_graph_edges(H, W)
model = HybridPixelGraphNca(C, edges, alpha=1.0, beta=1.0).to(device)

# Optimizer and Loss
lr = cfg["training"]["learning_rate"]
weight_decay = cfg["training"]["weight_decay"]
num_epochs = cfg["training"]["num_epochs"]

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
sobel = SobelFilter(in_channels=3).to(device)
lambda_sobel = 0.1

# Logging and Saving
results_dir = f"results/pixelwise_{target_name}"
os.makedirs(results_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(results_dir, "tb_logs"))

def save_comparison(target_img, pred_rgb, epoch, save_dir):
    timg = target_img.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    pimg = pred_rgb.detach().cpu().squeeze(0).clamp(0,1).permute(1,2,0).numpy()
    fig, axs = plt.subplots(1,2, figsize=(7,3))
    axs[0].imshow(timg)
    axs[0].set_title('Target')
    axs[1].imshow(pimg)
    axs[1].set_title('Prediction')
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_{epoch:04d}.png")
    plt.close(fig)

save_interval = 10

print("Seed shape:", seed.shape)
print("Target shape:", target_img.shape)

# ============ Training Loop ============

for epoch in trange(num_epochs, desc="Training"):
    model.train()
    optimizer.zero_grad()
    
    # --- Generate a batch of seeds (some fresh, some from history) ---
    seeds = seed.repeat(BATCH_SIZE, 1, 1, 1)  # (B, C, H, W)
    targets = target_img.repeat(BATCH_SIZE, 1, 1, 1)  # (B, 3, H, W)

    # (Optional) Randomly “damage” or “reset” some seeds for regeneration ability
    """if epoch > 0:
        with torch.no_grad():
            mask = torch.rand(BATCH_SIZE, 1, 1, 1, device=device) < 0.5
            seeds = torch.where(mask, seed, seeds)""" 

    # --- Random K steps per batch ---
    K = np.random.randint(MIN_STEPS, MAX_STEPS + 1)
    pred = model(seeds, steps=K)  # (B, C, H, W)
    pred_rgb = pred[:, 1:4]
    alive = pred[:, 0:1]

    # ---- Main loss: MSE
    alive_mask = (alive > 0.1).float()
    loss_main = (((pred_rgb - targets) ** 2) * alive_mask).sum() / (alive_mask.sum() + 1e-8)

    # ---- Edge loss ----
    edge_pred = sobel(pred_rgb)
    edge_target = sobel(targets)
    edge_loss = F.mse_loss(edge_pred, edge_target)

    # ---- Ghost color penalty ----
    target_mask = (targets.sum(dim=1, keepdim=True) > 0).float()
    ghost_penalty = ((pred_rgb ** 2) * (1 - target_mask)).mean()
    lambda_ghost = 0.001

    # ---- Hidden channel penalty ----
    hidden_channels = pred[:, 4:]
    hidden_penalty = (hidden_channels ** 2).mean()
    lambda_hidden = 0.001

    # ---- Alive mask binary regularization ----
    alive_binary_loss = (alive * (1 - alive)).mean()
    lambda_alive = 0.001

    # ---- Combine losses ----
    loss = (
        F.mse_loss(pred_rgb, targets)
        #loss_main          # Alive-masked MSE on RGB channels
        #+ lambda_sobel * edge_loss    # Sobel edge similarity (optional)
        #+ lambda_ghost * ghost_penalty  # Penalize RGB outside target area
        #+ lambda_hidden * hidden_penalty  # Keep hidden channels small (optional)
        #+ lambda_alive * alive_binary_loss  # Encourage alive mask to be near 0 or 1 (optional)
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    writer.add_scalar('Loss/train', loss.item(), epoch)

    # ---- Save Results ----
    if (epoch + 1) % cfg["logging"]["log_interval"] == 0:
        print(f"[LOG] Epoch {epoch+1:04d} - Loss: {loss.item():.6f}")

    if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
        # Save predicted RGB for first sample in batch
        pred_np = pred_rgb[0].detach().cpu().clamp(0, 1).permute(1,2,0).numpy()
        plt.imsave(f"{results_dir}/pred_{epoch+1:04d}.png", pred_np)
        save_comparison(target_img, pred_rgb[0:1], epoch + 1, results_dir)

        edge_pred_np = edge_pred[0].detach().cpu().clamp(0, 1).mean(0).numpy()
        edge_target_np = edge_target[0].detach().cpu().clamp(0, 1).mean(0).numpy()
        plt.imsave(f"{results_dir}/edge_pred_{epoch+1:04d}.png", edge_pred_np, cmap='gray')
        plt.imsave(f"{results_dir}/edge_target_{epoch+1:04d}.png", edge_target_np, cmap='gray')

# Save Model and Log
torch.save(model.state_dict(), os.path.join(results_dir, "model_final.pth"))
print(f"\nTraining complete for {target_name}. Final model and images saved.")

experiment_log = {
    "target": target_name,
    "config": cfg,
    "final_loss": float(loss.item()),
    "loss_function": "MSE (alive-masked) + Edge + Ghost + Hidden + AliveReg",
    "model_type": "HybridPixelGraphNca",
    "K": f"{MIN_STEPS}-{MAX_STEPS}",
    "num_epochs": num_epochs,
}
with open(os.path.join(results_dir, "experiment_log.json"), "w") as f:
    json.dump(experiment_log, f, indent=2)
