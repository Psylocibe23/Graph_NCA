import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
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

# Data
target_idx = 9  
target_name = cfg['data']['targets'][target_idx]
dataset = SingleEmojiDataset('config.json', target_name)
seed, target_img = dataset[0]
seed = seed.unsqueeze(0).to(device)  # (1, C, H, W)
target_img = target_img.unsqueeze(0).to(device)  # (1, 3, H, W)

C = cfg["model"]["channels"]["C"]
H = cfg["data"]["canvas"]["P"] * cfg["data"]["canvas"]["H"]  # Multiplied by P because at pixel level we want the full canvas
W = cfg["data"]["canvas"]["P"] * cfg["data"]["canvas"]["W"]
K = cfg["model"]["iterations"]["K"]

# Model and Edges
edges = create_pixel_graph_edges(H, W)
model = HybridPixelGraphNca(C, edges, alpha=1.0, beta=1.0).to(device)

# Optimizer and Loss
lr = cfg["training"]["learning_rate"]
weight_decay = cfg["training"]["weight_decay"]
num_epochs = cfg["training"]["num_epochs"]

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.MSELoss()
sobel = SobelFilter(in_channels=3).to(device)
lambda_sobel = 0.1  # Weight for edge loss

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

    pred = model(seed, steps=K)  # (1, C, H, W)
    pred_rgb = pred[:, 1:4] 
    alive = pred[:, 0:1]  # Should be ~binary due to output sigmoid

    # ---- Main loss on RGB ----
    main_loss = loss_fn(pred_rgb, target_img)

    # ---- Edge loss ----
    edge_pred = sobel(pred_rgb)
    edge_target = sobel(target_img)
    edge_loss = F.mse_loss(edge_pred, edge_target)

    # ---- Ghost color penalty (RGB in non-target regions) ----
    target_mask = (target_img.sum(dim=1, keepdim=True) > 0).float()
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
        main_loss
        + lambda_sobel * edge_loss
        + lambda_ghost * ghost_penalty
        + lambda_hidden * hidden_penalty
        + lambda_alive * alive_binary_loss
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    writer.add_scalar('Loss/train', loss.item(), epoch)

    # ---- Save Results ----
    if (epoch + 1) % cfg["logging"]["log_interval"] == 0:
        print(f"[LOG] Epoch {epoch+1:04d} - Loss: {loss.item():.6f}")

    if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
        # Save predicted RGB
        pred_np = pred_rgb.detach().cpu().squeeze(0).clamp(0, 1).permute(1,2,0).numpy()
        plt.imsave(f"{results_dir}/pred_{epoch+1:04d}.png", pred_np)
        # Save comparison
        save_comparison(target_img, pred_rgb, epoch + 1, results_dir)

        # Save Sobel edge maps
        edge_pred_np = edge_pred.detach().cpu().squeeze(0).clamp(0, 1).mean(0).numpy()
        edge_target_np = edge_target.detach().cpu().squeeze(0).clamp(0, 1).mean(0).numpy()
        plt.imsave(f"{results_dir}/edge_pred_{epoch+1:04d}.png", edge_pred_np, cmap='gray')
        plt.imsave(f"{results_dir}/edge_target_{epoch+1:04d}.png", edge_target_np, cmap='gray')

# Save Model and Log
torch.save(model.state_dict(), os.path.join(results_dir, "model_final.pth"))
print(f"\nTraining complete for {target_name}. Final model and images saved.")

experiment_log = {
    "target": target_name,
    "config": cfg,
    "final_loss": loss.item(),
    "loss_function": "MSE + Sobel Edge Loss",
    "model_type": "HybridPixelGraphNca",
    "K": K,
    "num_epochs": num_epochs,
}
with open(os.path.join(results_dir, "experiment_log.json"), "w") as f:
    json.dump(experiment_log, f, indent=2)
