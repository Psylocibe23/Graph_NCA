import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader import SingleEmojiDataset
from utils.graph_utils import create_touching_edges
from modules.graph_nca import GraphNCA
import json
import os


# Load config
cfg = json.load(open('config.json'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Choose the target emoji (by index)
target_idx = 0
target_name = cfg['data']['targets'][target_idx]
print(f"Training on target: {target_name}")

# Data loading
dataset = SingleEmojiDataset('config.json', target_name)
seed, target_img = dataset[0]
seed = seed.unsqueeze(0).to(device)  # (1, C, PH, PW)
target_img = target_img.unsqueeze(0).to(device)  # (1, 3, PH, PW)

C = cfg["model"]["channels"]["C"]
d = cfg["model"]["channels"]["d"]
H = cfg["data"]["canvas"]["H"]
W = cfg["data"]["canvas"]["W"]
K = cfg["model"]["iterations"]["K"]
attention_hidden = cfg["model"]["attention"]["mlp_hidden"]
attention_layers = cfg["model"]["attention"]["mlp_layers"]
P = cfg["data"]["canvas"]["P"]

model = GraphNCA(C, H, W, d, K, attention_hidden, attention_layers).to(device)
edges = create_touching_edges(P)

# Training hyperparameters
lr = cfg["training"]["learning_rate"]
weight_decay = cfg["training"]["weight_decay"]
num_epochs = cfg["training"]["num_epochs"]

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.MSELoss()

os.makedirs(f"results/{target_name}", exist_ok=True)



import matplotlib.pyplot as plt
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

print("Target image shape:", target_img.shape)
print("Seed shape:", seed.shape)

target_mask = (target_img.sum(dim=1, keepdim=True) > 0).float()  # (1,1,H,W)

# Training loop
from tqdm import trange
for epoch in trange(num_epochs, desc="Training"):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    pred = model(seed, edges)  # (1, C, PH, PW)
    # Extract RGB channels (1,2,3)
    # Alive mask: (1, 1, PH, PW)
    alive_mask = pred[:, 0:1]
    # Only paint RGB where alive
    pred_rgb_alive = pred[:, 1:4] * alive_mask
    # Loss inside target
    loss_in = loss_fn(pred_rgb_alive * target_mask, target_img * target_mask)
    # Loss for "alive" cells outside target
    loss_out = (alive_mask * (1 - target_mask)).mean()
    loss = loss_in + 0.1 * loss_out
    loss.backward()
    optimizer.step()

    # Logging
    if (epoch + 1) % cfg["logging"]["log_interval"] == 0:
        print(f"\n[LOG] Epoch {epoch+1:04d} - Loss: {loss.item():.6f}")

    if (epoch + 1) % cfg["logging"]["save_interval"] == 0 or epoch == num_epochs - 1:
        am = alive_mask.detach().cpu().squeeze(0).clamp(0, 1).squeeze(0).numpy()
        plt.imsave(f"results/{target_name}/alive_{epoch+1:04d}.png", am, cmap='gray')

        pred_np = pred_rgb_alive.detach().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()
        plt.imsave(f"results/{target_name}/pred_{epoch+1:04d}.png", pred_np)
        save_comparison(target_img, pred_rgb_alive, epoch+1, f"results/{target_name}")
        print("Predicted RGB shape:", pred_rgb_alive.shape)

torch.save(model.state_dict(), f"results/{target_name}/model_final.pth")
print(f"\nTraining complete for {target_name}. Final model and images saved.")