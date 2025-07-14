import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from utils.data_loader import SingleEmojiDataset
from utils.graph_utils import create_touching_edges
from modules.graph_nca import GraphNCA
from utils.losses import ca_loss
from modules.sobel_filters import SobelFilter
import json
import os
import datetime 
from tqdm import trange 
from torch.utils.tensorboard import SummaryWriter



# Load config
cfg = json.load(open('config.json'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Choose the target emoji (by index)
target_idx = 2
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
writer = SummaryWriter(log_dir=f"results/{target_name}/tb_logs")
loss_fn = nn.MSELoss()

save_interval = 10  # Save images every 10 epochs
sobel = SobelFilter(in_channels=3).to(device)
lambda_sobel = 0.1  # weight of the edge loss

for epoch in trange(num_epochs, desc="Training"):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    pred = model(seed, edges)  # (1, C, PH, PW)
    pred_rgb = pred[:, 1:4]  # (1, 3, Ph, PW)
    main_loss = loss_fn(pred_rgb, target_img)
    # Sobel edge loss
    edge_pred = sobel(pred_rgb)
    edge_target = sobel(target_img)
    edge_loss = F.mse_loss(edge_pred, edge_target)
    # Combine
    loss = main_loss + lambda_sobel * edge_loss
    loss.backward()
    optimizer.step()

    writer.add_scalar('Loss/train', loss.item(), epoch)

    if (epoch + 1) % cfg["logging"]["log_interval"] == 0:
        print(f"\n[LOG] Epoch {epoch+1:04d} - Loss: {loss.item():.6f}")

    if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
        # Save alive mask
        alive_mask = pred[:, 0:1].detach().cpu().squeeze(0).clamp(0, 1).squeeze(0).numpy()
        plt.imsave(f"results/{target_name}/alive_{epoch+1:04d}.png", alive_mask, cmap='gray')

        # Save predicted RGB
        pred_np = pred_rgb.detach().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()
        plt.imsave(f"results/{target_name}/pred_{epoch+1:04d}.png", pred_np)

        # Save comparison
        save_comparison(target_img, pred_rgb, epoch + 1, f"results/{target_name}")
        print("Predicted RGB shape:", pred_rgb.shape)

        # Save Sobel edge maps
        edge_pred_np = edge_pred.detach().cpu().squeeze(0).clamp(0, 1).mean(0).numpy()
        edge_target_np = edge_target.detach().cpu().squeeze(0).clamp(0, 1).mean(0).numpy()
        plt.imsave(f"results/{target_name}/edge_pred_{epoch+1:04d}.png", edge_pred_np, cmap='gray')
        plt.imsave(f"results/{target_name}/edge_target_{epoch+1:04d}.png", edge_target_np, cmap='gray')



torch.save(model.state_dict(), f"results/{target_name}/model_final.pth")
print(f"\nTraining complete for {target_name}. Final model and images saved.")

experiment_log = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "target": target_name,
    "config": cfg,
    "final_loss": loss.item(),
    "loss_function": "Cellular Automata Loss",
    "attention": cfg["model"]["attention"]["type"],
    "K": K,
    "num_epochs": num_epochs,
}
with open(f"results/{target_name}/experiment_log.json", "w") as f:
    json.dump(experiment_log, f, indent=2)