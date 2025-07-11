import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class SingleEmojiDataset(Dataset):
    """
    Dataset that returns (seed_canvas, target_mask) for a single emoji.
    Used for single emoji target training (self-organizing patterns).
    """
    def __init__(self, config_path, target_name):
        super().__init__()
        # Load configurations
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            data_cfg = cfg['data']
            model_cfg = cfg['model']

        # Set canvas and paths
        self.emojis_dir = data_cfg['emojis_dir']
        self.P = data_cfg['canvas']['P']
        self.H = data_cfg['canvas']['H']
        self.W = data_cfg['canvas']['W']
        self.C = model_cfg['channels']['C']  # 3 rgb channels, 1 state channel, C-4 hidden channels

        # Load target image
        img_path = os.path.join(self.emojis_dir, target_name)
        img = Image.open(img_path).convert("RGB")  
        # Resize (if needed)
        self.to_tensor = T.Compose([
            T.Resize((self.P*self.H, self.P*self.W)),
            T.ToTensor(),
        ])
        self.target = self.to_tensor(img).squeeze(0)  # (3, PH, PW)

        # Precompute canvas
        canvas = torch.zeros(self.C, self.P*self.H, self.P*self.W)
        # Set the center pixel to alive (i.e. channel 0 equals to 1)
        cy, cx = (self.P*self.H)//2, (self.P*self.W)//2
        canvas[0, cy, cx] = 1.0
        self.seed = canvas  # (C, P*H, P*W)

    def __len__(self):
        # We only have one sample
        return 1
    
    def __getitem__(self, idx):
        """
        returns:
        - seed (C, P*H, P*W)  the initial global state
        - target (3, P*H, P*W)  The target image our NCAs will grow to
        """
        return self.seed.clone(), self.target.clone()
    


if __name__=='__main__':
    cfg = json.load(open('config.json'))
    target = cfg['data']['targets'][0]
    print(f"Running smoke test on {target}")
    ds = SingleEmojiDataset('config.json', target)
    seed, img = ds[0]
    print(f"Seed shape: {seed.shape}")
    print(f"Target shape:{img.shape}")
    cy, cx = seed.shape[1]//2, seed.shape[2]//2
    print("Seed center value:", seed[0,cy,cx].item())
    print("Target range:", img.min().item(), "to", img.max().item())
    # 5) Visualize seed channel-0
    seed_ch0 = seed[0].mul(255).byte().cpu().numpy()
    seed_pil = Image.fromarray(seed_ch0, mode='L')
    seed_pil.show(title="Seed (channel 0)")
    # 6) Visualize target RGB
    tgt_np = img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    tgt_pil = Image.fromarray(tgt_np, mode='RGB')
    tgt_pil.show(title=f"Target: {target}")
