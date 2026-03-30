import torch
import numpy as np
from PIL import Image

def load_target(path, size=64, device="cpu"):
    img = Image.open(path).convert('RGBA').resize((size, size), Image.LANCZOS)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:] # Premultiply
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

def generate_chaos(size, batch_size, device, density=0.1):
    x = torch.zeros(batch_size, 16, size, size, device=device)
    mask = (torch.rand(batch_size, 1, size, size, device=device) < density).float()
    x[:, 3:4] = mask
    x[:, :3] = torch.rand(batch_size, 3, size, size, device=device) * mask
    x[:, 4:] = torch.randn(batch_size, 12, size, size, device=device) * 0.1 * mask
    return x

class HybridPool:
    def __init__(self, size, device):
        self.size = size
        self.device = device
        self.pool = generate_chaos(64, size, device, density=0.15)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        batch = self.pool[idx].clone()
        # Reset d'un échantillon pour forcer l'apprentissage du chaos pur
        batch[0] = generate_chaos(64, 1, self.device, density=0.15)
        return batch, idx

    def commit(self, batch, idx):
        self.pool[idx] = batch