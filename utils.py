import torch
import numpy as np
from PIL import Image

TAILLE = 64
def load_target(path, grid_size=TAILLE, device="cpu"):
    img = Image.open(path).convert('RGBA').resize((grid_size, grid_size), Image.LANCZOS)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:] # Premultiply
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

def generate_chaos(grid_size, batch_size, device, density=0.1):
    x = torch.zeros(batch_size, 16, grid_size, grid_size, device=device)
    mask = (torch.rand(batch_size, 1, grid_size, grid_size, device=device) < density).float()
    x[:, 3:4] = mask
    x[:, :3] = torch.rand(batch_size, 3, grid_size, grid_size, device=device) * mask
    x[:, 4:] = torch.randn(batch_size, 12, grid_size, grid_size, device=device) * 0.1 * mask
    return x

class HybridPool:
    def __init__(self, pool_size, device):
        self.size = pool_size
        self.device = device
        self.pool = generate_chaos(TAILLE, pool_size, device, density=0.15)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        batch = self.pool[idx].clone()
        # Reset d'un échantillon pour forcer l'apprentissage du chaos pur
        batch[0] = generate_chaos(TAILLE, 1, self.device, density=0.15)
        return batch, idx

    def commit(self, batch, idx):
        self.pool[idx] = batch