import torch
import torch.nn as nn
import torch.nn.functional as F

class EvolutiveNCA(nn.Module):
    def __init__(self, channel_n=16, hidden_n=256):
        super().__init__()
        self.channel_n = channel_n
        
        # Branche Lenia (Mouvement et Évolution)
        self.lenia_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_n, channel_n, 1)
        )
        
        # Branche Target (Reconstruction d'Image)
        self.target_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_n, channel_n, 1)
        )
        
        # Initialisation à zéro pour la branche Target (stabilité)
        nn.init.zeros_(self.target_net[-1].weight)
        nn.init.zeros_(self.target_net[-1].bias)

        # Pré-calcul des filtres de perception pour éviter toute recréation en boucle.
        fx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        fy = fx.t()
        sobel = torch.stack([fx, fy]).unsqueeze(1)  # (2, 1, 3, 3)
        kernel = sobel.repeat(self.channel_n, 1, 1, 1)  # (2*channel_n, 1, 3, 3)
        self.register_buffer("perception_kernel", kernel)

    def perceive(self, x):
        kernel = self.perception_kernel.to(dtype=x.dtype)
        grads = F.conv2d(x, kernel, padding=1, groups=self.channel_n)
        return torch.cat([x, grads], dim=1)

    def forward(self, x, alpha=0.0):
        p = self.perceive(x)
        
        # Calcul des forces Lenia vs Target
        dx_l = self.lenia_net(p)
        dx_t = self.target_net(p)
        
        # Mélange (alpha=0: Lenia, alpha=1: Target)
        dx = torch.lerp(dx_l, dx_t, alpha)
        
        # Mise à jour stochastique
        update_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < 0.5).to(x.dtype)
        x = x + dx * update_mask
        
        # Masque de vie
        alive = (F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1).to(x.dtype)
        return x * alive