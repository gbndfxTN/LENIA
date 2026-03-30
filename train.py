import os
from contextlib import nullcontext
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from model import EvolutiveNCA
from utils import load_target, HybridPool

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITERATIONS = 20000
SAVE_EVERY = 500
LOG_DIR = "logs/hybrid_nca"
TARGET_IMG = "heart.png"
BATCH_SIZE = 4

def train():
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    target = load_target(TARGET_IMG, device=DEVICE)
    target_batch = target.expand(BATCH_SIZE, -1, -1, -1)
    model = EvolutiveNCA().to(DEVICE)
    use_amp = DEVICE.type == "cuda"
    if use_amp:
        def amp_context():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        def amp_context():
            return nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Optimiseurs séparés pour chaque cerveau
    opt_lenia = optim.Adam(model.lenia_net.parameters(), lr=1e-3)
    opt_target = optim.Adam(model.target_net.parameters(), lr=2e-3)
    
    pool = HybridPool(512, DEVICE)

    print(f"Entraînement lancé sur {DEVICE}...")

    for i in range(ITERATIONS + 1):
        # --- PHASE 1 : Entraînement Lenia (Alpha = 0) ---
        x_l, _ = pool.sample(BATCH_SIZE)
        with amp_context():
            for _ in range(32):
                x_l = model(x_l, alpha=0.0)
            # Loss Lenia : on veut une densité de 15% environ
            loss_lenia = torch.abs(x_l[:, 3].mean() - 0.15).mean()

        opt_lenia.zero_grad(set_to_none=True)
        scaler.scale(loss_lenia).backward()
        scaler.unscale_(opt_lenia)
        clip_grad_norm_(model.lenia_net.parameters(), max_norm=1.0)
        scaler.step(opt_lenia)
        scaler.update()

        # --- PHASE 2 : Entraînement Target (Alpha = 1) ---
        x_t, idx = pool.sample(BATCH_SIZE)
        n_steps = torch.randint(64, 96, (1,)).item()
        with amp_context():
            for _ in range(n_steps):
                x_t = model(x_t, alpha=1.0)
            loss_target = torch.nn.functional.mse_loss(x_t[:, :4], target_batch)

        opt_target.zero_grad(set_to_none=True)
        scaler.scale(loss_target).backward()
        scaler.unscale_(opt_target)
        clip_grad_norm_(model.target_net.parameters(), max_norm=1.0)
        scaler.step(opt_target)
        scaler.update()

        # Update Pool
        pool.commit(x_t.detach(), idx)

        # --- MONITORING ---
        if i % 100 == 0:
            writer.add_scalar("Loss/Lenia", loss_lenia.item(), i)
            writer.add_scalar("Loss/Target", loss_target.item(), i)
            
            # Aperçu visuel
            with torch.no_grad():
                vis_img = x_t[0, :3].clamp(0, 1)
                writer.add_image("Preview/Target_Mode", vis_img, i)
            
            print(f"Ite {i} | Lenia: {loss_lenia.item():.4f} | Target: {loss_target.item():.6f}")

        if i % SAVE_EVERY == 0:
            torch.save(model.state_dict(), "checkpoints/hybrid_nca_latest.pth")

    writer.close()
    print("Entraînement terminé !")

if __name__ == "__main__":
    train()