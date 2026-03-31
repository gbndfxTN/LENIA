import torch
import cv2
import numpy as np
import os
from model import EvolutiveNCA # Importe l'architecture depuis model.py

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "mon_modele.pth" # Remplace par le nom exact de ton fichier .pth
GRID_SIZE = 96                # Taille de l'univers (96x96)
WINDOW_SIZE = 512             # Taille de la fenêtre d'affichage (512x512 pixels)

# --- 1. Initialisation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Simulation lancée sur : {device}")

# Instanciation du modèle
model = EvolutiveNCA(channel_n=16, hidden_n=256).to(device)

# Chargement des poids sauvegardés
if os.path.exists(MODEL_PATH):
    # map_location=device garantit que le modèle se charge correctement (CPU ou GPU)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Succès : Les poids du fichier '{MODEL_PATH}' ont été chargés !")
else:
    print(f"Attention : Fichier '{MODEL_PATH}' introuvable. Le modèle utilisera des poids aléatoires.")

model.eval() # Mode inférence (désactive les comportements d'entraînement)

# --- 2. Création de l'Univers ---
def init_grid(size, device):
    """Crée une grille vide et plante une graine au centre."""
    x = torch.zeros(1, 16, size, size, device=device)
    center = size // 2
    # On active les canaux à partir de l'index 3 (dont le canal "vivant") au centre
    x[:, 3:, center-5:center+5, center-5:center+5] = 1.0 
    return x

x = init_grid(GRID_SIZE, device)

# Variables d'état
current_alpha = 0.0 # 0.0 = Lenia (Libre), 1.0 = Target (Cible)
running = True

print("\n=== COMMANDES ===")
print("[T] : Basculer entre Mode Libre et Mode Cible")
print("[R] : Réinitialiser la grille (planter une nouvelle graine)")
print("[Q] ou [Echap] : Quitter la simulation")
print("=================\n")

# --- 3. Boucle de Simulation ---
while running:
    with torch.no_grad():
        # L'automate calcule l'étape suivante
        x = model(x, alpha=current_alpha)
        
    # --- 4. Rendu Visuel avec OpenCV ---
    # Extraction du canal 3 (masque de vie/densité), limitation entre 0 et 1
    img = x[0, 3].cpu().clamp(0, 1).numpy()
    
    # Conversion en format image standard (0-255)
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Agrandissement de l'image (INTER_NEAREST préserve l'aspect pixel net)
    img_display = cv2.resize(img_uint8, (WINDOW_SIZE, WINDOW_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # Passage en couleur BGR pour l'interface textuelle
    img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
    
    # --- 5. Interface (HUD) ---
    if current_alpha == 0.0:
        text = "Mode: LIBRE (Lenia) - Alpha: 0.0"
        color = (255, 150, 0) # Bleu/Cyan (Format BGR dans OpenCV)
    else:
        text = "Mode: CIBLE (Target) - Alpha: 1.0"
        color = (0, 0, 255) # Rouge (Format BGR)
        
    # Affichage des textes sur l'image
    cv2.putText(img_display, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img_display, "T: Basculer | R: Reset | Q: Quitter", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Affichage de la fenêtre
    cv2.imshow("NCA - Horloge Vivante", img_display)
    
    # --- 6. Gestion du Clavier ---
    # waitKey(1000) = 1 step par seconde (1 fps)
    key = cv2.waitKey(1000) & 0xFF
    
    if key == ord('q') or key == 27: # 27 = Touche Echap
        running = False
    elif key == ord('t'):
        # Bascule entre 0.0 et 1.0
        current_alpha = 1.0 if current_alpha == 0.0 else 0.0
        print(f"Bascule -> Nouvel alpha : {current_alpha}")
    elif key == ord('r'):
        # Réinitialisation
        x = init_grid(GRID_SIZE, device)
        print("Grille réinitialisée avec une nouvelle graine.")

# Nettoyage à la fermeture
cv2.destroyAllWindows()