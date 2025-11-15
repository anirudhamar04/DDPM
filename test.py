from modules import RefineNetNCSN
from ncsn import NCSN
import torch
import matplotlib.pyplot as plt
import torchvision

device="cuda"
model_path=r"models\180_ema.pth"
model=RefineNetNCSN(device=device).to(device)
ncsn=NCSN(img_size=64,device=device)
model.load_state_dict(torch.load(model_path, map_location=device))

ncsn_better = NCSN(
    sigma_min=0.01, sigma_max=50.0, num_noise_levels=50,
    img_size=64, device="cuda",
    langevin_steps=100,  # More steps
    langevin_step_size=5e-4
)
samples = ncsn_better.sample_annealed(model, n=3)

grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)

# Convert to numpy: (C, H, W) -> (H, W, C)
grid_np = grid.permute(1, 2, 0).cpu().numpy()

# Plot
plt.figure(figsize=(12, 12))
plt.imshow(grid_np)
plt.axis('off')
plt.title('NCSN Generated Samples (10 Langevin Steps)', fontsize=14)
plt.tight_layout()
plt.show()