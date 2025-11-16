from modules import RefineNetNCSN
from ncsn import NCSN
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np

device="cuda"
model_path=r"models\320.pth"
model=RefineNetNCSN(device=device).to(device)
model.eval()
model.load_state_dict(torch.load(model_path, map_location=device))

ncsn_better = NCSN(
    sigma_min=0.01, sigma_max=50.0, num_noise_levels=10,
    img_size=64, device="cuda",
    langevin_steps=100,
    langevin_step_size=1e-6
)

# ✅ DEBUG: Check what sigma values the model sees
print("Sigma values:", ncsn_better.sigmas.cpu().numpy())

# ✅ DEBUG: Test score prediction on a simple case
print("\n=== Testing Score Prediction ===")
# Create a simple test: clean image + noise
test_image = torch.zeros(1, 3, 64, 64, device=device)  # Black image
test_sigma = torch.tensor([10.0], device=device)  # Medium noise level
test_noisy = test_image + torch.randn_like(test_image) * test_sigma

with torch.no_grad():
    score = model(test_noisy, test_sigma)
    print(f"Score magnitude: {torch.abs(score).mean().item():.4f}")
    print(f"Score mean: {score.mean().item():.4f}")
    print(f"Score std: {score.std().item():.4f}")
    
    # Expected: score should point toward clean image
    # For Gaussian: score = -(noisy - clean) / σ²
    # If clean=0, noisy=noise, then score = -noise / σ² (should be negative of noise direction)
    expected_score = -(test_noisy - test_image) / (test_sigma ** 2)
    print(f"Expected score magnitude: {torch.abs(expected_score).mean().item():.4f}")
    print(f"Score error: {torch.abs(score - expected_score).mean().item():.4f}")

# ✅ DEBUG: Check if sampling is making progress
print("\n=== Testing Sampling Progress ===")
x = torch.randn((1, 3, 64, 64), device=device) * ncsn_better.sigmas[0]
print(f"Initial x range: [{x.min().item():.2f}, {x.max().item():.2f}]")
print(f"Initial x std: {x.std().item():.2f}")

# Run a few steps
sigma = ncsn_better.sigmas[0]
sigma_L = ncsn_better.sigmas[-1]
alpha = ncsn_better.langevin_step_size * (sigma ** 2) / (sigma_L ** 2)
print(f"First sigma: {sigma.item():.2f}, alpha: {alpha.item():.6f}")

with torch.no_grad():
    for step in range(10):
        score = model(x, sigma.expand(1))
        noise_term = torch.randn_like(x) * torch.sqrt(alpha)
        x = x + (alpha / 2) * score + noise_term
        if step % 2 == 0:
            print(f"Step {step}: x range=[{x.min().item():.2f}, {x.max().item():.2f}], "
                  f"|score|={torch.abs(score).mean().item():.4f}")

print(f"After 10 steps: x range: [{x.min().item():.2f}, {x.max().item():.2f}]")
print(f"After 10 steps: x std: {x.std().item():.2f}")

# Now try full sampling
print("\n=== Full Sampling ===")
samples = ncsn_better.sample_annealed(model, n=3)
print(f"Final samples range: [{samples.min().item()}, {samples.max().item()}]")
print(f"Final samples mean: {samples.float().mean().item():.2f}")

grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
grid_np = grid.permute(1, 2, 0).cpu().numpy()
grid_np = np.clip(grid_np, 0, 255).astype(np.uint8)

plt.figure(figsize=(12, 12))
plt.imshow(grid_np)
plt.axis('off')
plt.title('NCSN Generated Samples', fontsize=14)
plt.tight_layout()
plt.show()