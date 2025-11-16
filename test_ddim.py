import torch
import matplotlib.pyplot as plt
import numpy as np
from ddpm import Diffusion
from modules import UNet


def inference_ddim_3_images(model_path=None, device="cuda", img_size=64, num_steps=50, eta=0.0):
    """
    Generate 3 images using DDIM sampling and display them with pyplot.
    
    Args:
        model_path: Path to DDPM checkpoint. If None, uses latest from models/DDPM_UNCONDITIONAL/
        device: Device to use (default: "cuda")
        img_size: Image size (default: 64)
        num_steps: Number of DDIM steps (default: 50)
        eta: DDIM eta parameter, 0.0=deterministic (default: 0.0)
    """
    import os
    
    # Setup device
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Find model path if not provided
    if model_path is None:
        model_dir = "models/DDPM_UNCONDITIONAL"
        if os.path.exists(model_dir):
            checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if checkpoints:
                # Get latest checkpoint by epoch number
                def get_epoch(filename):
                    try:
                        return int(filename.split('.')[0])
                    except ValueError:
                        return -1
                checkpoints.sort(key=get_epoch, reverse=True)
                model_path = os.path.join(model_dir, checkpoints[0])
                print(f"Using latest checkpoint: {model_path}")
            else:
                raise FileNotFoundError(f"No checkpoints found in {model_dir}")
        else:
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = UNet(device=device).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
    
    # Create diffusion instance
    diffusion = Diffusion(
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=img_size,
        device=device
    )
    
    # Generate 3 images using DDIM
    print(f"Generating 3 images using DDIM with {num_steps} steps...")
    samples = diffusion.sample_ddim(
        model=model,
        n=3,
        num_steps=num_steps,
        eta=eta
    )
    
    # Convert to numpy for display
    # samples is uint8 tensor of shape (3, 3, 64, 64) in range [0, 255]
    samples_np = samples.permute(0, 2, 3, 1).cpu().numpy()  # (3, 64, 64, 3)
    
    # Display images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        axes[i].imshow(samples_np[i])
        axes[i].axis('off')
        axes[i].set_title(f'Generated Image {i+1}', fontsize=12)
    
    plt.suptitle(f'DDIM Generated Images ({num_steps} steps, eta={eta})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("Images displayed successfully!")
    return samples


if __name__ == "__main__":
    inference_ddim_3_images(model_path="models/DDPM_UNCONDITIONAL/183.pth", num_steps=200)