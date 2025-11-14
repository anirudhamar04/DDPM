import os
import argparse
import torch
import logging
from .ddpm import Diffusion
from .modules import UNet
from .classifier import Classifier
from .utils import save_images, setup_logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%I:%M:%S %p')


def load_ddpm_model(checkpoint_path, device):
    """
    Load DDPM UNet model from checkpoint.
    
    Args:
        checkpoint_path: Path to DDPM checkpoint (.pth file)
        device: Device to load model on
    
    Returns:
        Loaded UNet model
    """
    logging.info(f"Loading DDPM model from {checkpoint_path}")
    model = UNet(device=device).to(device)
    
    # DDPM checkpoints are saved as state_dict only
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    logging.info("DDPM model loaded successfully")
    return model


def load_classifier(checkpoint_path, device):
    """
    Load classifier model from checkpoint.
    
    Args:
        checkpoint_path: Path to classifier checkpoint (.pth file)
        device: Device to load model on
    
    Returns:
        Loaded Classifier model
    """
    logging.info(f"Loading classifier from {checkpoint_path}")
    
    # Load checkpoint to get num_classes
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get num_classes from checkpoint or default to 75
    num_classes = checkpoint.get('num_classes', 75)
    
    # Initialize classifier with timestep support (required for guided diffusion)
    model = Classifier(
        num_classes=num_classes,
        use_timestep=True,
        device=device
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"Classifier loaded successfully (num_classes={num_classes})")
    return model, num_classes


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Get all .pth files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoints:
        return None
    
    # Try to find 'best.pth' first
    if 'best.pth' in checkpoints:
        return os.path.join(checkpoint_dir, 'best.pth')
    
    # Otherwise, find the checkpoint with highest epoch number
    try:
        # Extract epoch numbers and find max
        epoch_numbers = []
        for ckpt in checkpoints:
            try:
                epoch = int(ckpt.split('.')[0])
                epoch_numbers.append((epoch, ckpt))
            except ValueError:
                continue
        
        if epoch_numbers:
            latest_epoch, latest_ckpt = max(epoch_numbers, key=lambda x: x[0])
            return os.path.join(checkpoint_dir, latest_ckpt)
    except Exception as e:
        logging.warning(f"Error finding latest checkpoint: {e}")
    
    # Fallback: return first checkpoint
    return os.path.join(checkpoint_dir, checkpoints[0])


def main():
    parser = argparse.ArgumentParser(description='Generate images using classifier-guided diffusion')
    parser.add_argument('--ddpm_path', type=str, default=None,
                        help='Path to DDPM checkpoint. If not provided, uses latest from models/DDPM_UNCONDITIONAL/')
    parser.add_argument('--classifier_path', type=str, default=None,
                        help='Path to classifier checkpoint. If not provided, uses models/CLASSIFIER/best.pth')
    parser.add_argument('--target_class', type=int, required=True,
                        help='Target class index (0 to num_classes-1)')
    parser.add_argument('--n_samples', type=int, default=8,
                        help='Number of images to generate (default: 8)')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Guidance scale for classifier guidance (default: 1.0, higher = stronger conditioning)')
    parser.add_argument('--output_dir', type=str, default='results/CLASSIFIER_GUIDED',
                        help='Directory to save generated images (default: results/CLASSIFIER_GUIDED)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size (default: 64)')
    
    args = parser.parse_args()
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load DDPM model
    if args.ddpm_path is None:
        ddpm_dir = "models/DDPM_UNCONDITIONAL"
        args.ddpm_path = find_latest_checkpoint(ddpm_dir)
        if args.ddpm_path is None:
            raise FileNotFoundError(f"No DDPM checkpoint found in {ddpm_dir}")
    
    ddpm_model = load_ddpm_model(args.ddpm_path, device)
    
    # Load classifier
    if args.classifier_path is None:
        args.classifier_path = "models/CLASSIFIER/best.pth"
        if not os.path.exists(args.classifier_path):
            raise FileNotFoundError(f"Classifier checkpoint not found at {args.classifier_path}")
    
    classifier_model, num_classes = load_classifier(args.classifier_path, device)
    
    # Validate target class
    if args.target_class < 0 or args.target_class >= num_classes:
        raise ValueError(f"target_class must be between 0 and {num_classes-1}, got {args.target_class}")
    
    # Initialize diffusion
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    # Generate images
    logging.info(f"Generating {args.n_samples} images for class {args.target_class} with guidance scale {args.guidance_scale}")
    generated_images = diffusion.sample_guided(
        model=ddpm_model,
        classifier=classifier_model,
        target_class=args.target_class,
        n=args.n_samples,
        guidance_scale=args.guidance_scale
    )
    
    # Save images
    output_filename = f"class_{args.target_class}_guidance_{args.guidance_scale:.2f}_n{args.n_samples}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    save_images(generated_images, output_path)
    
    logging.info(f"Generated images saved to {output_path}")


if __name__ == "__main__":
    main()


