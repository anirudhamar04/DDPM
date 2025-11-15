import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from .utils import setup_logging, save_images, get_data
from .modules import RefineNetNCSN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%I:%M:%S %p')


class NCSN:
    """
    Noise Conditional Score Network (NCSN) for generative modeling.
    Based on "Generative Modeling by Estimating Gradients of the Data Distribution"
    by Song & Ermon, NeurIPS 2019.
    """
    
    def __init__(self, sigma_min=0.01, sigma_max=50.0, num_noise_levels=50, 
                 img_size=64, device="cuda", langevin_steps=10, langevin_step_size=1e-5):
        """
        Args:
            sigma_min: Minimum noise level (finest scale)
            sigma_max: Maximum noise level (coarsest scale)
            num_noise_levels: Number of discrete noise levels
            img_size: Image size
            device: Device to run on
            langevin_steps: Number of Langevin steps per noise level
            langevin_step_size: Step size for Langevin dynamics
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_noise_levels = num_noise_levels
        self.img_size = img_size
        self.device = device
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size
        
        # Create geometric progression of noise levels: σ_1 > σ_2 > ... > σ_L
        # σ_i = σ_max * (σ_min/σ_max)^(i/(L-1))
        # When i=0: σ_max, when i=L-1: σ_min
        self.sigmas = torch.tensor([
            sigma_max * (sigma_min / sigma_max) ** (i / (num_noise_levels - 1))
            for i in range(num_noise_levels)
        ], device=device)
        
        logging.info(f"NCSN initialized with {num_noise_levels} noise levels")
        logging.info(f"Sigma range: [{self.sigmas[-1].item():.4f}, {self.sigmas[0].item():.4f}]")
    
    def add_noise(self, x, sigma_idx):
        """
        Add noise to clean data.
        
        Args:
            x: Clean data (B, C, H, W) in range [-1, 1]
            sigma_idx: Index of noise level (0 to num_noise_levels-1) or tensor of indices
        
        Returns:
            Noisy data and noise
        """
        if isinstance(sigma_idx, int):
            sigma = self.sigmas[sigma_idx]
        else:
            sigma = self.sigmas[sigma_idx].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x) * sigma
        noisy_x = x + noise
        return noisy_x, noise
    
    def sample_noise_level(self, batch_size):
        """Sample a random noise level index for training."""
        return torch.randint(0, self.num_noise_levels, (batch_size,), device=self.device)
    
    def get_score_target(self, x_clean, x_noisy, sigma_idx):
        """
        Compute target score: ∇_x log p_σ(x) = -(x_noisy - x_clean) / σ²
        
        Args:
            x_clean: Clean data
            x_noisy: Noisy data
            sigma_idx: Noise level index or tensor of indices
        
        Returns:
            Target score
        """
        if isinstance(sigma_idx, int):
            sigma = self.sigmas[sigma_idx]
        else:
            sigma = self.sigmas[sigma_idx].view(-1, 1, 1, 1)
        
        # Score = -(noisy - clean) / σ²
        score_target = -(x_noisy - x_clean) / (sigma ** 2 + 1e-8)
        return score_target
    
    def sample(self, model, n):
        """
        Generate samples using Langevin dynamics.
        Following the NCSN paper Algorithm 1: α_i = ε · σ_i² / σ_L²
        
        Args:
            model: Trained score network
            n: Number of samples to generate
        
        Returns:
            Generated samples (n, C, H, W) in range [-1, 1]
        """
        logging.info(f"Sampling {n} new images with NCSN...")
        model.eval()
        
        # Start from highest noise level
        x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device) * self.sigmas[0]
        
        # σ_L is the smallest noise level
        sigma_L = self.sigmas[-1]
        
        with torch.no_grad():
            # Iterate through noise levels from high to low
            for i, sigma in enumerate(tqdm(self.sigmas, desc="NCSN Sampling")):
                # Paper formula: α_i = ε · σ_i² / σ_L²
                alpha = self.langevin_step_size * (sigma ** 2) / (sigma_L ** 2)
                
                # Run Langevin dynamics for this noise level
                for step in range(self.langevin_steps):
                    # Predict score
                    sigma_batch = sigma.expand(n)
                    score = model(x, sigma_batch)
                    
                    # Langevin step: x ← x + (α/2) * score + √α * z
                    noise_term = torch.randn_like(x) * torch.sqrt(alpha)
                    x = x + (alpha / 2) * score + noise_term
        
        # Clamp to valid range
        x = torch.clamp(x, -1, 1)
        
        # Convert to [0, 255] uint8 format
        x = (x + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        return x
    
    def sample_annealed(self, model, n):
        """
        Generate samples using annealed Langevin dynamics.
        Following the NCSN paper Algorithm 1 exactly:
        - α_i = ε · σ_i² / σ_L² where ε is step size and σ_L is smallest noise level
        
        Args:
            model: Trained score network
            n: Number of samples to generate
        
        Returns:
            Generated samples (n, C, H, W) in range [-1, 1]
        """
        logging.info(f"Sampling {n} new images with Annealed NCSN...")
        model.eval()
        
        # Start from highest noise level
        x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device) * self.sigmas[0]
        
        # σ_L is the smallest noise level (last in the list)
        sigma_L = self.sigmas[-1]
        
        with torch.no_grad():
            for i, sigma in enumerate(tqdm(self.sigmas, desc="Annealed Sampling")):
                # Paper formula: α_i = ε · σ_i² / σ_L²
                alpha = self.langevin_step_size * (sigma ** 2) / (sigma_L ** 2)
                
                # Run Langevin dynamics
                for step in range(self.langevin_steps):
                    sigma_batch = sigma.expand(n)
                    score = model(x, sigma_batch)
                    
                    # Langevin step: x ← x + (α/2) * score + √α * z
                    noise_term = torch.randn_like(x) * torch.sqrt(alpha)
                    x = x + (alpha / 2) * score + noise_term
        
        x = torch.clamp(x, -1, 1)
        
        # Convert to [0, 255] uint8 format
        x = (x + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        return x
    
    def sample_guided(self, model, classifier, target_class, n, guidance_scale=1.0):
        """
        Classifier-guided sampling for NCSN.
        
        Args:
            model: Trained score network
            classifier: Trained classifier model
            target_class: Target class index (int or tensor)
            n: Number of images to generate
            guidance_scale: Strength of classifier guidance
        
        Returns:
            Generated samples (n, C, H, W) in range [-1, 1]
        """
        logging.info(f"Sampling {n} new images with classifier-guided NCSN (class={target_class}, scale={guidance_scale})...")
        model.eval()
        classifier.eval()
        
        # Convert target_class to tensor if needed
        if isinstance(target_class, int):
            target_class = torch.full((n,), target_class, dtype=torch.long, device=self.device)
        else:
            target_class = target_class.to(self.device)
        
        # Start from highest noise level
        x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device) * self.sigmas[0]
        
        # σ_L is the smallest noise level
        sigma_L = self.sigmas[-1]
        
        with torch.no_grad():
            for i, sigma in enumerate(tqdm(self.sigmas, desc="Guided Sampling")):
                # Paper formula: α_i = ε · σ_i² / σ_L²
                alpha = self.langevin_step_size * (sigma ** 2) / (sigma_L ** 2)
                
                for step in range(self.langevin_steps):
                    sigma_batch = sigma.expand(n)
                    
                    # Get score from model
                    score = model(x, sigma_batch)
                    
                    # Compute classifier guidance
                    if guidance_scale > 0:
                        x.requires_grad_(True)
                        
                        # Get classifier logits (need to adapt classifier for sigma)
                        # For now, use timestep-like mapping: map sigma to timestep range
                        # This assumes classifier accepts timestep
                        t_mapped = ((sigma / self.sigmas[0]) * 999).long().clamp(0, 999)
                        logits = classifier(x, t_mapped)
                        
                        # Compute log probability for target class
                        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                        target_log_prob = log_probs.gather(1, target_class.unsqueeze(1)).squeeze(1)
                        
                        # Compute gradient
                        grad = torch.autograd.grad(
                            outputs=target_log_prob.sum(),
                            inputs=x,
                            create_graph=False,
                            retain_graph=False,
                            only_inputs=True
                        )[0]
                        
                        x = x.detach()
                        
                        # Modify score with classifier gradient
                        score = score - guidance_scale * sigma * grad
                        
                        del grad, logits, log_probs, target_log_prob
                    
                    # Langevin step
                    noise_term = torch.randn_like(x) * torch.sqrt(alpha)
                    x = x + (alpha / 2) * score + noise_term
                    
                    # Clean up
                    del score
                
                # Clear cache periodically
                if i % 2 == 0:
                    torch.cuda.empty_cache()
        
        x = torch.clamp(x, -1, 1)
        
        # Convert to [0, 255] uint8 format
        x = (x + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        return x


def train(args):
    """
    Training loop for NCSN.
    """
    setup_logging(args.run_name)
    device = args.device if torch.cuda.is_available() else "cpu"
    dataloader = get_data(args)
    model = RefineNetNCSN(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,betas=(0.9,0.999))
    mse = nn.MSELoss()
    ncsn = NCSN(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        num_noise_levels=args.num_noise_levels,
        img_size=args.image_size,
        device=device
    )
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    
    # Initialize EMA model
    ema_decay = getattr(args, 'ema_decay', 0.9999)  # Default EMA decay
    ema_model = RefineNetNCSN(device=device).to(device)
    ema_model.load_state_dict(model.state_dict())  # Initialize with same weights
    ema_model.eval()  # EMA model is always in eval mode
    
    def update_ema(ema_model, model, decay):
        """Update EMA model weights"""
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    start_epoch = 0
    if hasattr(args, 'model_path') and args.model_path:
        if os.path.exists(args.model_path):
            logging.info(f"Loading checkpoint from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint)
            
            # Try to load EMA weights if available
            ema_path = args.model_path.replace('.pth', '_ema.pth')
            if os.path.exists(ema_path):
                logging.info(f"Loading EMA weights from {ema_path}")
                ema_model.load_state_dict(torch.load(ema_path, map_location=device))
            else:
                # Initialize EMA from current model if no EMA checkpoint
                ema_model.load_state_dict(model.state_dict())
            
            filename = os.path.basename(args.model_path)
            try:
                start_epoch = int(filename.split('.')[0]) + 1
                logging.info(f"Resuming training from epoch {start_epoch}")
            except ValueError:
                logging.warning(f"Could not extract epoch number from {filename}, starting from epoch 0")
        else:
            logging.warning(f"Checkpoint path {args.model_path} does not exist, starting from scratch")
    
    logging.info(f"Using EMA with decay={ema_decay}")
    
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0.0
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)  # Images in range [-1, 1]
            batch_size = images.shape[0]
            
            # Sample noise levels for each image in batch
            sigma_indices = ncsn.sample_noise_level(batch_size)
            
            # Add noise
            x_noisy, noise = ncsn.add_noise(images, sigma_indices)
            
            # Get target score
            score_target = ncsn.get_score_target(images, x_noisy, sigma_indices)
            
            # Get sigma values for conditioning
            sigma_values = ncsn.sigmas[sigma_indices]  # (B,)
            
            # Predict score
            score_pred = model(x_noisy, sigma_values)
            
            # FIXED: Compute weighted loss correctly
            # NCSN loss: L = E[λ(σ) * ||s_θ(x, σ) - ∇_x log p_σ(x)||²]
            # where λ(σ) = σ²
            # Compute squared error per element
            squared_error = (score_pred - score_target) ** 2  # (B, C, H, W)
            
            # Get per-sample weights: σ² for each sample
            sigma_weights = ncsn.sigmas[sigma_indices] ** 2  # (B,)
            sigma_weights = sigma_weights.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
            
            # Weight the squared error, then average over all dimensions
            weighted_squared_error = squared_error * sigma_weights  # (B, C, H, W)
            loss = weighted_squared_error.mean()
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update EMA model after optimizer step
            update_ema(ema_model, model, ema_decay)
            
            epoch_loss += loss.item()
            pbar.set_postfix(Loss=loss.item())
            logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)
            
            # Log per-noise-level statistics periodically
            if i % 100 == 0:
                with torch.no_grad():
                    # Log average loss per noise level
                    for idx in range(ncsn.num_noise_levels):
                        mask = (sigma_indices == idx)
                        if mask.any():
                            level_loss = weighted_squared_error[mask].mean().item()
                            logger.add_scalar(f"Loss/NoiseLevel_{idx}_sigma_{ncsn.sigmas[idx]:.2f}", 
                                            level_loss, global_step=epoch * l + i)
        
        
        avg_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        logger.add_scalar("Epoch/Avg_Loss", avg_loss, epoch)
        
        if epoch % 10 == 0:
            # Sample and save images using EMA model (better quality)
            sampled_images = ncsn.sample_annealed(ema_model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.png"))
            
            # Save both regular and EMA models
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}.pth"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_ema.pth"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    args.run_name = "NCSN_UNCONDITIONAL"
    args.epochs = 100
    args.batch_size = 8
    args.image_size = 64
    args.dataset_path = r"data\train"
    args.device = "cuda"
    args.lr = 0.001  # Paper uses 0.001
    
    # NCSN-specific parameters (matching paper)
    args.sigma_min = 0.01
    args.sigma_max = 50.0
    args.num_noise_levels = 50  # Paper uses 50+ noise levels
    
    train(args)


if __name__ == "__main__":
    launch()