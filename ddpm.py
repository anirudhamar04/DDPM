import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from .utils import setup_logging,save_images,get_data
from .modules import UNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%I:%M:%S %p')

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps=noise_steps
        self.beta_start=beta_start
        self.beta_end=beta_end
        self.img_size=img_size
        self.device=device

        self.beta=self.prepare_noise_schedule().to(device)
        self.alpha=1.0-self.beta
        self.alpha_hat=torch.cumprod(self.alpha,dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start,self.beta_end,self.noise_steps)
    
    def noise_images(self,x,t):
        sqrt_alpha_hat=torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_minus_alpha_hat=torch.sqrt(1-self.alpha_hat[t])[:,None,None,None]
        E=torch.randn_like(x)

        return sqrt_alpha_hat*x+sqrt_one_minus_alpha_hat*E,E

    def sample_timesteps(self,n):
        return torch.randint(low=1,high=self.noise_steps,size=(n,))

    def sample(self, model,n):
        logging.info(f"Sampling {n} new images.......")
        model.eval()
        with torch.no_grad():
            x=torch.randn((n,3,self.img_size,self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1,self.noise_steps)),position=0):
                t=(torch.ones(n)*i).long().to(self.device)
                predicted_noise=model(x,t)
                alpha=self.alpha[t][:,None,None,None]
                alpha_hat=self.alpha_hat[t][:,None,None,None]
                beta=self.beta[t][:,None,None,None]
                if i>1:
                    noise = torch.randn_like(x)
                else:
                    noise=torch.zeros_like(x)
                
                x=1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
        model.train()
        x=(x.clamp(-1,1)+1)/2
        x=(x*255).type(torch.uint8)

        return x
    
    def sample_guided(self, model, classifier, target_class, n, guidance_scale=1.0):
        """
        Classifier-guided sampling: generate images conditioned on a target class.
        Memory-optimized version for large batch sizes.
        
        Args:
            model: DDPM UNet model
            classifier: Trained classifier model (should support timestep input)
            target_class: Target class index (int or tensor of shape (n,))
            n: Number of images to generate
            guidance_scale: Strength of classifier guidance (higher = stronger class conditioning)
        
        Returns:
            Generated images as uint8 tensor (n, 3, img_size, img_size)
        """
        logging.info(f"Sampling {n} new images with classifier guidance (class={target_class}, scale={guidance_scale})...")
        model.eval()
        classifier.eval()
        
        # Clear any existing gradients
        model.zero_grad(set_to_none=True)
        classifier.zero_grad(set_to_none=True)
        
        # Convert target_class to tensor if needed
        if isinstance(target_class, int):
            target_class = torch.full((n,), target_class, dtype=torch.long, device=self.device)
        else:
            target_class = target_class.to(self.device)
        
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            t = (torch.ones(n) * i).long().to(self.device)
            
            # Get predicted noise from DDPM model (no gradients needed)
            with torch.no_grad():
                predicted_noise = model(x, t)
            
            # Compute classifier guidance
            if guidance_scale > 0:
                # Temporarily enable gradients only for this computation
                x_clone = x.detach().requires_grad_(True)
                
                # Get classifier logits
                logits = classifier(x_clone, t)
                
                # Compute log probability for target class
                # Use log_softmax for numerical stability
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                target_log_prob = log_probs.gather(1, target_class.unsqueeze(1)).squeeze(1)
                
                # Compute gradient: grad_x log p(y|x_t, t)
                grad = torch.autograd.grad(
                    outputs=target_log_prob.sum(),
                    inputs=x_clone,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True
                )[0]
                
                # Modify predicted noise with classifier gradient
                # Formula: predicted_noise_guided = predicted_noise - guidance_scale * sigma_t * grad_x
                # where sigma_t = sqrt(beta_t)
                sigma_t = torch.sqrt(self.beta[t])[:, None, None, None]
                predicted_noise = predicted_noise - guidance_scale * sigma_t * grad
                
                # Explicitly delete intermediate tensors to free memory
                del x_clone, logits, log_probs, target_log_prob, grad, sigma_t
                
                # Clear cache periodically
                if i % 200 == 0:
                    torch.cuda.empty_cache()
            
            # Denoising step (same as regular sampling) - no gradients needed
            with torch.no_grad():
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
                # Clean up intermediate tensors
                del predicted_noise, alpha, alpha_hat, beta, noise, t
        
        # Final cleanup
        torch.cuda.empty_cache()
        model.train()
        classifier.train()
        
        # Convert to [0, 255] uint8 format
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        return x
    
    def sample_ddim(self, model, n, num_steps=50, eta=0.0, x_T=None):
        """
        DDIM (Denoising Diffusion Implicit Models) sampling.
        Deterministic sampling that can use fewer steps than DDPM.
        
        Args:
            model: DDPM UNet model (same network trained for DDPM)
            n: Number of images to generate (ignored if x_T is provided)
            num_steps: Number of sampling steps (default: 50, can be much less than noise_steps)
            eta: Controls stochasticity. eta=0.0 is fully deterministic DDIM, 
                 eta=1.0 recovers DDPM sampling. Default: 0.0 (deterministic)
            x_T: Optional starting noise tensor (n, 3, H, W) in range [-1, 1].
                 If provided, starts sampling from this noise instead of random noise.
                 Useful for inversion + sampling round-trips.
        
        Returns:
            Generated images as uint8 tensor (n, 3, img_size, img_size)
        """
        if x_T is not None:
            n = x_T.shape[0]
            x = x_T.clone().to(self.device)
            logging.info(f"DDIM Sampling {n} image(s) from provided noise with {num_steps} steps (eta={eta})...")
        else:
            logging.info(f"DDIM Sampling {n} new images with {num_steps} steps (eta={eta})...")
        
        model.eval()
        
        # Create a sequence of timesteps to sample from
        # Use evenly spaced timesteps from the full schedule
        # We want to go from T (noise_steps-1) down to 0
        step_size = max(1, self.noise_steps // num_steps)
        timesteps = list(range(0, self.noise_steps, step_size))
        # Ensure we include the final timestep (T) and 0
        if timesteps[-1] != self.noise_steps - 1:
            timesteps.append(self.noise_steps - 1)
        if timesteps[0] != 0:
            timesteps.insert(0, 0)
        # Remove duplicates and sort
        timesteps = sorted(list(set(timesteps)))
        
        with torch.no_grad():
            # Start from provided noise or pure noise
            if x_T is None:
                x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            # Reverse through the selected timesteps (from T down to 0)
            for i in tqdm(range(len(timesteps) - 1), position=0, desc="DDIM Sampling"):
                t_curr = timesteps[-(i+1)]  # Current timestep (larger, e.g., T, T-step_size, ...)
                t_next = timesteps[-(i+2)]  # Next timestep (smaller, or 0)
                
                # Create timestep tensor
                t = (torch.ones(n) * t_curr).long().to(self.device)
                
                # Predict noise
                predicted_noise = model(x, t)
                
                # Get alpha_hat values
                alpha_hat_curr = self.alpha_hat[t_curr]
                # For t_next, use alpha_hat[0] if we're at the final step, otherwise use alpha_hat[t_next]
                if t_next < 0:
                    alpha_hat_next = self.alpha_hat[0]  # Use first timestep's alpha_hat
                else:
                    alpha_hat_next = self.alpha_hat[t_next]
                
                # Predict x_0 (denoised image)
                # x_0 = (x_t - sqrt(1 - alpha_hat_t) * epsilon) / sqrt(alpha_hat_t)
                pred_x0 = (x - torch.sqrt(1.0 - alpha_hat_curr)[:, None, None, None] * predicted_noise) / \
                          torch.sqrt(alpha_hat_curr)[:, None, None, None]
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1.0 - alpha_hat_next)[:, None, None, None] * predicted_noise
                
                # Add stochastic noise if eta > 0
                # Variance term for DDIM: sigma_t^2 = eta^2 * (1 - alpha_hat_{t-1}) / (1 - alpha_hat_t) * (1 - alpha_t)
                if eta > 0 and t_next >= 0:
                    alpha_curr = self.alpha[t_curr]
                    alpha_hat_prev = alpha_hat_next  # This is alpha_hat[t_next]
                    # Avoid division by zero
                    denominator = (1.0 - alpha_hat_curr).clamp(min=1e-8)
                    sigma_t_sq = eta ** 2 * (1.0 - alpha_hat_prev) / denominator * (1.0 - alpha_curr)
                    sigma_t_sq = sigma_t_sq.clamp(min=0.0)  # Ensure non-negative
                    noise = torch.randn_like(x) * torch.sqrt(sigma_t_sq)[:, None, None, None]
                else:
                    noise = torch.zeros_like(x)
                
                # DDIM update: x_{t-1} = sqrt(alpha_hat_{t-1}) * pred_x0 + sqrt(1 - alpha_hat_{t-1}) * predicted_noise + noise
                x = torch.sqrt(alpha_hat_next)[:, None, None, None] * pred_x0 + dir_xt + noise
        
        model.train()
        
        # Convert to [0, 255] uint8 format
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        return x
    
    def invert_ddim(self, model, x0, num_steps=50, eta=0.0, return_intermediates=False):
        """
        DDIM inversion: convert an image to noise using the reverse DDIM process.
        This is useful for image editing tasks where you invert, modify, and then denoise.
        
        Args:
            model: DDPM UNet model (same network trained for DDPM)
            x0: Input image(s) to invert. Can be:
                - Tensor of shape (n, 3, H, W) in range [-1, 1] (normalized)
                - Tensor of shape (n, 3, H, W) in range [0, 255] (uint8)
                - If uint8, will be automatically converted to [-1, 1]
            num_steps: Number of inversion steps (default: 50, should match sampling steps)
            eta: Controls stochasticity. eta=0.0 is fully deterministic DDIM, 
                 eta=1.0 recovers DDPM forward process. Default: 0.0 (deterministic)
            return_intermediates: If True, returns list of all intermediate x_t values
        
        Returns:
            If return_intermediates=False:
                x_T: Inverted noise tensor (n, 3, H, W) in range [-1, 1]
            If return_intermediates=True:
                (x_T, intermediates): Tuple of final noise and list of all x_t values
        """
        model.eval()
        
        # Convert input to [-1, 1] range if needed
        if x0.dtype == torch.uint8:
            x0 = x0.float() / 255.0  # [0, 1]
            x0 = x0 * 2.0 - 1.0  # [-1, 1]
        elif x0.max() > 1.0:
            # Assume [0, 255] range
            x0 = x0 / 255.0
            x0 = x0 * 2.0 - 1.0
        
        x0 = x0.to(self.device)
        n = x0.shape[0]
        
        logging.info(f"DDIM Inverting {n} image(s) with {num_steps} steps (eta={eta})...")
        
        # Create a sequence of timesteps to invert through
        # We go forward from 0 to T (opposite of sampling)
        step_size = max(1, self.noise_steps // num_steps)
        timesteps = list(range(0, self.noise_steps, step_size))
        # Ensure we include the final timestep (T) and 0
        if timesteps[-1] != self.noise_steps - 1:
            timesteps.append(self.noise_steps - 1)
        if timesteps[0] != 0:
            timesteps.insert(0, 0)
        # Remove duplicates and sort
        timesteps = sorted(list(set(timesteps)))
        
        intermediates = [x0] if return_intermediates else None
        
        with torch.no_grad():
            x = x0.clone()
            
            # Forward through the selected timesteps (from 0 to T)
            for i in tqdm(range(len(timesteps) - 1), position=0, desc="DDIM Inversion"):
                t_curr = timesteps[i]  # Current timestep (smaller, e.g., 0, step_size, ...)
                t_next = timesteps[i + 1]  # Next timestep (larger, or T)
                
                # Create timestep tensor
                t = (torch.ones(n) * t_curr).long().to(self.device)
                
                # Predict noise at current timestep
                predicted_noise = model(x, t)
                
                # Get alpha_hat values
                alpha_hat_curr = self.alpha_hat[t_curr]
                alpha_hat_next = self.alpha_hat[t_next]
                
                # Predict x_0 from current x_t
                # x_0 = (x_t - sqrt(1 - alpha_hat_t) * epsilon) / sqrt(alpha_hat_t)
                pred_x0 = (x - torch.sqrt(1.0 - alpha_hat_curr)[:, None, None, None] * predicted_noise) / \
                          torch.sqrt(alpha_hat_curr)[:, None, None, None]
                
                # Direction pointing to x_{t+1}
                dir_xt_next = torch.sqrt(1.0 - alpha_hat_next)[:, None, None, None] * predicted_noise
                
                # Add stochastic noise if eta > 0
                # Variance term for DDIM inversion: sigma_t^2 = eta^2 * (1 - alpha_hat_{t+1}) / (1 - alpha_hat_t) * (1 - alpha_t)
                if eta > 0 and t_next < self.noise_steps - 1:
                    alpha_curr = self.alpha[t_curr]
                    # Avoid division by zero
                    denominator = (1.0 - alpha_hat_curr).clamp(min=1e-8)
                    sigma_t_sq = eta ** 2 * (1.0 - alpha_hat_next) / denominator * (1.0 - alpha_curr)
                    sigma_t_sq = sigma_t_sq.clamp(min=0.0)  # Ensure non-negative
                    noise = torch.randn_like(x) * torch.sqrt(sigma_t_sq)[:, None, None, None]
                else:
                    noise = torch.zeros_like(x)
                
                # DDIM inversion update: x_{t+1} = sqrt(alpha_hat_{t+1}) * pred_x0 + sqrt(1 - alpha_hat_{t+1}) * predicted_noise + noise
                x = torch.sqrt(alpha_hat_next)[:, None, None, None] * pred_x0 + dir_xt_next + noise
                
                if return_intermediates:
                    intermediates.append(x.clone())
        
        model.train()
        
        if return_intermediates:
            return x, intermediates
        else:
            return x
        

def train(args):
    setup_logging(args.run_name)
    device=args.device if torch.cuda.is_available() else "cpu"
    dataloader=get_data(args)
    model=UNet(device=device).to(device)
    optimizer=optim.AdamW(model.parameters(),lr=args.lr)
    mse=nn.MSELoss()
    diffusion=Diffusion(img_size=args.image_size,device=device)
    logger = SummaryWriter(os.path.join("runs",args.run_name))
    l=len(dataloader)

    start_epoch = 0
    if hasattr(args, 'model_path') and args.model_path:
        if os.path.exists(args.model_path):
            logging.info(f"Loading checkpoint from {args.model_path}")
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            # Extract epoch number from filename (e.g., "95.pth" -> 95)
            filename = os.path.basename(args.model_path)
            try:
                start_epoch = int(filename.split('.')[0]) + 1
                logging.info(f"Resuming training from epoch {start_epoch}")
            except ValueError:
                logging.warning(f"Could not extract epoch number from {filename}, starting from epoch 0")
        else:
            logging.warning(f"Checkpoint path {args.model_path} does not exist, starting from scratch")

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar=tqdm(dataloader)
        for i, (images,_) in enumerate(pbar):
            images=images.to(device)
            t=diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t,noise=diffusion.noise_images(images,t)
            predicted_noise=model(x_t,t)
            loss=mse(noise,predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE",loss.item(),global_step=epoch*l+i)
        

        sampled_images=diffusion.sample(model,n=images.shape[0])
        save_images(sampled_images,os.path.join("results",args.run_name,f"{epoch}.png"))
        torch.save(model.state_dict(),os.path.join("models",args.run_name,f"{epoch}.pth"))



def launch():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to checkpoint to resume training from')
    args= parser.parse_args()
    args.run_name="DDPM_UNCONDITIONAL"
    args.epochs=184 #Steps per epoch = ceil(6500 / 8)= 812.5 → 813 steps per epoch 100,000 steps 100,000 / 813 ≈ 123 epochs
    args.batch_size=8
    args.image_size=64
    args.dataset_path=r"data\train"
    args.device="cuda"
    args.lr=3e-4
    train(args)

if __name__=="__main__":
    launch()