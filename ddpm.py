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