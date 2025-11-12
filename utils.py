import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def load_real_images_for_fid(dataset_path, image_size=64, num_images=1000, batch_size=8):
    """
    Load real images from dataset for FID computation.
    
    Args:
        dataset_path: Path to the dataset
        image_size: Size of images
        num_images: Number of images to load
        batch_size: Batch size for loading
    
    Returns:
        Tensor of real images (N, C, H, W) in range [-1, 1]
    """
    from types import SimpleNamespace
    
    args = SimpleNamespace()
    args.dataset_path = dataset_path
    args.image_size = image_size
    args.batch_size = batch_size
    
    dataloader = get_data(args)
    real_images_list = []
    # Collect all images from the dataset
    for images, _ in dataloader:
        real_images_list.append(images)
    
    # Concatenate all batches
    all_real_images = torch.cat(real_images_list, dim=0)
    
    # Randomly sample num_images
    if len(all_real_images) > num_images:
        # Generate random indices
        indices = torch.randperm(len(all_real_images))[:num_images]
        real_images = all_real_images[indices]
    else:
        real_images = all_real_images
        print(f"Warning: Only {len(all_real_images)} images available, less than requested {num_images}")
    
    return real_images