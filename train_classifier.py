import os
import csv
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import torchvision
import torchvision.transforms as transforms

from .utils import setup_logging
from .classifier import Classifier
from .ddpm import Diffusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%I:%M:%S %p')


def get_data_with_split(args, val_split=0.2):
    """
    Load dataset and split into train/validation sets.
    
    Args:
        args: Arguments object with dataset_path, image_size, batch_size
        val_split: Fraction of data to use for validation (default 0.2)
    
    Returns:
        train_loader, val_loader, num_classes
    """
    # Training transforms (with augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize(80),  # args.image_size + 1/4 * args.image_size
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),  # Additional augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load full dataset without transforms first to get class info
    full_dataset_no_transform = torchvision.datasets.ImageFolder(args.dataset_path, transform=None)
    num_classes = len(full_dataset_no_transform.classes)
    
    # Get indices for split
    total_size = len(full_dataset_no_transform)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    # Create indices for train/val split
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets with appropriate transforms
    train_dataset = torchvision.datasets.ImageFolder(
        args.dataset_path,
        transform=train_transforms
    )
    val_dataset = torchvision.datasets.ImageFolder(
        args.dataset_path,
        transform=val_transforms
    )
    
    # Create subset datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logging.info(f"Dataset split: {train_size} training, {val_size} validation samples")
    logging.info(f"Number of classes: {num_classes}")
    
    return train_loader, val_loader, num_classes


def validate(model, val_loader, criterion, device, diffusion):
    """
    Validate the model on validation set.
    
    Returns:
        val_loss, val_accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            timesteps = diffusion.sample_timesteps(images.shape[0]).to(device)
            noisy_images, _ = diffusion.noise_images(images, timesteps)

            # Forward pass
            logits = model(noisy_images, timesteps)
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
    
    model.train()
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate_fixed_t(model, val_loader, criterion, device, diffusion, t_fixed: int):
    """
    Validate at a fixed timestep t_fixed. For t=0, use clean images.
    Returns (avg_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validating t={t_fixed}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            if t_fixed == 0:
                t = torch.zeros(images.size(0), dtype=torch.long, device=device)
                noisy_images = images
            else:
                t = torch.full((images.size(0),), int(t_fixed), dtype=torch.long, device=device)
                noisy_images, _ = diffusion.noise_images(images, t)
            logits = model(noisy_images, t)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return total_loss / len(val_loader), 100.0 * correct / total


def train(args):
    """
    Main training function for the classifier.
    """
    setup_logging(args.run_name)
    device = args.device if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Load data with train/val split
    train_loader, val_loader, num_classes = get_data_with_split(args, val_split=args.val_split)
    
    # Initialize model
    model = Classifier(
        num_classes=num_classes,
        use_timestep=True,
        device=device
    ).to(device)

    # Diffusion helper for generating noisy inputs
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard logger
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    
    # Training state
    best_val_acc = 0.0
    start_epoch = 0

    # Prepare metrics file for later plotting
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    metrics_csv = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
    # Long-form per-timestep validation metrics
    metrics_per_t_csv = os.path.join(results_dir, "metrics_per_t.csv")
    if not os.path.exists(metrics_per_t_csv):
        with open(metrics_per_t_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "timestep", "val_loss", "val_acc"])
    
    # Load checkpoint if specified
    if hasattr(args, 'model_path') and args.model_path:
        if os.path.exists(args.model_path):
            logging.info(f"Loading checkpoint from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            logging.info(f"Resuming training from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
        else:
            logging.warning(f"Checkpoint path {args.model_path} does not exist, starting from scratch")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Curriculum on timesteps: expand maximum t over training
            max_t = diffusion.noise_steps - 1
            curr_max_t = max(1, int((epoch + 1) / args.epochs * max_t))
            timesteps = torch.randint(low=0, high=curr_max_t, size=(images.shape[0],), device=device)
            mix_mask = (torch.rand(images.size(0), device=device) < 0.2)
            timesteps[mix_mask] = 0
            noisy_images, _ = diffusion.noise_images(images, timesteps)
            # after sampling timesteps
        
            # Forward pass
            optimizer.zero_grad()
            logits = model(noisy_images, timesteps)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            
            # Update progress bar
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            logger.add_scalar("Train/Loss", loss.item(), global_step)
            logger.add_scalar("Train/Accuracy", current_acc, global_step)
            logger.add_scalar("Train/Avg_timestep", timesteps.float().mean().item(), global_step)
        
        # Epoch summary
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation (fixed timesteps). Use t=0 metric for model selection.
        logging.info("Running validation...")
        fixed_ts = [0, 250, 500, 750, diffusion.noise_steps - 1]
        per_t_results = {}
        for t_fixed in fixed_ts:
            v_loss, v_acc = validate_fixed_t(model, val_loader, criterion, device, diffusion, t_fixed)
            per_t_results[t_fixed] = (v_loss, v_acc)
            # log to tensorboard
            logger.add_scalar(f"Val_t{t_fixed}/Loss", v_loss, epoch)
            logger.add_scalar(f"Val_t{t_fixed}/Accuracy", v_acc, epoch)
        val_loss, val_acc = per_t_results[0]
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch metrics
        logger.add_scalar("Epoch/Train_Loss", avg_train_loss, epoch)
        logger.add_scalar("Epoch/Train_Accuracy", train_acc, epoch)
        logger.add_scalar("Epoch/Val_Loss", val_loss, epoch)
        logger.add_scalar("Epoch/Val_Accuracy", val_acc, epoch)
        logger.add_scalar("Epoch/Learning_Rate", current_lr, epoch)
        
        logging.info(
            f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}"
        )

        # Append epoch metrics to CSV
        with open(metrics_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{train_acc:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}", f"{current_lr:.8f}"])
        # Also persist per-t results in a long-form CSV
        with open(os.path.join(results_dir, "metrics_per_t.csv"), mode="a", newline="") as f:
            writer = csv.writer(f)
            for t_fixed, (v_loss, v_acc) in per_t_results.items():
                writer.writerow([epoch + 1, t_fixed, f"{v_loss:.6f}", f"{v_acc:.4f}"])
        
        # Save checkpoint
        checkpoint_path = os.path.join("models", args.run_name, f"{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'num_classes': num_classes,
        }, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join("models", args.run_name, "best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'num_classes': num_classes,
            }, best_model_path)
            logging.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    logging.info("Training completed!")
    logging.info(f"Best validation accuracy: {best_val_acc:.2f}%")


def launch():
    """Launch training with default arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Train Butterfly Classifier')
    parser.add_argument('--model_path', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Training configuration
    args.run_name = "CLASSIFIER"
    args.image_size = 64
    args.dataset_path = r"data\train"
    args.weight_decay = 2e-4
    args.val_split = 0.1  # 10% for validation
    
    train(args)


if __name__ == "__main__":
    launch()

