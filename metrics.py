import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from tqdm import tqdm


def get_inception_model(device='cuda'):
    """
    Load and prepare the Inception v3 model for feature extraction.
    
    Args:
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        Inception v3 model in evaluation mode
    """
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove the final classification layer
    model = model.to(device)
    model.eval()
    return model


def get_inception_features(images, model, device='cuda', batch_size=50):
    """
    Extract features from images using Inception v3 model.
    
    Args:
        images: Tensor of images (N, C, H, W) with values in range [-1, 1] or [0, 1]
        model: Inception v3 model
        device: Device to run inference on
        batch_size: Batch size for processing
    
    Returns:
        Numpy array of features (N, 2048)
    """
    model.eval()
    features = []
    
    # Normalize images to [0, 1] if they're in [-1, 1] range
    if images.min() < 0:
        images = (images + 1) / 2
    
    # Resize images to 299x299 for Inception v3
    if images.shape[2] != 299 or images.shape[3] != 299:
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i+batch_size].to(device)
            # Inception v3 expects 3-channel RGB images
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            elif batch.shape[1] != 3:
                raise ValueError(f"Expected 1 or 3 channels, got {batch.shape[1]}")
            
            feat = model(batch)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate the Fréchet distance between two multivariate Gaussians.
    
    Args:
        mu1: Mean of the first distribution
        sigma1: Covariance matrix of the first distribution
        mu2: Mean of the second distribution
        sigma2: Covariance matrix of the second distribution
        eps: Small value for numerical stability
    
    Returns:
        Fréchet distance (FID score)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(images, model, device='cuda', batch_size=50):
    """
    Calculate mean and covariance statistics for a set of images.
    
    Args:
        images: Tensor of images (N, C, H, W)
        model: Inception v3 model
        device: Device to run inference on
        batch_size: Batch size for processing
    
    Returns:
        Tuple of (mean, covariance) numpy arrays
    """
    features = get_inception_features(images, model, device, batch_size)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def FID(real_images, generated_images, device='cuda', batch_size=50):
    """
    Calculate the Fréchet Inception Distance (FID) between real and generated images.
    
    FID measures the distance between the feature distributions of real and generated images
    using the Inception v3 network. Lower FID scores indicate better quality and diversity
    of generated images.
    
    Args:
        real_images: Tensor of real images (N, C, H, W) with values in range [-1, 1] or [0, 1]
        generated_images: Tensor of generated images (M, C, H, W) with values in range [-1, 1] or [0, 1]
        device: Device to run inference on ('cuda' or 'cpu')
        batch_size: Batch size for processing images
    
    Returns:
        FID score (float). Lower is better.
    
    Example:
        >>> real_imgs = torch.randn(100, 3, 64, 64)
        >>> gen_imgs = torch.randn(100, 3, 64, 64)
        >>> fid_score = FID(real_imgs, gen_imgs, device='cuda')
        >>> print(f"FID Score: {fid_score:.2f}")
    """
    print("Loading Inception v3 model...")
    model = get_inception_model(device)
    
    print("Calculating statistics for real images...")
    mu1, sigma1 = calculate_activation_statistics(real_images, model, device, batch_size)
    
    print("Calculating statistics for generated images...")
    mu2, sigma2 = calculate_activation_statistics(generated_images, model, device, batch_size)
    
    print("Calculating FID score...")
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return float(fid_value)

