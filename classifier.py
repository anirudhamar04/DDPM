import torch
import torch.nn as nn
import torchvision.models as models


class Classifier(nn.Module):
    """
    Classifier for Butterfly dataset with 75 classes.
    Based on ResNet18 architecture, adapted for 64x64 images.
    Optionally supports timestep embeddings for noise-aware classification.
    """
    def __init__(self, num_classes=75, use_timestep=False, time_dim=256, device="cuda", dropout_p: float = 0.3, noise_steps: int = 1000):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.use_timestep = use_timestep
        self.device = device
        self.time_dim = time_dim
        self.noise_steps = noise_steps
        
        # Load pretrained ResNet18 and modify for our use case
        resnet = models.resnet18(weights=None)  # No pretrained weights for 64x64
        
        # Modify first conv layer for 64x64 images (original expects 224x224)
        # Keep original kernel size but adjust if needed
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        # Keep more spatial information for 64x64 inputs
        self.maxpool = nn.Identity()
        
        # ResNet layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Adaptive pooling for variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Timestep embedding layers (if needed for classifier-guided diffusion)
        if use_timestep:
            # Learned timestep embedding tends to work well for classifiers
            self.time_lookup = nn.Embedding(self.noise_steps, 512)
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(512, 512),
            )
            # Feature dimension after ResNet layers (512 for ResNet18)
            #self.fc = nn.Linear(512 + 512, num_classes)  # Concatenate features + time embedding
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(512 + 512, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(512, num_classes),
            )
        
    def pos_encoding(self, t, channels):
        """Sinusoidal positional encoding for timesteps (same as UNet)"""
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t=None):
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, 3, 64, 64) in range [-1, 1]
            t: Optional timestep tensor of shape (B,) for noise-aware classification
        
        Returns:
            logits: Class logits of shape (B, num_classes)
        """
        # ResNet forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 512)
        
        # Add timestep embedding if needed
        if self.use_timestep and t is not None:
            # Clamp and embed discrete timesteps
            t = t.clamp_(min=0, max=self.noise_steps - 1).long()
            t_emb = self.time_lookup(t)  # (B, 512)
            t_emb = self.time_proj(t_emb)  # (B, 512)
            x = torch.cat([x, t_emb], dim=1)  # (B, 1024 when use_timestep)
        
        # Classification head
        logits = self.head(x)
        
        return logits

