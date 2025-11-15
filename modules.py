import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual=residual
        if not mid_channels:
            mid_channels=out_channels
        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,out_channels),
        )

    def forward(self,x):
        if self.residual:
            return F.gelu(x+self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels,emb_dim=256):
        super().__init__()
        self.maxpool_conv=nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,in_channels,residual=True),
            DoubleConv(in_channels,out_channels),
        )

        self.emb_layer=nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
    
    def forward(self,x,t):
        x=self.maxpool_conv(x)
        emb=self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[2],x.shape[3])
        return x+emb
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels,emb_dim=256):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.conv= nn.Sequential(
            DoubleConv(in_channels,in_channels,residual=True),
            DoubleConv(in_channels,out_channels),
        )
        self.emb_layer=nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self,x,skip_x,t):
        x=self.up(x)
        x=torch.cat([skip_x,x],dim=1)
        x=self.conv(x)
        emb=self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[2],x.shape[3])
        return x+emb
    
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention,self).__init__()
        self.channels=channels
        self.size=size
        self.mha=nn.MultiheadAttention(channels,4,batch_first=True)
        self.ln=nn.LayerNorm([channels])
        self.ff_self=nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels,channels),
            nn.GELU(),
            nn.Linear(channels,channels),
        )

    def forward(self,x):
        x=x.view(-1,self.channels,self.size*self.size).swapaxes(1,2)
        x_ln=self.ln(x)
        attention_value, _ = self.mha(x_ln,x_ln,x_ln)
        attention_value=attention_value+x
        attention_value=self.ff_self(attention_value)+attention_value
        return attention_value.swapaxes(2,1).view(-1,self.channels,self.size,self.size)


class UNet(nn.Module):
    def __init__(self,c_in=3,c_out=3,time_dim=256,device="cuda"):
        super(UNet,self).__init__()
        self.device=device
        self.time_dim=time_dim
        self.inc=DoubleConv(c_in,64)
        self.down1=Down(64,128)
        self.sa1=SelfAttention(128,32)
        self.down2=Down(128,256)
        self.sa2=SelfAttention(256,16)
        self.down3=Down(256,256)
        self.sa3=SelfAttention(256,8)
        
        self.bot1=DoubleConv(256,512)
        self.bot2=DoubleConv(512,512)
        self.bot3=DoubleConv(512,512)

        self.up1=Up(768,128)  # 512 (from bot3) + 256 (skip from x3) = 768
        self.sa4=SelfAttention(128,16)
        self.up2=Up(256,64)
        self.sa5=SelfAttention(64,32)
        self.up3=Up(128,64)
        self.sa6=SelfAttention(64,64)
        self.outc=nn.Conv2d(64,c_out,kernel_size=1)
    
    def pos_encoding(self,t,channels):
        inv_freq=1.0/(
            10000**(torch.arange(0,channels,2,device=self.device).float()/channels)
        )
        pos_enc_a=torch.sin(t.repeat(1,channels//2)*inv_freq)
        pos_enc_b=torch.cos(t.repeat(1,channels//2)*inv_freq)
        pos_enc=torch.cat([pos_enc_a,pos_enc_b],dim=-1)
        
        return pos_enc

    def forward(self,x,t):
        t=t.unsqueeze(-1).type(torch.float)
        t=self.pos_encoding(t,self.time_dim)

        x1=self.inc(x)
        x2=self.down1(x1,t)
        x2=self.sa1(x2)
        x3=self.down2(x2,t)
        x3=self.sa2(x3)
        x4=self.down3(x3,t)
        x4=self.sa3(x4)

        x4=self.bot1(x4)
        x4=self.bot2(x4)
        x4=self.bot3(x4)

        x=self.up1(x4,x3,t)
        x=self.sa4(x)
        x=self.up2(x,x2,t)
        x=self.sa5(x)
        x=self.up3(x,x1,t)
        x=self.sa6(x)
        output=self.outc(x)

        return output


# ============================================================================
# RefineNet Architecture for NCSN
# Based on "Generative Modeling by Estimating Gradients of the Data Distribution"
# by Song & Ermon, NeurIPS 2019.
# ============================================================================

class ResidualConvUnit(nn.Module):
    """
    Residual Convolution Unit (RCU) for RefineNet.
    Applies residual convolutions with normalization.
    """
    def __init__(self, channels, n_blocks=2):
        super(ResidualConvUnit, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, channels),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, x):
        residual = x
        for block in self.blocks:
            x = block(x)
        return x + residual


class MultiResolutionFusion(nn.Module):
    """
    Multi-Resolution Fusion (MRF) module.
    Fuses features from different resolutions.
    """
    def __init__(self, channels_list, out_channels):
        super(MultiResolutionFusion, self).__init__()
        self.channels_list = channels_list
        self.out_channels = out_channels
        
        # Convolution layers for each input resolution
        self.convs = nn.ModuleList()
        for channels in channels_list:
            if channels != out_channels:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(8, out_channels),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.convs.append(nn.Identity())
    
    def forward(self, *inputs):
        # Upsample all inputs to the same size (largest resolution)
        target_size = inputs[0].shape[2:]
        upsampled = []
        
        for i, x in enumerate(inputs):
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x = self.convs[i](x)
            upsampled.append(x)
        
        # Sum all upsampled features
        return sum(upsampled)


class ChainedResidualPooling(nn.Module):
    """
    Chained Residual Pooling (CRP) module.
    Applies chained max pooling operations with residual connections.
    """
    def __init__(self, channels, n_stages=2):
        super(ChainedResidualPooling, self).__init__()
        self.stages = nn.ModuleList()
        for i in range(n_stages):
            self.stages.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, channels),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, x):
        out = x
        for stage in self.stages:
            out = stage(out) + out
        return out


class RefineNetBlock(nn.Module):
    """
    RefineNet Block combining RCU, MRF, and CRP.
    """
    def __init__(self, channels_list, out_channels, n_rcu=2, n_crp=2, emb_dim=256):
        super(RefineNetBlock, self).__init__()
        self.out_channels = out_channels
        
        # RCU blocks for each input
        self.rcus = nn.ModuleList()
        for channels in channels_list:
            self.rcus.append(ResidualConvUnit(channels, n_rcu))
        
        # Multi-Resolution Fusion
        self.mrf = MultiResolutionFusion(channels_list, out_channels)
        
        # Chained Residual Pooling
        self.crp = ChainedResidualPooling(out_channels, n_crp)
        
        # Final RCU
        self.final_rcu = ResidualConvUnit(out_channels, n_rcu)
        
        # Sigma conditioning layer
        self.sigma_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
    
    def forward(self, *inputs, sigma_emb=None):
        # Apply RCU to each input
        rcu_outputs = [rcu(x) for rcu, x in zip(self.rcus, inputs)]
        
        # Multi-Resolution Fusion
        fused = self.mrf(*rcu_outputs)
        
        # Add sigma conditioning if provided
        if sigma_emb is not None:
            sigma_scale = self.sigma_cond(sigma_emb)[:, :, None, None]
            fused = fused + sigma_scale
        
        # Chained Residual Pooling
        pooled = self.crp(fused)
        
        # Final RCU
        output = self.final_rcu(pooled)
        
        return output


class RefineNetNCSN(nn.Module):
    """
    RefineNet for Noise Conditional Score Networks (NCSN).
    This is the correct architecture as used in the original NCSN paper.
    Based on "Generative Modeling by Estimating Gradients of the Data Distribution"
    by Song & Ermon, NeurIPS 2019.
    """
    def __init__(self, c_in=3, c_out=3, sigma_dim=256, device="cuda"):
        super(RefineNetNCSN, self).__init__()
        self.device = device
        self.sigma_dim = sigma_dim
        
        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (downsampling path) - similar to ResNet blocks
        # Block 1: 64 -> 128, H/2 x W/2
        self.encoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: 128 -> 256, H/4 x W/4
        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: 256 -> 512, H/8 x W/8
        self.encoder3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True)
        )
        
        # RefineNet blocks (decoder path)
        # RefineNet Block 4: processes the deepest features (x3)
        self.refinenet4 = RefineNetBlock([512], 256, n_rcu=2, n_crp=2, emb_dim=sigma_dim)
        
        # RefineNet Block 3: fuses features from encoder2 (x2) and refinenet4 output
        self.refinenet3 = RefineNetBlock([256, 256], 256, n_rcu=2, n_crp=2, emb_dim=sigma_dim)
        
        # RefineNet Block 2: fuses features from encoder1 (x1) and refinenet3 output
        self.refinenet2 = RefineNetBlock([128, 256], 256, n_rcu=2, n_crp=2, emb_dim=sigma_dim)
        
        # RefineNet Block 1: fuses features from encoder0 (x0) and refinenet2 output
        self.refinenet1 = RefineNetBlock([64, 256], 256, n_rcu=2, n_crp=2, emb_dim=sigma_dim)
        
        # Final output layer
        self.conv_out = nn.Sequential(
            ResidualConvUnit(256, n_blocks=2),
            nn.Conv2d(256, c_out, kernel_size=1, bias=True)
        )
        
        # Sigma embedding following NCSN paper
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, sigma_dim),
            nn.SiLU(),
            nn.Linear(sigma_dim, sigma_dim),
        )
    
    def sigma_embedding(self, sigma):
        """
        Embed continuous sigma (noise level) values.
        Following NCSN paper: use log(sigma) for numerical stability.
        
        Args:
            sigma: Noise levels (B,) - actual sigma values
        
        Returns:
            Embedded sigma (B, sigma_dim)
        """
        sigma_log = torch.log(sigma.clamp(min=1e-8)).unsqueeze(-1)  # (B, 1)
        return self.sigma_embed(sigma_log)  # (B, sigma_dim)
    
    def forward(self, x, sigma):
        """
        Forward pass.
        
        Args:
            x: Noisy images (B, C, H, W) in range [-1, 1]
            sigma: Noise levels (B,) - actual sigma values, not indices
        
        Returns:
            Predicted score (B, C, H, W) - gradient of log probability
        """
        # Embed sigma values
        sigma_emb = self.sigma_embedding(sigma)  # (B, sigma_dim)
        
        # Encoder path
        x0 = self.conv_in(x)  # (B, 64, H, W)
        x1 = self.encoder1(x0)  # (B, 128, H/2, W/2)
        x2 = self.encoder2(x1)  # (B, 256, H/4, W/4)
        x3 = self.encoder3(x2)  # (B, 512, H/8, W/8)
        
        # RefineNet decoder path with sigma conditioning
        # Each block fuses features from corresponding encoder level and previous refinenet block
        r4 = self.refinenet4(x3, sigma_emb=sigma_emb)  # (B, 256, H/8, W/8)
        
        # Upsample r4 to match x2 resolution (H/4 x W/4)
        r4_up = F.interpolate(r4, size=x2.shape[2:], mode='bilinear', align_corners=False)
        r3 = self.refinenet3(x2, r4_up, sigma_emb=sigma_emb)  # (B, 256, H/4, W/4)
        
        # Upsample r3 to match x1 resolution (H/2 x W/2)
        r3_up = F.interpolate(r3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        r2 = self.refinenet2(x1, r3_up, sigma_emb=sigma_emb)  # (B, 256, H/2, W/2)
        
        # Upsample r2 to match x0 resolution (H x W)
        r2_up = F.interpolate(r2, size=x0.shape[2:], mode='bilinear', align_corners=False)
        r1 = self.refinenet1(x0, r2_up, sigma_emb=sigma_emb)  # (B, 256, H, W)
        
        # r1 is already at original resolution (H x W)
        # Final output
        output = self.conv_out(r1)
        
        return output


# Alias for backward compatibility

    
