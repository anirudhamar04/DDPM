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

class CondInstanceNormPlusPlus(nn.Module):
    """
    Conditional Instance Normalization++ (InstanceNorm2dPlus) for NCSN.
    Matches the official implementation: normalizes instance stats AND channel means.
    Conditions on sigma embedding as specified in the paper.
    """
    def __init__(self, num_features, emb_dim=256, bias=True):
        super(CondInstanceNormPlusPlus, self).__init__()
        self.num_features = num_features
        self.bias = bias
        
        # Use standard InstanceNorm2d without affine (we'll add our own)
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        
        # Learnable parameters for channel mean normalization
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Conditional parameters from sigma embedding
        self.cond_scale = nn.Sequential(
            nn.Linear(emb_dim, num_features),
            nn.SiLU()
        )
        self.cond_shift = nn.Sequential(
            nn.Linear(emb_dim, num_features),
            nn.SiLU()
        )
    
    def forward(self, x, sigma_emb):
        """
        Args:
            x: Input tensor (B, C, H, W)
            sigma_emb: Sigma embedding (B, emb_dim)
        """
        # Compute means across spatial dimensions: (B, C)
        means = torch.mean(x, dim=(2, 3))
        
        # Normalize means across channels: (B, 1)
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means_normalized = (means - m) / (torch.sqrt(v + 1e-5))
        
        # Apply instance normalization
        h = self.instance_norm(x)
        
        # Add normalized means back with learnable alpha
        h = h + means_normalized[..., None, None] * self.alpha[..., None, None]
        
        # Apply learnable gamma and beta
        if self.bias:
            h = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
        else:
            h = self.gamma.view(-1, self.num_features, 1, 1) * h
        
        # Apply conditional scale and shift from sigma embedding
        cond_scale = self.cond_scale(sigma_emb)  # (B, C)
        cond_shift = self.cond_shift(sigma_emb)  # (B, C)
        cond_scale = cond_scale.view(-1, self.num_features, 1, 1)
        cond_shift = cond_shift.view(-1, self.num_features, 1, 1)
        
        out = cond_scale * h + cond_shift
        return out


class ResidualConvUnit(nn.Module):
    """
    Residual Convolution Unit (RCU) for RefineNet.
    PRE-ACTIVATION: norm -> act -> conv (as in official code)
    """
    def __init__(self, channels, n_blocks=2, emb_dim=256):
        super(ResidualConvUnit, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            for j in range(2):  # 2 stages per block: norm->act->conv
                self.blocks.append(CondInstanceNormPlusPlus(channels, emb_dim))
                self.blocks.append(nn.ELU(inplace=True))
                self.blocks.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
    
    def forward(self, x, sigma_emb):
        residual = x
        i = 0
        while i < len(self.blocks):
            if isinstance(self.blocks[i], CondInstanceNormPlusPlus):
                x = self.blocks[i](x, sigma_emb)
            elif isinstance(self.blocks[i], nn.ELU):
                x = self.blocks[i](x)
            else:  # Conv2d
                x = self.blocks[i](x)
            i += 1
        return x + residual


class MultiResolutionFusion(nn.Module):
    """
    Multi-Resolution Fusion (MRF) module.
    Order: norm -> conv (as in official code, no ELU after)
    """
    def __init__(self, channels_list, out_channels, emb_dim=256):
        super(MultiResolutionFusion, self).__init__()
        self.channels_list = channels_list
        self.out_channels = out_channels
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for channels in channels_list:
            self.norms.append(CondInstanceNormPlusPlus(channels, emb_dim))
            if channels != out_channels:
                self.convs.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=True))
            else:
                self.convs.append(nn.Identity())
    
    def forward(self, *inputs, sigma_emb):
        target_size = inputs[0].shape[2:]
        upsampled = []
        
        for i, x in enumerate(inputs):
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            x = self.norms[i](x, sigma_emb)
            x = self.convs[i](x)
            upsampled.append(x)
        
        return sum(upsampled)


class ChainedResidualPooling(nn.Module):
    """
    Chained Residual Pooling (CRP) module.
    Order: act -> norm -> pool -> conv (as in official code)
    Uses average pooling (not max pooling) as specified in the paper.
    """
    def __init__(self, channels, n_stages=2, emb_dim=256):
        super(ChainedResidualPooling, self).__init__()
        self.n_stages = n_stages
        self.act = nn.ELU(inplace=True)
        self.pools = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_stages):
            self.pools.append(nn.AvgPool2d(kernel_size=5, stride=1, padding=2))
            self.norms.append(CondInstanceNormPlusPlus(channels, emb_dim))
            self.convs.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
    
    def forward(self, x, sigma_emb):
        out = x
        for i in range(self.n_stages):
            out = self.act(out)
            out = self.norms[i](out, sigma_emb)
            out = self.pools[i](out)
            out = self.convs[i](out)
            out = out + x  # Residual connection
        return out


class RefineNetBlock(nn.Module):
    """
    RefineNet Block combining RCU, MRF, and CRP.
    All components use CondInstanceNorm++ and ELU as per paper.
    """
    def __init__(self, channels_list, out_channels, n_rcu=2, n_crp=2, emb_dim=256):
        super(RefineNetBlock, self).__init__()
        self.out_channels = out_channels
        
        # RCU blocks for each input (with sigma conditioning)
        self.rcus = nn.ModuleList()
        for channels in channels_list:
            self.rcus.append(ResidualConvUnit(channels, n_rcu, emb_dim))
        
        # Multi-Resolution Fusion (with sigma conditioning)
        self.mrf = MultiResolutionFusion(channels_list, out_channels, emb_dim)
        
        # Chained Residual Pooling (with sigma conditioning)
        self.crp = ChainedResidualPooling(out_channels, n_crp, emb_dim)
        
        # Final RCU (with sigma conditioning)
        self.final_rcu = ResidualConvUnit(out_channels, n_rcu, emb_dim)
    
    def forward(self, *inputs, sigma_emb):
        # Apply RCU to each input (with sigma conditioning)
        rcu_outputs = [rcu(x, sigma_emb) for rcu, x in zip(self.rcus, inputs)]
        
        # Multi-Resolution Fusion (with sigma conditioning)
        fused = self.mrf(*rcu_outputs, sigma_emb=sigma_emb)
        
        # Chained Residual Pooling (with sigma conditioning)
        pooled = self.crp(fused, sigma_emb)
        
        # Final RCU (with sigma conditioning)
        output = self.final_rcu(pooled, sigma_emb)
        
        return output


class RefineNetNCSN(nn.Module):
    """
    RefineNet for Noise Conditional Score Networks (NCSN).
    Matches the paper architecture exactly:
    - 4-cascaded RefineNet
    - Pre-activation residual blocks
    - CondInstanceNorm++ (no batch normalization)
    - Average pooling (not max pooling)
    - ELU activations
    - Dilated convolutions (except first block)
    - For 64x64 images: 128 filters for first cascade, doubled for others
    """
    def __init__(self, c_in=3, c_out=3, sigma_dim=256, device="cuda", img_size=64):
        super(RefineNetNCSN, self).__init__()
        self.device = device
        self.sigma_dim = sigma_dim
        
        # For 64x64 images, use 128 filters for first cascade (paper: halved for MNIST, full for CelebA/CIFAR-10)
        # Since we're using 64x64, we'll use the full size (128 for first cascade)
        filters_cascade1 = 128
        filters_cascade2 = 256
        filters_cascade3 = 512
        filters_cascade4 = 512
        
        # Initial convolution with CondInstanceNorm++ (will be applied in forward)
        self.conv_in = nn.Conv2d(c_in, filters_cascade1, kernel_size=7, padding=3, bias=False)
        self.norm_in = CondInstanceNormPlusPlus(filters_cascade1, sigma_dim)
        
        # Encoder with pre-activation residual blocks and dilated convolutions
        # Block 1: First block uses stride=2, others use dilation
        self.encoder1_conv1 = nn.Conv2d(filters_cascade1, filters_cascade1, kernel_size=3, padding=1, stride=2, bias=False)
        self.encoder1_norm1 = CondInstanceNormPlusPlus(filters_cascade1, sigma_dim)
        self.encoder1_conv2 = nn.Conv2d(filters_cascade1, filters_cascade2, kernel_size=3, padding=1, bias=False)
        self.encoder1_norm2 = CondInstanceNormPlusPlus(filters_cascade2, sigma_dim)
        
        # Block 2: Use dilation=2 instead of stride
        self.encoder2_conv1 = nn.Conv2d(filters_cascade2, filters_cascade2, kernel_size=3, padding=2, dilation=2, bias=False)
        self.encoder2_norm1 = CondInstanceNormPlusPlus(filters_cascade2, sigma_dim)
        self.encoder2_conv2 = nn.Conv2d(filters_cascade2, filters_cascade3, kernel_size=3, padding=1, bias=False)
        self.encoder2_norm2 = CondInstanceNormPlusPlus(filters_cascade3, sigma_dim)
        
        # Block 3: Use dilation=4
        self.encoder3_conv1 = nn.Conv2d(filters_cascade3, filters_cascade3, kernel_size=3, padding=4, dilation=4, bias=False)
        self.encoder3_norm1 = CondInstanceNormPlusPlus(filters_cascade3, sigma_dim)
        self.encoder3_conv2 = nn.Conv2d(filters_cascade3, filters_cascade4, kernel_size=3, padding=1, bias=False)
        self.encoder3_norm2 = CondInstanceNormPlusPlus(filters_cascade4, sigma_dim)
        
        # RefineNet blocks (decoder path) - 4 cascades
        # RefineNet Block 4: processes the deepest features (x3)
        self.refinenet4 = RefineNetBlock([filters_cascade4], filters_cascade3, n_rcu=2, n_crp=2, emb_dim=sigma_dim)
        
        # RefineNet Block 3: fuses features from encoder2 (x2) and refinenet4 output
        self.refinenet3 = RefineNetBlock([filters_cascade3, filters_cascade3], filters_cascade2, n_rcu=2, n_crp=2, emb_dim=sigma_dim)
        
        # RefineNet Block 2: fuses features from encoder1 (x1) and refinenet3 output
        self.refinenet2 = RefineNetBlock([filters_cascade2, filters_cascade2], filters_cascade1, n_rcu=2, n_crp=2, emb_dim=sigma_dim)
        
        # RefineNet Block 1: fuses features from encoder0 (x0) and refinenet2 output
        self.refinenet1 = RefineNetBlock([filters_cascade1, filters_cascade1], filters_cascade1, n_rcu=2, n_crp=2, emb_dim=sigma_dim)
        
        # Final output layer with CondInstanceNorm++
        self.final_rcu = ResidualConvUnit(filters_cascade1, n_blocks=2, emb_dim=sigma_dim)
        self.conv_out = nn.Conv2d(filters_cascade1, c_out, kernel_size=1, bias=True)
        
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
        Forward pass with pre-activation residual blocks and CondInstanceNorm++.
        
        Args:
            x: Noisy images (B, C, H, W) in range [-1, 1]
            sigma: Noise levels (B,) - actual sigma values, not indices
        
        Returns:
            Predicted score (B, C, H, W) - gradient of log probability
        """
        # Embed sigma values
        sigma_emb = self.sigma_embedding(sigma)  # (B, sigma_dim)
        
        # Initial convolution: norm -> act -> conv (pre-activation)
        x0 = self.norm_in(self.conv_in(x), sigma_emb)
        x0 = F.elu(x0, inplace=True)
        
        # Encoder path with PRE-ACTIVATION: norm -> act -> conv -> norm -> act -> conv
        # Block 1
        x1 = self.encoder1_norm1(x0, sigma_emb)
        x1 = F.elu(x1, inplace=True)
        x1 = self.encoder1_conv1(x1)
        x1 = self.encoder1_norm2(x1, sigma_emb)
        x1 = F.elu(x1, inplace=True)
        x1 = self.encoder1_conv2(x1)
        
        # Block 2
        x2 = self.encoder2_norm1(x1, sigma_emb)
        x2 = F.elu(x2, inplace=True)
        x2 = self.encoder2_conv1(x2)
        x2 = self.encoder2_norm2(x2, sigma_emb)
        x2 = F.elu(x2, inplace=True)
        x2 = self.encoder2_conv2(x2)
        
        # Block 3
        x3 = self.encoder3_norm1(x2, sigma_emb)
        x3 = F.elu(x3, inplace=True)
        x3 = self.encoder3_conv1(x3)
        x3 = self.encoder3_norm2(x3, sigma_emb)
        x3 = F.elu(x3, inplace=True)
        x3 = self.encoder3_conv2(x3)
        
        # RefineNet decoder path with sigma conditioning
        # Each block fuses features from corresponding encoder level and previous refinenet block
        r4 = self.refinenet4(x3, sigma_emb=sigma_emb)  # (B, 512, H/2, W/2)
        
        # Upsample r4 to match x2 resolution
        r4_up = F.interpolate(r4, size=x2.shape[2:], mode='bilinear', align_corners=False)
        r3 = self.refinenet3(x2, r4_up, sigma_emb=sigma_emb)  # (B, 256, H/2, W/2)
        
        # Upsample r3 to match x1 resolution
        r3_up = F.interpolate(r3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        r2 = self.refinenet2(x1, r3_up, sigma_emb=sigma_emb)  # (B, 128, H/2, W/2)
        
        # Upsample r2 to match x0 resolution
        r2_up = F.interpolate(r2, size=x0.shape[2:], mode='bilinear', align_corners=False)
        r1 = self.refinenet1(x0, r2_up, sigma_emb=sigma_emb)  # (B, 128, H, W)
        
        # Final RCU and output
        r1_final = self.final_rcu(r1, sigma_emb)
        output = self.conv_out(r1_final)
        
        return output


# Alias for backward compatibility

    
