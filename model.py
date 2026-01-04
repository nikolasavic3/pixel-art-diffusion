import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AttentionBlock(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.kv_proj = nn.Linear(emb_dim, channels * 2)

    def forward(self, x, cond_emb):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        x_ln = self.ln(x_flat)
        
        kv = self.kv_proj(cond_emb).unsqueeze(1)
        k, v = kv.chunk(2, dim=-1)
        
        attn_out, _ = self.mha(query=x_ln, key=k, value=v)
        x_flat = x_flat + attn_out
        return x_flat.permute(0, 2, 1).view(b, c, h, w)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(t_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = F.silu(self.bn1(self.conv1(x)))
        t_emb = self.time_mlp(F.silu(t))[(...,) + (None,) * 2]
        h = self.bn2(self.conv2(h + t_emb))
        return F.silu(h + self.shortcut(x))

class ScalableUNet(nn.Module):
    def __init__(self, num_classes, img_size=16, base_c=128, t_dim=256):
        super().__init__()
        self.t_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU()
        )
        self.label_emb = nn.Embedding(num_classes + 1, t_dim)
        
        # Encoder
        self.init_conv = nn.Conv2d(3, base_c, 3, padding=1)
        self.down1 = ResBlock(base_c, base_c, t_dim)          # Output: base_c
        self.down2 = ResBlock(base_c, base_c * 2, t_dim)      # Output: base_c * 2
        
        # Mid
        self.mid_attn = AttentionBlock(base_c * 2, t_dim)
        self.mid_res = ResBlock(base_c * 2, base_c * 2, t_dim)
        
        # Decoder
        # Concatenation logic: 
        # up1_in = mid_res_out (base_c * 2) + down2_out (base_c * 2) = base_c * 4
        self.up1 = ResBlock(base_c * 4, base_c * 2, t_dim)
        
        # up2_in = up1_out (base_c * 2) + down1_out (base_c) = base_c * 3
        self.up2 = ResBlock(base_c * 3, base_c, t_dim)
        
        self.final_attn = AttentionBlock(base_c, t_dim)
        self.out = nn.Conv2d(base_c, 3, 3, padding=1)

    def forward(self, x, t, labels):
        t_vec = self.t_mlp(t)
        l_vec = self.label_emb(labels)
        c_vec = t_vec + l_vec 
        
        # Encoder
        x1 = self.init_conv(x)         # [B, base_c, 16, 16]
        x1_res = self.down1(x1, c_vec) # [B, base_c, 16, 16]
        
        x2_in = F.avg_pool2d(x1_res, 2) # [B, base_c, 8, 8]
        x2_res = self.down2(x2_in, c_vec) # [B, base_c * 2, 8, 8]
        
        # Mid
        x_mid = self.mid_attn(x2_res, l_vec)
        x_mid = self.mid_res(x_mid, c_vec)
        
        # Decoder
        # 1. Up to 8x8 (already 8x8, but conceptually we process)
        # Concatenate x_mid (base_c*2) with x2_res (base_c*2)
        x_up1 = torch.cat([x_mid, x2_res], dim=1) # [B, base_c * 4, 8, 8]
        x_up1 = self.up1(x_up1, c_vec)            # [B, base_c * 2, 8, 8]
        
        # 2. Up to 16x16
        x_up2_in = F.interpolate(x_up1, scale_factor=2, mode='nearest') # [B, base_c * 2, 16, 16]
        x_up2_in = torch.cat([x_up2_in, x1_res], dim=1)                 # [B, base_c * 3, 16, 16]
        x_up2 = self.up2(x_up2_in, c_vec)                               # [B, base_c, 16, 16]
        
        x_final = self.final_attn(x_up2, l_vec)
        return self.out(x_final)

class PixelDiffusion:
    def __init__(self, model, image_size=16, device="cpu"):
        self.model = model.to(device)
        self.image_size = image_size
        self.device = device
        self.betas = torch.linspace(1e-4, 0.04, 1000).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def train_step(self, images, labels, optimizer, loss_fn):
        self.model.train()
        n = images.shape[0]
        t = torch.randint(0, 1000, (n,), device=self.device).long()
        
        alpha_t = self.alpha_hat[t][:, None, None, None]
        noise = torch.randn_like(images)
        x_t = torch.sqrt(alpha_t) * images + torch.sqrt(1 - alpha_t) * noise
        
        # Drop labels for CFG
        mask = (torch.rand(n, device=self.device) > 0.15).long()
        labels = labels * mask
        
        pred_noise = self.model(x_t, t, labels)
        loss = loss_fn(noise, pred_noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, labels, cfg_scale=7.5):
        self.model.eval()
        n = labels.shape[0]
        x = torch.randn((n, 3, self.image_size, self.image_size), device=self.device)
        
        for i in reversed(range(1000)):
            t = (torch.ones(n) * i).long().to(self.device)
            
            n_cond = self.model(x, t, labels)
            n_uncond = self.model(x, t, torch.zeros_like(labels))
            eps = n_uncond + cfg_scale * (n_cond - n_uncond)
            
            a_t = self.alpha_hat[i]
            a_prev = self.alpha_hat[i-1] if i > 0 else torch.tensor(1.0).to(self.device)
            beta_t = self.betas[i]
            
            noise = torch.randn_like(x) if i > 0 else 0
            x = (1 / torch.sqrt(self.alphas[i])) * (x - (beta_t / torch.sqrt(1 - a_t)) * eps) + torch.sqrt(beta_t) * noise
            
        x = (x.clamp(-1, 1) + 1) / 2
        return (x * 15).round() / 15