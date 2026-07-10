import torch
import torch.nn as nn
import math
import numpy as np
from scipy.io import savemat

class MoCapVAEDataset(torch.utils.data.Dataset):
    def __init__(self, X_full, X_ds):
        self.X_full = X_full
        self.X_ds   = X_ds

    def __len__(self):
        return self.X_full.shape[0]

    def __getitem__(self, i):
        x_full = self.X_full[i]
        x_ds   = self.X_ds[i]
        return x_ds.squeeze(), x_full.squeeze()

class TrajectoryVAE(nn.Module):
    def __init__(self, D, d_model, nhead, num_layers_enc, num_layers_dec, dim_ff, dropout, K, d_latent, T_full):
        super().__init__()
        self.K = K
        self.T_full = T_full # Time length
        self.enc = TrajectoryEncoder(D, d_model, nhead, num_layers_enc, dim_ff, dropout, K, d_latent)
        self.dec = TrajectoryDecoder(D, d_model, nhead, num_layers_dec, dim_ff, dropout, K, d_latent, T_full)
        self.vector_field = VectorField(d_latent, d_hidden=128)
        self.register_buffer("t_grid", torch.linspace(0.0, 1.0, steps=K))
        self.register_buffer("t_grid_full", torch.linspace(0.0, 1.0, steps=T_full))
        self.step_dec = TimeStepDecoder(D, d_latent, d_hidden=256, use_time_embed=True)

    def reparameterize(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def cnf_traj_from_C(self, C):
        z0 = C[:, 0, :]
        return rk4_integrate(self.vector_field, z0, self.t_grid)

    def cnf_traj_full(self, C_or_z0):
        if C_or_z0.dim() == 3:
            z0 = C_or_z0[:, 0, :]
        else:
            z0 = C_or_z0
        return rk4_integrate(self.vector_field, z0, self.t_grid_full)

    def forward(self, x_ds, deterministic=False):
        mu, logvar = self.enc(x_ds)
        C = self.reparameterize(mu, logvar, deterministic)
        ztraj_K = self.cnf_traj_from_C(C)
        ztraj_T = self.cnf_traj_full(C)
        B = x_ds.size(0)
        t_norm = self.t_grid_full.view(1, -1, 1).expand(B, -1, -1)
        x_hat = self.step_dec(ztraj_T, t_norm=t_norm)
        return x_hat, mu, logvar, C, ztraj_K, ztraj_T

class TimeStepDecoder(nn.Module):
    def __init__(self, D, d_latent, d_hidden=256, use_time_embed=True):
        super().__init__()
        self.use_time_embed = use_time_embed
        t_dim = 16 if use_time_embed else 0
        in_dim = d_latent + t_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, D)
        )

    @staticmethod
    def time_embedding(t_norm, dim=16):
        device = t_norm.device
        B, T, _ = t_norm.shape
        i = torch.arange(dim//2, device=device).float()
        freqs = 2.0 ** i  # 1,2,4,8,...
        ang = t_norm * freqs.view(1,1,-1) * 2.0*math.pi
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return emb

    def forward(self, z_full, t_norm=None):
        if self.use_time_embed:
            emb = self.time_embedding(t_norm, dim=16)
            h = torch.cat([z_full, emb], dim=-1)
        else:
            h = z_full
        B, T, _ = h.shape
        x_hat = self.net(h.view(B*T, -1)).view(B, T, -1)
        return x_hat

class TrajectoryEncoder(nn.Module):
    def __init__(self, D, d_model, nhead, num_layers, dim_ff, dropout, K, d_latent):
        super().__init__()
        self.K = K
        self.inp = InputProj(D, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.seg_queries = nn.Parameter(torch.randn(K, d_model))
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.to_mu = nn.Linear(d_model, d_latent)
        self.to_logvar = nn.Linear(d_model, d_latent)

    @staticmethod
    def _segment_bounds(T_ds, K):
        b = torch.linspace(0, T_ds, steps=K+1).long().tolist()
        for i in range(K):
            if b[i+1] <= b[i]:
                b[i+1] = min(b[i]+1, T_ds)
        b[-1] = T_ds
        return b

    def forward(self, x_ds):
        B, T_ds, _ = x_ds.shape
        h = self.inp(x_ds)
        h = self.pos(h)

        mask = torch.triu(torch.ones(T_ds, T_ds, device=x_ds.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask)

        bounds = self._segment_bounds(T_ds, self.K)
        seg_feats = []
        for k in range(self.K):
            s, e = bounds[k], bounds[k+1]
            kv = h[:, s:e, :].contiguous()
            qk = self.seg_queries[k].unsqueeze(0).unsqueeze(1).expand(B, 1, -1).contiguous()
            ctx, _ = self.mha(qk, kv, kv, need_weights=False)
            seg_feats.append(ctx.squeeze(1))

        Hk = torch.stack(seg_feats, dim=1)

        mu = self.to_mu(Hk)
        logvar = self.to_logvar(Hk)
        return mu, logvar

class TrajectoryDecoder(nn.Module):
    def __init__(self, D, d_model, nhead, num_layers, dim_ff, dropout, K, d_latent, T_full):
        super().__init__()
        self.T_full = T_full
        self.query_pos = SinusoidalPositionalEncoding(d_model, max_len=T_full+1)
        self.mem_proj = nn.Linear(d_latent, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out = OutputProj(d_model, D)

        self.query_pad = nn.Parameter(torch.randn(T_full, d_model))

    def forward(self, C):
        B = C.size(0)
        mem = self.mem_proj(C)

        Q = self.query_pad.unsqueeze(0).expand(B, self.T_full, -1).contiguous()
        Q = self.query_pos(Q)
        H = self.decoder(tgt=Q, memory=mem)

        x_hat = self.out(H)
        return x_hat

class VectorField(nn.Module):
    def __init__(self, d_latent, d_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + 1, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_latent)
        )
    def forward(self, z, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=z.device, dtype=z.dtype)
        if t.dim() == 0:
            t = t.expand(z.size(0),).unsqueeze(-1)  # [B,1]
        elif t.dim() == 1:
            t = t.unsqueeze(-1)  # [B,1]
        return self.net(torch.cat([z, t], dim=-1))

def rk4_integrate(f, z0, t_grid):
    B, d = z0.shape
    device = z0.device
    K = t_grid.shape[0]
    z = z0
    traj = [z0.unsqueeze(1)]  # [B,1,d]
    for k in range(1, K):
        t0, t1 = t_grid[k-1], t_grid[k]
        h = (t1 - t0).item()
        k1 = f(z, t0)                          # [B,d]
        k2 = f(z + 0.5*h*k1, t0 + 0.5*h)
        k3 = f(z + 0.5*h*k2, t0 + 0.5*h)
        k4 = f(z + h*k3,     t0 + h)
        z = z + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        traj.append(z.unsqueeze(1))
    return torch.cat(traj, dim=1)              # [B,K,d]

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]
    def forward(self, x):  # x: [B,T,d_model] -> add pos
        T = x.size(1)
        pe_slice = self.pe[:T].unsqueeze(0)        # [1,T,d]
        y = x + pe_slice
        return y

class InputProj(nn.Module):
    def __init__(self, D, d_model):
        super().__init__()
        self.lin = nn.Linear(D, d_model)
    def forward(self, x):   # [B,L,D] -> [B,L,d_model]
        y = self.lin(x)
        return y

class OutputProj(nn.Module):
    def __init__(self, d_model, D):
        super().__init__()
        self.lin = nn.Linear(d_model, D)
    def forward(self, x):   # [B,L,d_model] -> [B,L,D]
        y = self.lin(x)
        return y

def kld_normal(mu, logvar):
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=(-1, -2))  # sum over K,d

@torch.no_grad()
def compute_per_timestep_rmse(model,
                              X_ds_te_std,
                              X_full_te_phys,
                              mean_train,
                              std_train,
                              mean_train_t,
                              device):
    model.eval()

    N_te, T_full, D = X_full_te_phys.shape

    mean_train = mean_train.to(device)
    std_train = std_train.to(device)
    mean_train_t = mean_train_t.to(device)

    sq_err_model_sum = torch.zeros(T_full, device=device)
    sq_err_base_sum = torch.zeros(T_full, device=device)

    for i in range(N_te):
        x_ds_std = X_ds_te_std[i:i+1].to(device)
        x_full_phys = X_full_te_phys[i:i+1].to(device)

        x_hat_std, *_ = model(x_ds_std, deterministic=True)
        x_hat_phys = x_hat_std * std_train.view(1, 1, -1) + mean_train.view(1, 1, -1)

        err_model = (x_hat_phys - x_full_phys) ** 2
        mse_model_t = err_model.mean(dim=-1).squeeze(0)
        sq_err_model_sum += mse_model_t

        base_pred = mean_train_t.view(1, T_full, D).expand_as(x_full_phys)

        err_base = (base_pred - x_full_phys) ** 2
        mse_base_t = err_base.mean(dim=-1).squeeze(0)
        sq_err_base_sum += mse_base_t

    mse_model_t = sq_err_model_sum / N_te
    mse_base_t = sq_err_base_sum / N_te

    rmse_model_t = torch.sqrt(mse_model_t).detach().cpu().numpy()
    rmse_base_t = torch.sqrt(mse_base_t).detach().cpu().numpy()

    return rmse_model_t, rmse_base_t

@torch.no_grad()
def compute_per_timestep_rmse_masked(model,
                                     X_ds_te_std,
                                     X_full_te_phys,
                                     mean_train,
                                     std_train,
                                     mean_train_t,
                                     K,
                                     device):
    model.eval()

    N_te, T_full, D = X_full_te_phys.shape
    _, T_ds, _ = X_ds_te_std.shape

    base = T_ds // K
    remainder = T_ds % K
    seg0_end = base + (1 if remainder > 0 else 0)

    mean_train = mean_train.to(device)
    std_train = std_train.to(device)
    mean_train_t = mean_train_t.to(device)

    sq_err_model_sum = torch.zeros(T_full, device=device)
    sq_err_base_sum = torch.zeros(T_full, device=device)

    for i in range(N_te):
        x_ds_std = X_ds_te_std[i:i+1].to(device).clone()
        x_ds_std[:, seg0_end:, :] = 0.0

        x_full_phys = X_full_te_phys[i:i+1].to(device)

        x_hat_std, *_ = model(x_ds_std, deterministic=True)
        x_hat_phys = x_hat_std * std_train.view(1, 1, -1) + mean_train.view(1, 1, -1)

        err_model = (x_hat_phys - x_full_phys) ** 2
        mse_model_t = err_model.mean(dim=-1).squeeze(0)
        sq_err_model_sum += mse_model_t

        base_pred = mean_train_t.view(1, T_full, D).expand_as(x_full_phys)

        err_base = (base_pred - x_full_phys) ** 2
        mse_base_t = err_base.mean(dim=-1).squeeze(0)
        sq_err_base_sum += mse_base_t

    mse_model_t = sq_err_model_sum / N_te
    mse_base_t = sq_err_base_sum / N_te

    rmse_model_t = torch.sqrt(mse_model_t).detach().cpu().numpy()
    rmse_base_t = torch.sqrt(mse_base_t).detach().cpu().numpy()

    return rmse_model_t, rmse_base_t

@torch.no_grad()
def compute_latent_trajectories_3d_from_ds_std(model, X_ds_te_std, device):
    model.eval()
    z_list = []
    N_te = X_ds_te_std.shape[0]
    for i in range(N_te):
        x_ds_std = X_ds_te_std[i:i+1].to(device)
        mu, logvar = model.enc(x_ds_std)
        C = model.reparameterize(mu, logvar, deterministic=True)
        ztraj_T = model.cnf_traj_full(C)
        z_list.append(ztraj_T.squeeze(0).detach().cpu())
    Z = torch.stack(z_list, dim=0)
    return Z

@torch.no_grad()
def export_test_and_pred_to_mat(model,
                                X_ds_te_std,
                                X_full_te_phys,
                                mean_train, std_train,
                                device, K,
                                filename):
    model.eval()

    mean_train = mean_train.to(device)
    std_train  = std_train.to(device)

    N_te, T_full, D = X_full_te_phys.shape
    K_ignore = T_full // K
    T_used   = T_full - K_ignore

    X_gt_all   = np.zeros((N_te, T_used, D), dtype=np.float32)
    X_pred_all = np.zeros((N_te, T_used, D), dtype=np.float32)

    for i in range(N_te):
        x_ds_std    = X_ds_te_std[i:i+1].to(device)
        x_full_phys = X_full_te_phys[i:i+1].to(device)

        x_hat_std, *_ = model(x_ds_std, deterministic=True)
        x_hat_phys = x_hat_std * std_train.view(1, 1, -1) + mean_train.view(1, 1, -1)

        X_pred_all[i] = x_hat_phys[:, K_ignore:, :].squeeze(0).detach().cpu().numpy()
        X_gt_all[i]   = x_full_phys[:, K_ignore:, :].squeeze(0).detach().cpu().numpy()

    savemat(filename, {
        "X_gt":   X_gt_all,
        "X_pred": X_pred_all,
        "K":      np.array(K, dtype=np.int32),
        "T":      np.array(T_full, dtype=np.int32),
    })
    print(f"Saved mat to {filename}")