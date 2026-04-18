import torch

def posenc(x, L):
    freq = 2.0**torch.arange(L, device=x.device)
    x_expanded = x.unsqueeze(-1)*freq*torch.pi

    sin = torch.sin(x_expanded)
    cos = torch.cos(x_expanded)

    enc = torch.stack([sin, cos], dim=-1)
    enc = enc.view(*x.shape[:-1], -1)

    return torch.cat([x, enc], dim=-1)

def gen_rays(H, W, C2W, fx, fy, ox, oy):
    R = C2W[:3, :3]
    ray_origin = C2W[:3, 3]

    j, i = torch.meshgrid(torch.arange(H, device=C2W.device) + 0.5, torch.arange(W, device=C2W.device) + 0.5, indexing='ij')

    x = ((i - ox)).float()/fx
    y = (-(j - oy)).float()/fy
    z = -torch.ones_like(i).float() #blender convention

    dir = torch.stack([x,y,z], dim=-1)
    dir = dir @ R.T
    dir = dir/dir.norm(dim=-1, keepdim=True)

    return ray_origin[None, None, :].expand(H, W, 3), dir

def strat_sample(ray_origin, ray_dir, t0, t1, N):
    # ray_origin: (batch, 3)
    # ray_dir:    (batch, 3)
    # t0:         (batch, 1) or scalar
    # t1:         (batch, 1) or scalar
    device = ray_origin.device
    batch  = ray_origin.shape[0]

    # evenly spaced offsets from 0→1, one per sample
    u = torch.linspace(0.0, 1.0, N + 1, device=device)   # (N+1,)
    
    # interpolate per-ray: t0 + (t1 - t0) * u
    # t0, t1 are (batch, 1), u is (N+1,) → bins is (batch, N+1)
    bins  = t0 + (t1 - t0) * u                            # (batch, N+1)
    lower = bins[:, :-1]                                   # (batch, N)
    upper = bins[:, 1:]                                    # (batch, N)

    # jitter within each bin
    t = lower + (upper - lower) * torch.rand(batch, N, device=device)  # (batch, N)

    # 3D points along each ray
    points = ray_origin[:, None, :] + t[:, :, None] * ray_dir[:, None, :]
    # (batch,1,3) + (batch,N,1) * (batch,1,3) → (batch, N, 3)

    return points, t

def hierarchical_sample(sigma, t, N_f):
    w = get_points_w(sigma, t)  # (batch_size, N_c)
    w = w + 1e-5
    w = w/w.sum(dim=-1, keepdim=True)
    batch_size, N_c = t.shape

    zeros = torch.zeros(batch_size, 1, device=t.device)
    cdf = torch.cumsum(torch.cat([zeros, w], dim=-1), dim=-1)   # (batch_size, N_c + 1); [0, 1]

    u = torch.rand(batch_size, N_f, device=t.device)

    # cdf: (batch_size, N_c+1)
    # t:   (batch_size, N_c)
    # u:   (batch_size, N_f)

    idx = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)  # (batch_size, N_f)

    below = torch.clamp(idx - 1, min = 0)
    above = torch.clamp(idx, max=N_c)

    t_d = torch.cat([t, t[..., -1:]], dim = -1)

    idx_g = torch.stack([below, above], -1) #(batch_size, N_f, 2)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, N_f, -1), 2, idx_g)    # (batch_size, N_f, 2)
    t_g = torch.gather(t_d.unsqueeze(1).expand(-1, N_f, -1), 2, idx_g)      # (batch_size, N_f, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t_n = t_g[..., 0] + ((u - cdf_g[..., 0])/denom)*(t_g[..., 1] - t_g[..., 0])
    # torch.sort returns tuple
    return torch.sort(torch.cat([t, t_n], dim = -1), dim=-1).values

def get_points_w(sigma, t):
    sigma = sigma.squeeze(-1)   # (batch_size, N)

    delta = t[..., 1:] - t[..., :-1]    # (batch_size, N-1)
    last = torch.full(
        (t.shape[0], 1), 1e10, device = t.device
    )

    delta = torch.cat([delta, last], dim=-1)    # (batch_size, N)

    alpha = 1 - torch.exp(-sigma*delta)

    ones = torch.ones(alpha.shape[0], 1, device=t.device)

    T = torch.cumprod(
        torch.cat([ones, 1 - alpha + 1e-10], dim=-1), dim=-1
    )[..., :-1]                                 # (batch_size, N)
    # 1e-10 is for numerical stability
    return T*alpha

def vol_rendering(rgb, sigma, t, white_bg = True):
    # rgb:   (batch_size, N, 3)
    # sigma: (batch_size, N, 1)
    # t:     (batch_size, N)

    w = get_points_w(sigma, t)  # (batch_size, N)

    color = (w.unsqueeze(-1)*rgb).sum(dim=1)    # (batch_size, 3)

    if white_bg:
        acc = w.sum(dim=-1, keepdim=True)                 # (batch, 1)  — accumulated opacity
        color = color + (1.0 - acc)                       # blend with white

    return color

def render_rays(rays_o, rays_d, t0, t1, model_c, model_f, N_c, N_f, L_pos, L_dir, use_hierarchical=True, return_weights=False):

    batch_size = rays_o.shape[0]

    input_dim = 3 + 6*L_pos
    dir_dim = 3 + 6*L_dir

    d_c_exp = rays_d.unsqueeze(1).expand(-1, N_c, -1)
    d_f_exp = rays_d.unsqueeze(1).expand(-1, N_c + N_f, -1)

    # COARSE NETWORK

    s, t = strat_sample(rays_o, rays_d, t0, t1, N_c)

    s_c = posenc(s, L_pos).reshape(-1, input_dim)
    d_c = posenc(d_c_exp, L_dir).reshape(-1, dir_dim)

    rgb_c, sigma_c = model_c(s_c, d_c)

    rgb_c = rgb_c.reshape(batch_size, N_c, 3)
    sigma_c = sigma_c.reshape(batch_size, N_c, 1)

    predicted_c = vol_rendering(rgb_c, sigma_c, t)

    if not use_hierarchical:
        if return_weights:
            weights_c = get_points_w(sigma_c.detach(), t)

            return predicted_c, predicted_c, weights_c, t, weights_c, t

        return predicted_c, predicted_c

    # FINE NETWORK

    t_n = hierarchical_sample(sigma_c.detach(), t.detach(), N_f)
    s_n = rays_o[..., None, :] + t_n[..., None]*rays_d[..., None, :]

    s_f = posenc(s_n, L_pos).reshape(-1, input_dim)
    d_f = posenc(d_f_exp, L_dir).reshape(-1, dir_dim)


    rgb_f, sigma_f = model_f(s_f, d_f)

    rgb_f = rgb_f.reshape(batch_size, N_c + N_f, 3)
    sigma_f = sigma_f.reshape(batch_size, N_c + N_f, 1)

    predicted_f = vol_rendering(rgb_f, sigma_f, t_n)

    if return_weights:

        weights_c = get_points_w(sigma_c.detach(), t)
        weights_f = get_points_w(sigma_f.detach(), t_n)

        return predicted_c, predicted_f, weights_c, t, weights_f, t_n

    return predicted_c, predicted_f
