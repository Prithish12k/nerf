import torch
import torch.nn.functional as F
import numpy as np
import time
import csv
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_fn
import lpips as lpips_lib


def build_lpips(device):
    return lpips_lib.LPIPS(net="alex").to(device)


def build_csv_logger(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "iter",
        "train_loss",
        "coarse_loss",
        "fine_loss",
        "grad_norm",
        "psnr",
        "ssim",
        "lpips",
        "weight_entropy",
        "fine_concentration",
    ]

    file_exists = log_path.exists()
    f = open(log_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

    if not file_exists:
        writer.writeheader()
        f.flush()

    def log(**kwargs):
        row = {k: "" for k in fieldnames}
        row.update(kwargs)
        writer.writerow(row)
        f.flush()

    return log


def compute_psnr(mse: torch.Tensor) -> float:
    return (-10.0 * torch.log10(mse)).item()


def get_grad_norm(params, max_norm: float = 1.0) -> float:
    return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm).item()


def compute_img_metrics(predicted: torch.Tensor, gt: torch.Tensor,
                        lpips_model, device) -> dict:
    # PSNR
    mse  = F.mse_loss(predicted, gt)
    psnr = compute_psnr(mse)

    # SSIM
    # skimage expects HWC numpy in [0, 1] with channel_axis specified
    predicted_np = predicted.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    ssim_val = ssim_fn(predicted_np, gt_np, data_range=1.0, channel_axis=2)

    # LPIPS
    # lpips expects NCHW in [-1, 1]
    def to_lpips(t):
        return t.detach().permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

    with torch.no_grad():
        lpips_val = lpips_model(to_lpips(predicted), to_lpips(gt)).item()

    return {"psnr": psnr, "ssim": float(ssim_val), "lpips": lpips_val}


def compute_weight_entropy(weights: torch.Tensor, eps: float = 1e-8) -> float:
    w = weights / (weights.sum(dim=-1, keepdim=True) + eps)   # (N_rays, N_samples)
    entropy = -(w * torch.log(w + eps)).sum(dim=-1)            # (N_rays,)
    return entropy.mean().item()


def compute_fine_concentration(
    fine_weights: torch.Tensor,    # (N_rays, N_c+N_f)
    fine_t_vals: torch.Tensor,     # (N_rays, N_c+N_f)
    coarse_weights: torch.Tensor,  # (N_rays, N_c)
    coarse_t_vals: torch.Tensor,   # (N_rays, N_c)
    delta: float = 0.1,
) -> float:
    eps = 1e-8
    w_c = coarse_weights / (coarse_weights.sum(dim=-1, keepdim=True) + eps)
    surface_t = (w_c * coarse_t_vals).sum(dim=-1, keepdim=True)  # (N_rays, 1)

    near_surface = (fine_t_vals - surface_t).abs() < delta        # (N_rays, N_c+N_f) bool
    return near_surface.float().mean().item()


def save_validation_artifacts(rendered_rgb: torch.Tensor, gt_rgb: torch.Tensor,
                               depth_weights: torch.Tensor, depth_t_vals: torch.Tensor,
                               save_dir: str, iteration: int):
    import torchvision

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def hwc_to_chw(t):
        return t.detach().cpu().permute(2, 0, 1).clamp(0, 1)

    torchvision.utils.save_image(
        hwc_to_chw(rendered_rgb),
        save_dir / f"render_{iteration:06d}.png"
    )

    error_amplified = ((rendered_rgb - gt_rgb).abs() * 5.0).clamp(0, 1)
    torchvision.utils.save_image(
        hwc_to_chw(error_amplified),
        save_dir / f"error_{iteration:06d}.png"
    )

    H, W = rendered_rgb.shape[:2]
    eps  = 1e-8
    w    = depth_weights / (depth_weights.sum(dim=-1, keepdim=True) + eps)
    depth = (w * depth_t_vals).sum(dim=-1).reshape(H, W)

    d_min, d_max = depth.min(), depth.max()
    depth_norm   = (depth - d_min) / (d_max - d_min + eps)
    depth_rgb    = depth_norm.unsqueeze(-1).expand(H, W, 3)

    torchvision.utils.save_image(
        hwc_to_chw(depth_rgb),
        save_dir / f"depth_{iteration:06d}.png"
    )


def evaluate_test_set(render_fns, test_images_gt, lpips_model, device):

    all_psnr, all_ssim, all_lpips, all_times = [], [], [], []

    for fn, gt in zip(render_fns, test_images_gt):
        t_start = time.perf_counter()
        with torch.no_grad():
            rendered = fn()                     # (H, W, 3)
        t_end = time.perf_counter()

        m = compute_img_metrics(rendered, gt, lpips_model, device)
        all_psnr.append(m["psnr"])
        all_ssim.append(m["ssim"])
        all_lpips.append(m["lpips"])
        all_times.append(t_end - t_start)

    return {
        "psnr_mean":           np.mean(all_psnr),
        "psnr_std":            np.std(all_psnr),
        "ssim_mean":           np.mean(all_ssim),
        "ssim_std":            np.std(all_ssim),
        "lpips_mean":          np.mean(all_lpips),
        "lpips_std":           np.std(all_lpips),
        "inference_time_mean": np.mean(all_times),
    }
