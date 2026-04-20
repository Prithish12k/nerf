import argparse
import os
import math
import torch
import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image

from model import Net
from dataset import ColmapDataset, BlenderDataset
from render import render_rays, gen_rays


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def look_at(eye, target, up=np.array([0.0, 1.0, 0.0])):
    """Build a camera-to-world matrix looking from `eye` toward `target`."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    new_up = np.cross(right, forward)

    # OpenGL convention: camera looks along -Z
    C2W = np.eye(4, dtype=np.float32)
    C2W[:3, 0] = right
    C2W[:3, 1] = new_up
    C2W[:3, 2] = -forward
    C2W[:3, 3] = eye
    return torch.from_numpy(C2W)


def generate_spiral_poses(center, radius_start, radius_end, height_start,
                          height_end, n_frames, n_revolutions, up_axis="y"):
    """
    Generate camera poses spiraling inward around `center`.

    Args:
        center:         (3,) numpy array — point to look at
        radius_start:   starting radius of the spiral
        radius_end:     ending radius (smaller = closer)
        height_start:   starting camera height offset from center
        height_end:     ending camera height offset
        n_frames:       total number of frames
        n_revolutions:  number of full 360° rotations
        up_axis:        which axis is "up" — "y" (OpenGL) or "z"
    """
    poses = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)  # 0 → 1

        angle = 2.0 * math.pi * n_revolutions * t
        radius = radius_start + (radius_end - radius_start) * t
        height = height_start + (height_end - height_start) * t

        if up_axis == "y":
            eye = center + np.array([
                radius * math.cos(angle),
                height,
                radius * math.sin(angle),
            ], dtype=np.float32)
            up = np.array([0.0, 1.0, 0.0])
        else:  # z-up
            eye = center + np.array([
                radius * math.cos(angle),
                radius * math.sin(angle),
                height,
            ], dtype=np.float32)
            up = np.array([0.0, 0.0, 1.0])

        poses.append(look_at(eye, center, up))
    return poses


def render_frame(pose, dataset, model_c, model_f, config, t0, t1, device):
    """Render a single (H, W, 3) image for a given pose."""
    H, W = dataset.H, dataset.W
    batch_size = config["batch_size"]
    N_c, N_f = config["N_c"], config["N_f"]
    L_pos, L_dir = config["L_pos"], config["L_dir"]
    use_hier = config.get("use_hierarchical", True)

    rays_o, rays_d = gen_rays(H, W, pose.to(device),
                              dataset.fx, dataset.fy, dataset.ox, dataset.oy)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    t0_batch = torch.full((H * W, 1), float(t0), device=device)
    t1_batch = torch.full((H * W, 1), float(t1), device=device)

    model_c.eval()
    model_f.eval()
    pred_pixels = []
    with torch.no_grad():
        for i in range(0, H * W, batch_size):
            _, pred_f = render_rays(
                rays_o[i:i + batch_size], rays_d[i:i + batch_size],
                t0_batch[i:i + batch_size], t1_batch[i:i + batch_size],
                model_c, model_f, N_c, N_f, L_pos, L_dir,
                use_hierarchical=use_hier, return_weights=False,
            )
            pred_pixels.append(pred_f)
    return torch.cat(pred_pixels, dim=0).reshape(H, W, 3)


def main():
    parser = argparse.ArgumentParser(description="Render spiral video from trained NeRF")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--out", default="spiral_output", help="Output directory")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--revolutions", type=float, default=2.0, help="Number of 360° rotations")
    parser.add_argument("--radius_start", type=float, default=None, help="Starting radius (default: auto from dataset)")
    parser.add_argument("--radius_end", type=float, default=None, help="Ending radius (default: 0.5× start)")
    parser.add_argument("--height_start", type=float, default=None, help="Starting height offset (default: auto)")
    parser.add_argument("--height_end", type=float, default=None, help="Ending height offset (default: 0.5× start)")
    parser.add_argument("--fps", type=int, default=30, help="FPS for video (if ffmpeg available)")
    parser.add_argument("--data_dir", default=None, help="Override data_dir from config (e.g. for Colab)")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.device:
        config["device"] = args.device
    device = config["device"]

    # ── Load dataset (for intrinsics + bounds) ──
    if config["dataset_type"] == "Blender":
        dataset = BlenderDataset(
            basedir=config["data_dir"], split="train", device=device,
            precompute_rays=False, t0=config["t_near"], t1=config["t_far"],
        )
    elif config["dataset_type"] == "Colmap":
        dataset = ColmapDataset(
            basedir=config["data_dir"], device=device, precompute_rays=False,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {config['dataset_type']}")

    # ── Load model ──
    L_pos, L_dir = config["L_pos"], config["L_dir"]
    input_dim = 3 + 6 * L_pos
    dir_dim = 3 + 6 * L_dir

    model_c = Net(input_dim, dir_dim).to(device)
    model_f = Net(input_dim, dir_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model_c.load_state_dict(ckpt["model_c"])
    model_f.load_state_dict(ckpt["model_f"])
    print(f"Loaded checkpoint from step {ckpt['step']}")

    # ── Compute scene center + radius from training poses ──
    all_poses = torch.cat([dataset.poses, dataset.val_pose.unsqueeze(0)], dim=0)
    cam_centers = all_poses[:, :3, 3].cpu().numpy()  # (N, 3)
    scene_center = cam_centers.mean(axis=0)

    dists = np.linalg.norm(cam_centers - scene_center, axis=1)
    avg_radius = float(dists.mean())
    avg_height = float((cam_centers[:, 1] - scene_center[1]).mean())  # y-axis offset

    radius_start = args.radius_start if args.radius_start is not None else avg_radius
    radius_end = args.radius_end if args.radius_end is not None else avg_radius * 0.5
    height_start = args.height_start if args.height_start is not None else avg_height
    height_end = args.height_end if args.height_end is not None else avg_height * 0.5

    print(f"Scene center: {scene_center}")
    print(f"Spiral: radius {radius_start:.2f} → {radius_end:.2f}, "
          f"height {height_start:.2f} → {height_end:.2f}, "
          f"{args.revolutions} rev, {args.frames} frames")

    # ── Determine near/far bounds ──
    if config["dataset_type"] == "Blender":
        t0, t1 = config["t_near"], config["t_far"]
    else:
        # Use the median of all per-image bounds
        t0 = float(dataset.t0_img.median())
        t1 = float(dataset.t1_img.median())
    print(f"Bounds: t0={t0:.3f}, t1={t1:.3f}")

    # ── Generate spiral poses ──
    poses = generate_spiral_poses(
        center=scene_center,
        radius_start=radius_start,
        radius_end=radius_end,
        height_start=height_start,
        height_end=height_end,
        n_frames=args.frames,
        n_revolutions=args.revolutions,
    )

    # ── Render frames ──
    os.makedirs(args.out, exist_ok=True)
    for i, pose in enumerate(tqdm(poses, desc="Rendering spiral")):
        img = render_frame(pose, dataset, model_c, model_f, config, t0, t1, device)
        img_np = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(args.out, f"frame_{i:04d}.png"))

    print(f"\nSaved {args.frames} frames to {args.out}/")

    # ── Try to create video with ffmpeg ──
    video_path = os.path.join(args.out, "spiral.mp4")
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate {args.fps} "
        f"-i {args.out}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 "
        f"{video_path}"
    )
    ret = os.system(ffmpeg_cmd)
    if ret == 0:
        print(f"Video saved to {video_path}")
    else:
        print(f"ffmpeg not available or failed. Frames saved as PNGs in {args.out}/")
        print(f"To create video manually:\n  {ffmpeg_cmd}")


if __name__ == "__main__":
    main()
