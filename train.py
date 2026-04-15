import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml
from tqdm import tqdm

from dataset import ColmapDataset, BlenderDataset
from model import Net
from render import render_rays, gen_rays 

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _validate(dataset, model_c, model_f, config, iteration, device):
    model_c.eval()
    model_f.eval()

    H, W = dataset.H, dataset.W
    batch_size = config["batch_size"]

    with torch.no_grad():
        val_rays_o, val_rays_d = gen_rays(H, W, dataset.val_pose, dataset.fx, dataset.fy, dataset.ox, dataset.oy)
        val_rays_o = val_rays_o.reshape(-1, 3)
        val_rays_d = val_rays_d.reshape(-1, 3)

        t0, t1 = dataset.get_bounds(-1)

        predicted_pixels = []

        for i in range(0, H*W, batch_size):
            chunk_o = val_rays_o[i:i+batch_size]
            chunk_d = val_rays_d[i:i+batch_size]
            chunk_size = chunk_o.shape[0]

            _, pred_f = render_rays(chunk_o, chunk_d, t0, t1, model_c, model_f)
            predicted_pixels.append(pred_f)

        predicted_image = torch.cat(predicted_pixels, dim=0).reshape(H, W, 3)

        val_img = dataset.val_image

        mse  = torch.mean((predicted_image - val_img) ** 2)
        psnr = -10.0 * torch.log10(mse)
        print(f"iter {iteration:6d}  PSNR: {psnr.item():.2f} dB")

        # save side-by-side
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(predicted_image.cpu().numpy())
        plt.title(f"Predicted (iter {iteration})")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(val_img.cpu().numpy())
        plt.title("Ground Truth")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{config['save_dir']}/val/render_{iteration:06d}.png")
        plt.close()

def train(config):

    device = config["device"]
    save_dir = config["save_dir"]

    os.makedirs(f"{save_dir}/ckpts", exist_ok=True)
    os.makedirs(f"{save_dir}/val",   exist_ok=True)
    os.makedirs(f"{save_dir}/video", exist_ok=True)

    if config["dataset_type"] == "Blender":
        dataset = BlenderDataset(
            basedir = config["data_dir"],
            split = "train",
            device = device,
            precompute_rays = config["precompute_rays"],
            t0 = config["t_near"],
            t1 = config["t_far"]
        )

    elif config["dataset_type"] == "Colmap":
        dataset = ColmapDataset(
            basedir = config["data_dir"],
            device = device,
            precompute_rays = config["precompute_rays"]
        )

    else:
        raise ValueError(f"Unknown dataset_type: {config['dataset_type']}")

    L_pos = config["L_pos"]
    L_dir = config["L_dir"]

    input_dim = 3 + 6*L_pos
    dir_dim = 3 + 6*L_dir

    model_c = Net(input_dim, dir_dim).to(device)
    model_f = Net(input_dim, dir_dim).to(device)

    optimizer = optim.Adam(
        list(model_c.parameters()) + list(model_f.parameters()),
        lr = config["lr"]
    )

    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma = 0.1 ** (1.0/config["N_iters"])
    )

    start_iter = 0
    ckpt_path = config.get("resume_ckpt", None)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_c.load_state_dict(ckpt["model_c"])
        model_f.load_state_dict(ckpt["model_f"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = ckpt["step"] + 1
        print(f"Resumed from iteration {start_iter}") 

    pbar = tqdm(range(start_iter, config["N_iters"], ncols=100))

    for iteration in pbar:

        model_c.train()
        model_f.train()

        rays_o, rays_d, targe_rgb, t0, t1 = dataset.sample_batch(config["batch_size"])

        predicted_c, predicted_f = render_rays(rays_o, rays_d, t0, t1, model_c, model_f, config["N_c"], config["N_f"], L_pos, L_dir)

        loss = torch.mean((predicted_f - target_rgb)**2) + torch.mean((predicted_c - target_rgb)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if iteration % config["ckpt_every"] == 0 and iteration > 0:
            torch.save({
                "model_c":   model_c.state_dict(),
                "model_f":   model_f.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step":      iteration,
            }, f"{save_dir}/ckpts/ckpt_{iteration:06d}.pth")

        if iteration % config["val_every"] == 0 and iteration > 0:
            _validate(dataset, model_c, model_f, config, iteration, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to yaml config")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
