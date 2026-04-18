import torch
import torch.optim as optim
import os
import yaml
from tqdm import tqdm

from dataset import ColmapDataset, BlenderDataset
from model import Net
from render import render_rays, gen_rays
from metrics import (
    build_lpips,
    build_csv_logger,
    get_grad_norm,
    compute_img_metrics,
    compute_weight_entropy,
    compute_fine_concentration,
    save_validation_artifacts,
    evaluate_test_set,
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_render_fn(dataset, model_c, model_f, config, pose, t0, t1, device):
    H, W = dataset.H, dataset.W
    batch_size = config["batch_size"]
    N_c, N_f = config["N_c"], config["N_f"]
    L_pos, L_dir = config["L_pos"], config["L_dir"]
    use_hier = config.get("use_hierarchical", True)

    rays_o, rays_d = gen_rays(H, W, pose, dataset.fx, dataset.fy, dataset.ox, dataset.oy)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    # expand scalar bounds to (H*W, 1) for strat_sample broadcasting
    t0_batch = torch.full((H * W, 1), float(t0), device=device)
    t1_batch = torch.full((H * W, 1), float(t1), device=device)

    def render_fn():
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

    return render_fn


def _validate(dataset, model_c, model_f, config, iteration, device, lpips_model, log):
    model_c.eval()
    model_f.eval()

    H, W = dataset.H, dataset.W
    batch_size = config["batch_size"]
    N_c, N_f = config["N_c"], config["N_f"]
    L_pos, L_dir = config["L_pos"], config["L_dir"]
    is_hier = config.get("hierarchical", True)
    use_hier = config.get("use_hierarchical", True)

    with torch.no_grad():
        val_rays_o, val_rays_d = gen_rays(
            H, W, dataset.val_pose, dataset.fx, dataset.fy, dataset.ox, dataset.oy
        )
        val_rays_o = val_rays_o.reshape(-1, 3)
        val_rays_d = val_rays_d.reshape(-1, 3)

        # BlenderDataset: get_bounds ignores idx, returns global t0/t1
        # ColmapDataset:  t0_img[-1] is the val pose entry (last in _compute_bounds)
        t0_val, t1_val = dataset.get_bounds(-1)
        t0_batch = torch.full((H * W, 1), float(t0_val), device=device)
        t1_batch = torch.full((H * W, 1), float(t1_val), device=device)

        pred_pixels = []
        all_weights_c, all_t_c = [], []
        all_weights_f, all_t_f = [], []

        for i in range(0, H * W, batch_size):
            pred_c, pred_f, w_c, t_c, w_f, t_f = render_rays(
                val_rays_o[i:i + batch_size], val_rays_d[i:i + batch_size],
                t0_batch[i:i + batch_size], t1_batch[i:i + batch_size],
                model_c, model_f, N_c, N_f, L_pos, L_dir,
                use_hierarchical=use_hier, return_weights=True,
            )
            pred_pixels.append(pred_f)
            all_weights_c.append(w_c)
            all_t_c.append(t_c)
            all_weights_f.append(w_f)
            all_t_f.append(t_f)

        predicted_image = torch.cat(pred_pixels, dim=0).reshape(H, W, 3)
        weights_c = torch.cat(all_weights_c, dim=0)
        t_c = torch.cat(all_t_c, dim=0)
        weights_f = torch.cat(all_weights_f, dim=0)
        t_f = torch.cat(all_t_f, dim=0)

        val_img = dataset.val_image
        img_m = compute_img_metrics(predicted_image, val_img, lpips_model, device)

        weights_for_entropy = weights_f if is_hier else weights_c
        t_for_depth = t_f if is_hier else t_c
        entropy = compute_weight_entropy(weights_for_entropy)

        fine_conc = ""
        if is_hier:
            fine_conc = round(
                compute_fine_concentration(
                    weights_f, t_f, weights_c, t_c,
                    delta=config.get("fine_conc_delta", 0.1),
                ), 4
            )

        log(
            iter=iteration,
            psnr=round(img_m["psnr"], 4),
            ssim=round(img_m["ssim"], 4),
            lpips=round(img_m["lpips"], 4),
            weight_entropy=round(entropy, 4),
            fine_concentration=fine_conc,
        )

        print(
            f"iter {iteration:6d}  "
            f"PSNR={img_m['psnr']:.2f}dB  "
            f"SSIM={img_m['ssim']:.3f}  "
            f"LPIPS={img_m['lpips']:.4f}  "
            f"Entropy={entropy:.3f}"
            + (f"  FineConc={fine_conc:.3f}" if is_hier else "")
        )

        save_validation_artifacts(
            rendered_rgb=predicted_image,
            gt_rgb=val_img,
            depth_weights=weights_for_entropy,
            depth_t_vals=t_for_depth,
            save_dir=f"{config['save_dir']}/artifacts",
            iteration=iteration,
        )

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(predicted_image.cpu().numpy())
        plt.title(f"Predicted (iter {iteration})  PSNR={img_m['psnr']:.2f}dB")
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
    os.makedirs(f"{save_dir}/val", exist_ok=True)
    os.makedirs(f"{save_dir}/video", exist_ok=True)
    os.makedirs(f"{save_dir}/artifacts", exist_ok=True)

    if config["dataset_type"] == "Blender":
        dataset = BlenderDataset(
            basedir=config["data_dir"],
            split="train",
            device=device,
            precompute_rays=config["precompute_rays"],
            t0=config["t_near"],
            t1=config["t_far"],
        )
    elif config["dataset_type"] == "Colmap":
        dataset = ColmapDataset(
            basedir=config["data_dir"],
            device=device,
            precompute_rays=config["precompute_rays"],
        )
    else:
        raise ValueError(f"Unknown dataset_type: {config['dataset_type']}")

    L_pos, L_dir = config["L_pos"], config["L_dir"]
    input_dim = 3 + 6*L_pos
    dir_dim = 3 + 6*L_dir

    model_c = Net(input_dim, dir_dim).to(device)
    model_f = Net(input_dim, dir_dim).to(device)

    optimizer = optim.Adam(
        list(model_c.parameters()) + list(model_f.parameters()),
        lr=config["lr"],
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.1 ** (1.0/config["N_iters"]),
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

    lpips_model = build_lpips(device)
    log = build_csv_logger(f"{save_dir}/metrics.csv")

    pbar = tqdm(range(start_iter, config["N_iters"]), ncols=100)

    for iteration in pbar:
        model_c.train()
        model_f.train()

        rays_o, rays_d, target_rgb, t0, t1 = dataset.sample_batch(config["batch_size"])

        predicted_c, predicted_f = render_rays(
            rays_o, rays_d, t0, t1,
            model_c, model_f,
            config["N_c"], config["N_f"],
            L_pos, L_dir,
            use_hierarchical=config.get("use_hierarchical", True),
            return_weights=False,
        )

        coarse_loss = torch.mean((predicted_c - target_rgb)**2)
        fine_loss = torch.mean((predicted_f - target_rgb)**2)
        loss = coarse_loss + fine_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = get_grad_norm(
            list(model_c.parameters()) + list(model_f.parameters())
        )
        optimizer.step()
        scheduler.step()

        if iteration % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            log(
                iter=iteration,
                train_loss=round(loss.item(), 6),
                coarse_loss=round(coarse_loss.item(), 6),
                fine_loss=round(fine_loss.item(), 6),
                grad_norm=round(grad_norm, 6),
            )

        if iteration % config["ckpt_every"] == 0 and iteration > 0:
            torch.save({
                "model_c": model_c.state_dict(),
                "model_f": model_f.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": iteration,
            }, f"{save_dir}/ckpts/ckpt_{iteration:06d}.pth")

        if iteration % config["val_every"] == 0 and iteration > 0:
            _validate(dataset, model_c, model_f, config, iteration, device, lpips_model, log)

    print("\nRunning final evaluation...")
    t0_val, t1_val = dataset.get_bounds(-1)
    render_fns = [build_render_fn(dataset, model_c, model_f, config,
                                  dataset.val_pose, t0_val, t1_val, device)]
    results = evaluate_test_set(render_fns, [dataset.val_image], lpips_model, device)

    print(f"\n{config.get('model_name', save_dir)}")
    print(f"PSNR  : {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"SSIM  : {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"LPIPS : {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
    print(f"Infer : {results['inference_time_mean']:.2f} s/frame")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))
