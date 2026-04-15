import torch
import numpy as np
from PIL import Image
import os
import json
from render import gen_rays

def qvec2rotmat(q):
    q = q / torch.norm(q)
    w, x, y, z = q

    return torch.tensor([
        [1 - 2*(y**2) - 2*(z**2), 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*(x**2) - 2*(z**2), 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*(x**2) - 2*(y**2)]
    ], dtype=torch.float32)

def colmap_c2w(q, t):

    R = qvec2rotmat(q)
    R_inv = R.T
    C2W = torch.eye(4, device=R.device, dtype=R.dtype)

    C2W[:3, :3] = R_inv
    C2W[:3, 3] = -R_inv @ t
    C2W[:3, 1:3] *= -1          # flip Y and Z axes: OpenCV -> OpenGL

    return C2W

class BlenderDataset():
    def __init__(self, basedir, split, device, precompute_rays, t0, t1):
        self.device = device
        self.images, self.poses, self.val_image, self.val_pose = self._load_split(basedir, split)
        self.t0, self.t1 = t0, t1
        self.all_rays_o = None
        self.all_rays_d = None
        self.all_colors = None
        self.all_rays_o = None
        self.all_rays_d = None
        self.all_colors = None
        if precompute_rays:
            self.all_rays_o, self.all_rays_d, self.all_colors = self._precompute_all_rays()

    def _load_split(self, basedir, split):
        with open(f'{basedir}/transforms_{split}.json', 'r') as f:
            meta = json.load(f)

            poses = []
            images = []
            fov = meta["camera_angle_x"]

            for frame in meta["frames"]:
                pose = np.array(frame["transform_matrix"])
                poses.append(torch.from_numpy(pose).float())

                img_path = f"{basedir}/{frame['file_path']}.png"
                img = np.array(Image.open(img_path)).astype(np.float32) / 255.0   # (H, W, 4)
                img = img[..., :3] * img[..., 3:4] + (1.0 - img[..., 3:4])       # RGBA → RGB over white
                images.append(torch.from_numpy(img).float())

            poses = torch.stack(poses, dim=0).to(self.device)
            images = torch.stack(images, dim=0).to(self.device)

            H, W = images[0].shape[:2]

            self._derive_intrinsics(H, W, fov)
            return images[1:], poses[1:], images[0], poses[0]
        
    def _derive_intrinsics(self, H, W, fov):
        self.H = H
        self.W = W
        self.fx = self.fy = 0.5 * W / np.tan(0.5 * fov)
        self.ox = 0.5 * W
        self.oy = 0.5 * H

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx]
    
    def _precompute_all_rays(self):
        N = len(self)

        all_rays_o = []
        all_rays_dir = []
        all_colors = []

        for i in range(N):
            rays_o, rays_dir = gen_rays(self.H, self.W, self.poses[i], self.fx, self.fy, self.ox, self.oy)      # (H, W, 3)
            
            all_rays_o.append(rays_o.reshape(-1, 3))
            all_rays_dir.append(rays_dir.reshape(-1, 3))
            all_colors.append(self.images[i].reshape(-1, 3))

        all_rays_o = torch.cat(all_rays_o, dim=0).to(self.device)       # (N_img*H*W, 3)
        all_rays_dir = torch.cat(all_rays_dir, dim=0).to(self.device)   # (N_img*H*W, 3)
        all_colors = torch.cat(all_colors, dim=0).to(self.device)       # (N_img*H*W, 3)

        return all_rays_o, all_rays_dir, all_colors
    
    def get_rays(self, idx, pixel_coords=None):
        pose = self.poses[idx]

        rays_o, rays_d = gen_rays(
            self.H, self.W, pose,
            self.fx, self.fy,
            self.ox, self.oy
        )

        if pixel_coords is None:
            return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

        else:
            rows = pixel_coords[:, 0]
            cols = pixel_coords[:, 1]
            return rays_o[rows, cols], rays_d[rows, cols]
        
    def get_bounds(self, idx=None):
        return self.t0, self.t1

    def sample_batch(self, batch_size):
        idx = torch.randint(0, len(self)*self.H*self.W, (batch_size,), device=self.device)

        if self.all_rays_o is not None:
            rays_o = self.all_rays_o[idx]
            rays_d = self.all_rays_d[idx]
            cols = self.all_colors[idx]
            return rays_o, rays_d, cols, self.t0, self.t1
        
        else:
            img_idx = torch.randint(0, len(self), (1,)).item()
            row_idx = torch.randint(0, self.H, (batch_size,), device=self.device)
            col_idx = torch.randint(0, self.W, (batch_size,), device=self.device)

            pixel_coords = torch.stack([row_idx, col_idx], dim=-1)
            rays_o, rays_d = self.get_rays(img_idx, pixel_coords)
            cols = self.images[img_idx][row_idx, col_idx]

            return rays_o, rays_d, cols, self.t0, self.t1 

class ColmapDataset():
    def __init__(self, basedir, device, precompute_rays):
        self.device = device
        self.H, self.W, self.fx, self.fy, self.ox, self.oy = self._parse_camera(f"{basedir}/sparse_txt")
        self.images, self.poses, self.val_image, self.val_pose = self._parse_images(f"{basedir}/images", f"{basedir}/sparse_txt")
        self.points3D = self._parse_points(f"{basedir}/sparse_txt")
        self._compute_bounds()
        self.all_rays_o = None
        self.all_rays_d = None
        self.all_colors = None
        self.all_t0 = None
        self.all_t1 = None
        if precompute_rays:
            self._precompute_all_rays()
    
    def _parse_camera(self, path):
        camera = None

        with open(f"{path}/cameras.txt") as f:
            for line in f:
                line = line.strip()
                
                if line.startswith("#") or not line:
                    continue
                
                camera = line
                break

        parts = camera.split()
        _, model, W, H, fx, fy, ox, oy = parts

        W, H = int(W), int(H)
        fx, fy, ox, oy = map(float, (fx, fy, ox, oy))

        return H, W, fx, fy, ox, oy
    
    def _parse_images(self, image_path, sparse_path):
        pose = []
        dataline = 0
        with open(f"{sparse_path}/images.txt") as f:
            for line in f:

                line = line.strip()

                if line.startswith("#") or not line:
                    continue
                
                dataline += 1

                if dataline % 2 == 0:
                    continue
                
                parts = line.split()

                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                img = int(parts[-1].split(".")[0])

                pose.append([qw, qx, qy, qz, tx, ty, tz, img])

        pose = sorted(pose, key=lambda x: x[7])

        pose_mat = []
        images = []

        for p in pose:
            q = torch.tensor(p[:4], dtype=torch.float32)
            t = torch.tensor(p[4:7], dtype=torch.float32)
            img = p[7]

            img_path = f"{image_path}/{img:04d}.jpg"
            img = np.array(Image.open(img_path)).astype(np.float32) / 255.0   # (H, W, 3 or 4)
            if img.shape[-1] == 4:
                img = img[..., :3] * img[..., 3:4] + (1.0 - img[..., 3:4])       # RGBA to RBG (over white)
            images.append(torch.from_numpy(img).float())

            pose_mat.append(colmap_c2w(q, t))
        
        images, poses = torch.stack(images, dim=0).to(self.device), torch.stack(pose_mat, dim=0).to(self.device)

        return images[1:], poses[1:], images[0], poses[0]
    
    def _parse_points(self, sparse_path):
        pts = []
    
        with open(f"{sparse_path}/points3D.txt") as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                pts.append(torch.tensor([x, y, z], dtype=torch.float32))
        return torch.stack(pts, dim=0).to(self.device)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx]
    
    def _compute_bounds(self, quantile_low=0.05, quantile_high=0.95):
        t0s = []
        t1s = []

        for i in range(len(self) + 1):

            pose = self.val_pose if i == len(self) else self.poses[i]

            W2C = torch.inverse(pose)
            R = W2C[:3, :3]
            t = W2C[:3, 3]

            cam_pts = (R @ self.points3D.T).T + t
            z_pts = -cam_pts[:, 2]
            z_pts = z_pts[z_pts > 0]

            if len(z_pts) == 0:
                t0s.append(0.01)
                t1s.append(10.0)
                continue

            t_near = float(torch.quantile(z_pts, quantile_low) * 0.9)
            t_far  = float(torch.quantile(z_pts, quantile_high) * 1.1)
            t0s.append(max(t_near, 0.01))
            t1s.append(t_far)

        self.t0_img = torch.tensor(t0s, device=self.device)
        self.t1_img = torch.tensor(t1s, device=self.device)

    def _precompute_all_rays(self):
        all_rays_o, all_rays_d, all_colors = [], [], []
        all_t0, all_t1 = [], []

        for i in range(len(self)):
            rays_o, rays_d = gen_rays(self.H, self.W, self.poses[i], self.fx, self.fy, self.ox, self.oy)
            n = self.H * self.W

            all_rays_o.append(rays_o.reshape(-1, 3))
            all_rays_d.append(rays_d.reshape(-1, 3))
            all_colors.append(self.images[i].reshape(-1, 3))
            all_t0.append(torch.full((n,), self.t0_img[i].item(), device=self.device))
            all_t1.append(torch.full((n,), self.t1_img[i].item(), device=self.device))

        self.all_rays_o = torch.cat(all_rays_o, dim=0)
        self.all_rays_d = torch.cat(all_rays_d, dim=0)
        self.all_colors = torch.cat(all_colors, dim=0)
        self.all_t0 = torch.cat(all_t0, dim=0)[:, None]      # (N_img*H*W,)
        self.all_t1 = torch.cat(all_t1, dim=0)[:, None]

    def get_rays(self, idx, pixel_coords=None):
        pose = self.poses[idx]

        rays_o, rays_d = gen_rays(
            self.H, self.W, pose,
            self.fx, self.fy,
            self.ox, self.oy
        )

        if pixel_coords is None:
            return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

        else:
            rows = pixel_coords[:, 0]
            cols = pixel_coords[:, 1]
            return rays_o[rows, cols], rays_d[rows, cols]

    def get_bounds(self, idx):
        return self.t0_img[idx], self.t1_img[idx]

    def sample_batch(self, batch_size):
        if self.all_rays_o is not None:
            idx = torch.randint(0, len(self)*self.H*self.W, (batch_size,), device=self.device)
            return (
                self.all_rays_o[idx],
                self.all_rays_d[idx],
                self.all_colors[idx],
                self.all_t0[idx],    
                self.all_t1[idx],
            )
        else:
            img_idx = torch.randint(0, len(self), (1,)).item()
            row_idx = torch.randint(0, self.H, (batch_size,), device=self.device)
            col_idx = torch.randint(0, self.W, (batch_size,), device=self.device)
            pixel_coords = torch.stack([row_idx, col_idx], dim=-1)
            rays_o, rays_d = self.get_rays(img_idx, pixel_coords)
            cols = self.images[img_idx][row_idx, col_idx]
            t0, t1 = self.get_bounds(img_idx)    
            return rays_o, rays_d, cols, t0, t1
