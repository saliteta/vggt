import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import tifffile as tif
import random

M = torch.diag(torch.tensor([1, -1, 1])) 


def pytorch3d_to_colmap_batch(camera_params: torch.Tensor,
                              point_maps: torch.Tensor,
                              rgb_images: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch‐wise conversion from PyTorch3D conventions to COLMAP conventions:
      - camera_params: (B, 4, 4) world→camera transforms in PyTorch3D coords
      - point_maps:    (B, H, W, 3) 3D point coordinates in PyTorch3D world
      - rgb_images:    (B, H, W, 3) RGB images (unchanged)
    Returns:
      - T_colmap:      (B, 4, 4) world→camera transforms in COLMAP/OpenCV coords
      - point_maps:    (B, H, W, 3) rotated point maps in COLMAP coords
      - rgb_images:    unchanged
    """
    device = camera_params.device
    dtype = camera_params.dtype
    B, H, W, _ = point_maps.shape

    # 1) Define the Y‐axis flip matrix (PyTorch3D Y‐up → COLMAP Y‐down)
    M3 = torch.diag(torch.tensor([1, -1, 1], dtype=dtype, device=device))

    # 2) Extract R and t from the 4×4 transforms
    R_py3d = camera_params[:, :3, :3]  # (B, 3, 3)
    t_py3d = camera_params[:, :3, 3]   # (B, 3)

    # 3) Convert each: R_colmap = M3 @ R_py3d @ M3, t_colmap = M3 @ t_py3d
    R_colmap = M3 @ R_py3d @ M3  # broadcast M3 over batch
    t_colmap = (M3 @ t_py3d.unsqueeze(-1)).squeeze(-1)  # (B,3)

    # 4) Build full 4×4 COLMAP transforms
    T_colmap = torch.eye(4, dtype=dtype, device=device)\
        .unsqueeze(0).repeat(B, 1, 1)   # (B,4,4)
    T_colmap[:, :3, :3] = R_colmap
    T_colmap[:, :3, 3]    = t_colmap

    # 5) Rotate point maps by M3
    pts_flat = point_maps.reshape(-1, 3).T     # (3, B*H*W)
    pts_rot  = (M3 @ pts_flat).T               # (B*H*W, 3)
    pts_new  = pts_rot.reshape(B, H, W, 3)     # (B, H, W, 3)

    return pts_new, T_colmap, rgb_images


class PairedDataset(Dataset):
    def __init__(self, data_dir:Path, num_images=3, data_conversion=None):
        self.data_dir = data_dir
        self.dataset_meta = list(self.data_dir.glob("*"))
        self.dataset_meta.sort()
        self.num_images = num_images
        self.data_conversion = data_conversion
    
    def _load_point_maps(self, points_path: List[Path]) -> torch.Tensor:
        point_maps = []
        for path in points_path:
            point_map = tif.imread(str(path))
            point_map_tensor = torch.from_numpy(point_map)
            point_maps.append(point_map_tensor)
        return torch.stack(point_maps)

    def _load_camera_params(self, camera_path: Path) -> Dict[str, torch.Tensor]:
        with open(camera_path, "r") as f:
            camera_data = json.load(f)
        
        # Create dictionary with filename as key and 4x4 transformation matrix as value
        camera_transforms = {}
        
        for camera in camera_data["cameras"]:
            # Create 4x4 transformation matrix
            transform_matrix = torch.tensor(camera["full_transform"])
            
            # Use image filename as key
            camera_transforms[camera["image_filename"].split(".")[0]] = transform_matrix
            
        return camera_transforms
    
    def _transform_to_first_camera_coordinates(self, camera_transforms: torch.Tensor, point_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Assume each camera_transforms[k] is a 4×4 world→cameraₖ matrix.
        E0 = camera_transforms[0].T             # world→camera₀
        E0_inv = torch.inverse(E0)

        # 1) Transform all point-maps into camera₀-frame:
        aligned_points = []
        all_pts = []
        for pts in point_maps:
            H, W, _ = pts.shape
            flat = pts.reshape(-1, 3)
            ones = torch.ones(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)
            hom = torch.cat([flat, ones], dim=1)      # (N,4)
            cam0 = (E0 @ hom.T).T[:, :3]              # (N,3)
            aligned_points.append(cam0.reshape(H, W, 3))
            all_pts.append(cam0)


        # 2) Compute scale:
        all_pts = torch.cat(all_pts, dim=0)
        max_dist = all_pts.norm(dim=1).max()

        # 3) Scale point-maps and camera translations:
        for i in range(len(aligned_points)):
            aligned_points[i] /= max_dist

        aligned_cams = []
        for E_i in camera_transforms:
            # express cameraᵢ in cam₀ frame
            E_i_cam0 = E_i.T @ E0_inv
            # scale translation part
            E_i_cam0[:3, 3] /= max_dist
            aligned_cams.append(E_i_cam0)

        return torch.stack(aligned_cams), torch.stack(aligned_points)
            
    def _load_rgb_image(self, rgb_image_path: List[Path]) -> torch.Tensor:
        rgb_images = []
        for path in rgb_image_path:
            rgb_image = Image.open(str(path))
            rgb_image_tensor = torch.from_numpy(np.array(rgb_image))
            rgb_images.append(rgb_image_tensor[:,:,:3].permute(2,0,1) / 255.0) # remove alpha channel
        return torch.stack(rgb_images)

    def __len__(self):
        return len(self.dataset_meta)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_path = self.dataset_meta[idx]
        camera_path = list(sequence_path.glob("*.json"))[0]
        camera_params = self._load_camera_params(camera_path)
        point_map_path = list(sequence_path.glob("*.tiff"))
        rgb_image_path = list(sequence_path.glob("*.png"))
        
        point_map_path.sort()
        rgb_image_path.sort()

        num_images = min(self.num_images, len(point_map_path))
        shuffled_indices = random.sample(range(len(point_map_path)), num_images)

        camera_keys = list(camera_params.keys())
        camera_keys.sort()
        camera_keys = [camera_keys[i] for i in shuffled_indices]


        point_map_path = [point_map_path[i] for i in shuffled_indices]
        rgb_image_path = [rgb_image_path[i] for i in shuffled_indices]

        point_maps = self._load_point_maps(point_map_path)
        rgb_images = self._load_rgb_image(rgb_image_path)
        camera_params = torch.stack([camera_params[k] for k in camera_keys])


        # Transform everything to use first camera as reference coordinate system

        transformed_cameras, transformed_point_maps = self._transform_to_first_camera_coordinates(camera_params, point_maps)
        if self.data_conversion == None:
            return transformed_point_maps, transformed_cameras, rgb_images
        else:
            print(transformed_cameras)
            return self.data_conversion(transformed_cameras, transformed_point_maps, rgb_images)


class PairedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        # Note: num_workers=0 for simplicity since we're dealing with dictionaries
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self._collate_fn)
    
    def _collate_fn(self, batch):
        """Custom collate function to handle dictionary returns"""
        if len(batch) == 1:
            # For batch_size=1, just return the single item
            return batch[0]
        else:
            # For larger batches, you'd need to implement proper batching logic
            # This is more complex with dictionaries, so keeping simple for now
            return batch    


def from_gt(transformed_point_maps, transformed_cameras, rgb_images):
    keys = transformed_point_maps.keys()
    keys = sorted(keys)
    point_maps = []
    cameras = []
    rgb_images_list = []
    for key in keys:
        point_maps.append(transformed_point_maps[key])
        cameras.append(transformed_cameras[key])
        rgb_images_list.append(rgb_images[key])

    point_maps = torch.stack(point_maps)
    cameras = torch.stack(cameras)
    rgb_images_list = torch.stack(rgb_images_list)
    gt = {
        "world_points": point_maps,
        "extrinsic": cameras,
        "images": rgb_images_list,
    }
    print("Processing model outputs...")
    for key in gt.keys():
        if isinstance(gt[key], torch.Tensor):
            gt[key] = gt[key].cuda()  # remove batch dimension and convert to numpy

    return gt


if __name__ == "__main__":
    dataset = PairedDataset(Path("/media/bxiong/c6deb427-f841-4fb3-8707-2d0593655c63/bingMap/train"), data_conversion=pytorch3d_to_colmap_batch)

    point_maps, camera_params, rgb_images = dataset[0]
    print(camera_params)
    preds = {
        "world_points": point_maps.cpu().numpy(),
        "extrinsic": camera_params.cpu().numpy(),
        "images": rgb_images.cpu().numpy(),
    }
    from visualizer import viser_wrapper
    viser_wrapper(preds)