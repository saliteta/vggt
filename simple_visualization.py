"""
Head-only finetuning script for VGGT.
Freeze the aggregator; train only camera, depth, and point heads.
"""
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from vggt.models.vggt import VGGT

from mesh_loader.simple_data_loader import PairedDataset
from mesh_loader.alignment import align_model_to_gt
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import datetime
import argparse
from pathlib import Path

import open3d as o3d
import numpy as np


def visualize_point_clouds_with_extrinsics(
    gt_pts: torch.Tensor,
    pred_pts: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    pred_extrinsics: torch.Tensor,
    gt_colors: torch.Tensor,
    pred_colors: torch.Tensor,
    frame_size: float = 0.1
):
    """
    gt_pts:         (N,3)   tensor of ground truth point coordinates
    pred_pts:       (N,3)   tensor of predicted point coordinates
    gt_extrinsics:  (B,4,4) tensor of ground truth extrinsic matrices (camera→world), B cameras
    pred_extrinsics: (B,4,4) tensor of predicted extrinsic matrices (camera→world), B cameras
    gt_colors:      (N,3)   tensor of RGB colors (0–1) for GT cloud
    pred_colors:    (N,3)   tensor of RGB colors (0–1) for predicted cloud
    frame_size:     float   size of the camera axes when drawn
    """
    # Move everything to NumPy / CPU
    gt_pts_np = gt_pts.cpu().numpy().reshape(-1,3)
    pred_pts_np = pred_pts.cpu().numpy().reshape(-1,3)
    pred_pts_np = pred_pts_np
    gt_colors_np = gt_colors.cpu().numpy()
    pred_colors_np = pred_colors.cpu().numpy()
    gt_extrs = gt_extrinsics.cpu().numpy()  # Shape (B,4,4)
    pred_extrs = pred_extrinsics.cpu().numpy()  # Shape (B,4,4)

    geometries = []
    
    # 1) Build & color the ground truth point cloud (green)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts_np)
    gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors_np)
    
    # 2) Build & color the predicted point cloud (red)
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_pts_np)
    pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors_np)
    
    # 3) Build camera frames for each camera
    for i in range(gt_extrs.shape[0]):
        # Ground truth camera frame (green)

        gt_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0, 0, 0])
        gt_cam_frame.paint_uniform_color([0, 1, 0])  # Green for GT
        gt_cam_frame.transform(gt_extrs[i])
        geometries.append(gt_cam_frame)
        
        # Predicted camera frame (red)
        pred_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0, 0, 0])
        pred_cam_frame.paint_uniform_color([1, 0, 0])  # Red for prediction
        pred_cam_frame.transform(pred_extrs[i])
        geometries.append(pred_cam_frame)

    # Add point clouds to geometries
    geometries = [gt_pcd, pred_pcd] + geometries

    # 4) visualize all together
    o3d.visualization.draw_geometries(
        geometries,
        window_name="GT vs Predicted",
        width=800,
        height=600,
        point_show_normal=False
    )


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="/media/bxiong/data/bingData/eval")
    parser.add_argument("--model_path", type=str, default="/media/bxiong/data/bing_model/best_model.pth")
    return parser.parse_args()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = arg_parse()
    dataset_dir = Path(args.eval_dir)


    # Dataset and DataLoader
    val_dataset = PairedDataset(dataset_dir,num_images=5)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    print(f"len(val_loader): {len(val_loader)}")
    # Model
    model = VGGT()
    model.load_state_dict(torch.load(args.model_path, map_location=device)["model_state_dict"])
    #URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    #model.load_state_dict(torch.hub.load_state_dict_from_url(URL, map_location=device))
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model.eval()
    model = model.to(device)



    with torch.no_grad():
        for point_maps, camera_params, rgb_images in tqdm(val_loader, desc=f"Evaluation"):
            gt = {
                "images": rgb_images.to(device),
                "extrinsic": camera_params.to(device).squeeze(0),
                "world_points": point_maps.to(device).squeeze(0)    
            }
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = model(gt["images"])
            predict_extrinsics, _ = pose_encoding_to_extri_intri(preds['pose_enc'], gt["images"].shape[-2:])
            preds["extrinsic"] = predict_extrinsics.squeeze(0)
            preds["world_points"] = preds["world_points"].squeeze(0)    
            #preds = align_model_to_gt(preds, gt)

            print(preds["extrinsic"][:,:3,:].shape)
            print(gt["extrinsic"][:,:3,:].shape)
            print(preds["world_points"].shape)
            print(gt["world_points"].shape)
            exit()

            # Create colors for visualization
            num_points = preds["world_points"].reshape(-1,3).shape[0]
            color = gt["images"].squeeze(0).permute(0,2,3,1).reshape(-1,3)
            gt_colors = torch.ones(num_points, 3).to(device) * torch.tensor([0.0, 1.0, 0.0]).to(device)# Green for GT
            pred_colors = torch.ones(num_points, 3).to(device) * torch.tensor([1.0, 0.0, 0.0]).to(device)  # Red for prediction
            gt_colors = 0.1*gt_colors+ color*0.9 # Green for GT
            pred_colors = 0.1*pred_colors+ color*0.9 # Red for prediction
            gt_colors = gt_colors.cpu()
            pred_colors = pred_colors.cpu()
            
            # Visualize the comparison
            visualize_point_clouds_with_extrinsics(
                gt_pts=gt["world_points"],
                pred_pts=preds["world_points"],
                gt_extrinsics=gt["extrinsic"],
                pred_extrinsics=preds["extrinsic"],
                gt_colors=gt_colors,
                pred_colors=pred_colors,
                frame_size=0.1
            )



if __name__ == '__main__':
    main()