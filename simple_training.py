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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/media/bxiong/c6deb427-f841-4fb3-8707-2d0593655c63/bingMap")
    parser.add_argument("--sequence_length", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    return parser.parse_args()


def main():
    global_step = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    args = arg_parse()
    dataset_dir = Path(args.dataset_dir)

    if WANDB_AVAILABLE:
        wandb.init(project="vggt_head", name="vggt_head")
        print("Wandb logging to wandb")

    # Dataset and DataLoader
    train_dataset = PairedDataset(dataset_dir/"train")
    val_dataset = PairedDataset(dataset_dir/"eval")
    train_loader = DataLoader(train_dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)


    # Model
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    


    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16




    # Load camera poses
    # Freeze aggregator
    for p in model.aggregator.parameters():
        p.requires_grad = False
    model = model.to(device)

    # Only head params
    head_params = []
    head_params += list(model.camera_head.parameters())
    head_params += list(model.point_head.parameters())

    optimizer = optim.Adam(head_params, lr=args.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(1, 100 + 1):
        running_loss = 0.0
        model.train()
        for point_maps, camera_params, rgb_images in tqdm(train_loader, desc=f"Epoch {epoch}"):
            gt = {
                "images": rgb_images.to(device),
                "extrinsic": camera_params.to(device).squeeze(0),
                "world_points": point_maps.to(device).squeeze(0)
            }


            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = model(gt["images"])
            predict_extrinsics, _ = pose_encoding_to_extri_intri(preds['pose_enc'], gt["images"].shape[-2:])
            preds["extrinsic"] = predict_extrinsics.squeeze(0)
            preds["world_points"] = preds["world_points"].squeeze(0)
            preds = align_model_to_gt(preds, gt)
            loss_pose = criterion(preds["extrinsic"], gt["extrinsic"])
            loss_pts  = criterion(preds["world_points"], gt["world_points"].reshape(-1, 3))
            loss = loss_pose + loss_pts
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * gt["images"].size(0)
            global_step += 1
            

            if WANDB_AVAILABLE:
                wandb.log({"train/loss_step": loss.item(), "step": global_step})
        train_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch} train loss: {train_loss:.6f}")
        
        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            for point_maps, camera_params, rgb_images in tqdm(val_loader, desc=f"Epoch {epoch}"):
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
                preds = align_model_to_gt(preds, gt)
                loss_pose = criterion(preds["extrinsic"], gt["extrinsic"])
                loss_pts  = criterion(preds["world_points"], gt["world_points"].reshape(-1, 3))
                loss = loss_pose + loss_pts
                running_loss += loss.item() * gt["images"].size(0)
            val_loss = running_loss / len(val_dataset)
            print(f"Epoch {epoch} val loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss}, "model_save/best_model.pth")
            print(f"Saved best model to best_model.pth")
        # --- log per-epoch ---
        print(f"Epoch {epoch} train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")

        if WANDB_AVAILABLE:
            wandb.log({"train/loss_epoch": train_loss, "epoch": epoch})
            wandb.log({"val/loss_epoch": val_loss, "epoch": epoch})

        # --- save best ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("model_save", exist_ok=True)
            ckpt_path = "model_save/best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")
            if WANDB_AVAILABLE:
                wandb.save(ckpt_path)

    # ----------------------------------------------------------------
    # 5) Wrap up
    # ----------------------------------------------------------------
    if WANDB_AVAILABLE:
        wandb.finish()
    print("Training complete.")


if __name__ == '__main__':
    main()