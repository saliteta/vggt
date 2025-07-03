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


# Logging imports
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/home/bxiong/workspace/tile_mesh_rendering/camera/render_0000")
    return parser.parse_args()


def main():
    global_step = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    args = arg_parse()

    if TENSORBOARD_AVAILABLE:
        tb_logdir = os.path.join("runs", "vggt_head")
        writer = SummaryWriter(log_dir=tb_logdir)
        print(f"TensorBoard logging to {tb_logdir}")

    if WANDB_AVAILABLE:
        wandb.init(project="vggt_head", name="vggt_head")
        print("Wandb logging to wandb")

    # Dataset and DataLoader
    dataset = PairedDataset(args.dataset_dir)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True,
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

    optimizer = optim.Adam(head_params, lr=4e-5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(1, 100 + 1):
        model.train()
        running_loss = 0.0
        for point_maps, camera_params, rgb_images in tqdm(loader, desc=f"Epoch {epoch}"):
            keys = list(point_maps.keys())
            keys.sort()
            images = []
            gt_extrinsics = []
            gt_points = []
            for key in keys:
                images.append(rgb_images[key])
                gt_extrinsics.append(camera_params[key])
                gt_points.append(point_maps[key])
            gt = {
                "images": torch.stack(images).to(device).squeeze(1),
                "extrinsic": torch.stack(gt_extrinsics).to(device).squeeze(1),
                "world_points": torch.stack(gt_points).to(device).squeeze(1)
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
            
            if TENSORBOARD_AVAILABLE:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
            if WANDB_AVAILABLE:
                wandb.log({"train/loss_step": loss.item(), "step": global_step})
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch} loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss}, "model_save/best_model.pth")
            print(f"Saved best model to best_model.pth")
        # --- log per-epoch ---
        print(f"Epoch {epoch} loss: {epoch_loss:.6f}")
        if TENSORBOARD_AVAILABLE:
            writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
        if WANDB_AVAILABLE:
            wandb.log({"train/loss_epoch": epoch_loss, "epoch": epoch})

        # --- save best ---
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs("model_save", exist_ok=True)
            ckpt_path = "model_save/best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")
            if WANDB_AVAILABLE:
                wandb.save(ckpt_path)

    # ----------------------------------------------------------------
    # 5) Wrap up
    # ----------------------------------------------------------------
    if TENSORBOARD_AVAILABLE:
        writer.close()
    if WANDB_AVAILABLE:
        wandb.finish()
    print("Training complete.")


if __name__ == '__main__':
    main()