import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import glob
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from simple_data_loader import PairedDataset
from plyfile import PlyData, PlyElement
from alignment import align_model_to_gt

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def from_model(image_folder, model_path = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    if model_path is None:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    else:
        model.load_state_dict(torch.load(model_path)["model_state_dict"])

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {image_folder}...")
    image_names = glob.glob(os.path.join(image_folder, "*"))
    image_names = [os.path.join(image_folder, image_name) for image_name in image_names if image_name.endswith(".png")]
    image_names.sort()
    image_names = image_names[:3]
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)

    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].squeeze(0)  # remove batch dimension and convert to numpy

    return predictions

def from_gt(gt_folder):
    datasets = PairedDataset(gt_folder)
    transformed_point_maps, transformed_cameras, rgb_images  = datasets[0]
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
    predictions = {
        "world_points": point_maps[:3],
        "extrinsic": cameras[:3],
        "images": rgb_images_list[:3],
    }
    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cuda()  # remove batch dimension and convert to numpy

    return predictions

def umeyama_similarity(P:torch.Tensor, Q:torch.Tensor):
    """
    Compute best‐fit rotation R, scale s, and translation t
    such that:  Q ≈ s R P + t
    P, Q: (N,3) arrays with known correspondence.
    Returns R (3×3), s (scalar), t (3,).
    """
    assert P.shape == Q.shape
    N = P.shape[0]

    muP = P.mean(axis=0)
    muQ = Q.mean(axis=0)
    P0  = P - muP
    Q0  = Q - muQ

    C = (Q0.T @ P0) / N
    U, Svals, Vt = torch.linalg.svd(C)
    S = torch.eye(3).cuda()
    if torch.linalg.det(U) * torch.linalg.det(Vt) < 0:
        S[2,2] = -1
    R = U @ S @ Vt

    scale = (Svals * torch.diag(S)).sum() / torch.sum(P0**2)
    t     = muQ - scale * (R @ muP)

    return R, scale, t



def save_ply(points, colors, filename):
    points = points.cpu().numpy()
    colors = colors.cpu().numpy()
    colors = colors.reshape(-1, 3).astype('f4')
    colors = colors * 255
    colors = colors.astype('u1')

    vertex_dtype = [
        ('x',    'f4'),
        ('y',    'f4'),
        ('z',    'f4'),
        ('red',   'u1'),
        ('green', 'u1'),
        ('blue',  'u1'),
    ]

    vertex_data = np.empty(points.shape[0], dtype=vertex_dtype)
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    vertex_data['red']   = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue']  = colors[:, 2]

    ply_el = PlyElement.describe(vertex_data, name='vertex')
    ply_data = PlyData([ply_el], text=False)  # text=False → binary
    ply_data.write(filename)

    print(f"Wrote binary PLY with positions+colors to {filename}")



if __name__ == "__main__":

    image_folder = "/home/bxiong/workspace/tile_mesh_rendering/camera/render_0000"
    gt = from_gt("/home/bxiong/workspace/tile_mesh_rendering/camera/")
    predictions = from_model(image_folder, model_path="/home/bxiong/workspace/vggt/model_save/best_model.pth")
    #predictions = from_model(image_folder)

    predictions = align_model_to_gt(predictions, gt)

    print(gt["world_points"].shape, predictions["world_points"].shape)
    print(gt["images"].shape, predictions["images"].shape)
    save_ply(gt["world_points"].reshape(-1, 3), gt["images"].permute(0, 2, 3, 1).reshape(-1, 3), "gt.ply")
    save_ply(predictions["world_points"].reshape(-1, 3), predictions["images"].permute(0, 2, 3, 1).reshape(-1, 3), "predictions.ply")
    




