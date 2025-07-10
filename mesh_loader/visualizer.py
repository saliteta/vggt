import time
import threading
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
from dataclasses import dataclass



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
    se3 = torch.from_numpy(se3) if is_numpy else se3

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R

    R_transposed = R.transpose(1, 2)  # (N,3,3)
    top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
    inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix



# After setting the visualizer Data, we will have a parser to parse the data into the visualizer
@dataclass
class VisualizerData:
    images: torch.Tensor # (B, 3, H, W)
    world_points: torch.Tensor # (B, H, W, 3)
    extrinsic: torch.Tensor # (B, 4, 4) or (B, 3, 4)
    intrinsic: torch.Tensor | None = None # (B, 3, 3)
    shape: torch.Size | None = None # (S, H, W, 3)



class Visualizer:
    def __init__(self, gt_data: VisualizerData, pred_data: VisualizerData, port: int = 8080, background_mode: bool = False):
        self.port = port
        self.background_mode = background_mode
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self.gt_data = self.set_data(gt_data)
        self.pred_data = self.set_data(pred_data)

        self.point_cloud = None
        self.frames: List[viser.FrameHandle] = []
        self.frustums: List[viser.CameraFrustumHandle] = []
        self.add_point_cloud()
        self.add_frames()
        self.set_gui()

    
    def add_point_cloud(self):
        colors = self.gt_data.images.cpu().numpy()
        points = self.gt_data.world_points.cpu().numpy()
        colors_pred = self.pred_data.images.cpu().numpy()
        points_pred = self.pred_data.world_points.cpu().numpy()

        self.gt_point_cloud = self.server.scene.add_point_cloud(
            name="gt_pcd",
            points=points,
            colors=colors,
            point_size=0.001,
        )
        self.pred_point_cloud = self.server.scene.add_point_cloud(
            name="pred_pcd",
            points=points_pred,
            colors=colors_pred,
            point_size=0.001,
        )

    def add_frames(self):
        extrinsics = torch.cat([self.gt_data.extrinsic, self.pred_data.extrinsic], dim=0)
        images = torch.cat([self.gt_data.images, self.pred_data.images], dim=0).reshape(-1, self.gt_data.shape[1], self.gt_data.shape[2], 3)
        self.visualize_frames(extrinsics.cpu().numpy(), images.cpu().numpy())

    def set_gui(self):
        gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        gui_gt_points = self.server.gui.add_checkbox("Show GT Points", initial_value=True)
        gui_pred_points = self.server.gui.add_checkbox("Show Pred Points", initial_value=True)

        @gui_gt_points.on_update
        def _(_):
            self.gt_point_cloud.visible = gui_gt_points.value
            self.pred_point_cloud.visible = gui_pred_points.value
            for i in range(self.gt_data.shape[0]):
                self.frames[i].visible = gui_gt_points.value and gui_show_frames.value
                self.frustums[i].visible = gui_gt_points.value and gui_show_frames.value
        
        @gui_pred_points.on_update
        def _(_):
            self.gt_point_cloud.visible = gui_gt_points.value
            self.pred_point_cloud.visible = gui_pred_points.value
            for i in range(self.gt_data.shape[0], self.gt_data.shape[0]*2):
                self.frames[i].visible = gui_pred_points.value and gui_show_frames.value
                self.frustums[i].visible = gui_pred_points.value and gui_show_frames.value
        
        @gui_show_frames.on_update
        def _(_):
            for f in self.frames:   
                f.visible = gui_show_frames.value
            for fr in self.frustums:
                fr.visible = gui_show_frames.value
    

    def set_data(self, data: VisualizerData):
        assert len(data.images.shape) == 4 or len(data.images.shape) == 5
        assert len(data.world_points.shape) == 4 or len(data.world_points.shape) == 5
        assert len(data.extrinsic.shape) == 3 or len(data.extrinsic.shape) == 4
        assert data.intrinsic is None or len(data.intrinsic.shape) == 3 or len(data.intrinsic.shape) == 4

        if len(data.images.shape) == 5:
            data.images = data.images.squeeze(0)
        if len(data.world_points.shape) == 5:
            data.world_points = data.world_points.squeeze(0)
        if len(data.extrinsic.shape) == 4:
            data.extrinsic = data.extrinsic.squeeze(0)

        if data.intrinsic is not None and len(data.intrinsic.shape) == 4:
            data.intrinsic = data.intrinsic.squeeze(0)

        data.shape = data.world_points.shape
        data.images = data.images.permute(0, 2, 3, 1)
        data.images = data.images.reshape(-1, 3)
        data.world_points = data.world_points.reshape(-1, 3)
        scene_center = torch.mean(data.world_points, dim=0)
        data.world_points = data.world_points - scene_center
        extriscis = closed_form_inverse_se3(data.extrinsic)
        data.extrinsic = extriscis[:, :3, :]
        data.extrinsic[..., -1] -= scene_center

        return data
    

    def visualize(self):
        pass


    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in self.frames:
            f.remove()
        self.frames.clear()
        for fr in self.frustums:
            fr.remove()
        self.frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(self.gt_data.shape[0]*2)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = self.server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            self.frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = self.server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            self.frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def run(self):
        print("Starting viser server...")
        # If background_mode is True, spawn a daemon thread so the main thread can continue.
        if self.background_mode:

            def server_loop():
                while True:
                    time.sleep(0.001)
            self.server_thread = threading.Thread(target=server_loop, daemon=True)
            self.server_thread.start()
        else:
            while True:
                time.sleep(0.01)


def viser_wrapper(
    gt_dict: Dict[str, torch.Tensor],
    pred_dict: Dict[str, torch.Tensor],
    port: int = 8080,
    background_mode: bool = False,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")


    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    if len(images.shape) == 5:
        images = images.squeeze(0)
    if len(world_points_map.shape) == 5:
        world_points_map = world_points_map.squeeze(0)
    if len(extrinsics_cam.shape) == 4:
        extrinsics_cam = extrinsics_cam.squeeze(0).cpu().numpy()



    # Compute world points from depth if not using the precomputed point map

    world_points:torch.Tensor = world_points_map

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.permute(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).to(torch.uint8)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    cam_to_world_mat = cam_to_world_mat.cpu().numpy()
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points.cpu().numpy(), axis=0)
    points_centered = points.cpu().numpy() - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)


    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered,
        colors=colors_flat.cpu().numpy(),
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""


        selected_idx = int(gui_frame_selector.value)
        frame_mask = frame_indices == selected_idx

        combined_mask = frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images.cpu().numpy())

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server





