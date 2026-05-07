import tyro
import torch
import viser
import numpy as np
import time
from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import Optional, Any
from calibration_net.policy.model import load_model
from calibration_net.utils import mano_recon, transform_joint3d, scale_hand_fingers
from visualization.visualize_keypoint_recon import _make_bone_segments

@dataclass
class Args:
    input_dir: str
    model_path: str
    port: int = 8088
    fps: float = 30.0
    show_grid: bool = True
    point_size: float = 0.002
    line_width: float = 1.0


def _get_str(item):
    if isinstance(item, list) or isinstance(item, np.ndarray):
        item = item[0]
    if isinstance(item, bytes):
        return item.decode('utf-8')
    return str(item)


def main(args: Args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = load_model(args.model_path, device=device)
    model.eval()

    input_dir = Path(args.input_dir)
    
    pkl_files = sorted(list(input_dir.glob("*.pkl")))
    if not pkl_files:
        print(f"No pkl files found in {input_dir}")
        return

    server = viser.ViserServer(port=args.port)
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.25, 0.0, 0.25)
        client.camera.look_at = (0.0, 0.0, 0.0)
    
    if args.show_grid:
        server.scene.add_grid("/grid", 1.0, 1.0)

    cache: dict[str, Any] = {
        "file_idx": None,
        "is_playing": False,
        "mano_angles": None, 
        "keypoint_traj": None,
        "betas": None,
        "T": 1,
        "side": 'left'
    }

    # UI Controls
    file_slider = server.gui.add_slider(
        "File Index",
        min=0,
        max=max(0, len(pkl_files) - 1),
        step=1,
        initial_value=0,
        disabled=(len(pkl_files) == 1),
    )
    
    file_info = server.gui.add_markdown("File: ", "")
    
    playing = server.gui.add_button("▶ Play/Pause")
    
    frame_slider = server.gui.add_slider(
        "Frame",
        min=0,
        max=1,
        step=1,
        initial_value=0,
    )

    def load_file(file_idx: int) -> None:
        file_path = pkl_files[file_idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        file_info.content = f"**File:** `{file_path.name}`"
        
        # Determine keypoint key
        keypoint_key = "world_hand_joints"
        if 'world_hand_joints_biased' in data:
            keypoint_key = 'world_hand_joints_biased'
            
        keypoint_traj = data[keypoint_key]
        if keypoint_traj.ndim == 4:
            keypoint_traj = keypoint_traj[0] # taking first sample if batched

        side = _get_str(data.get("mano_side", "left"))
        
        if "mano_angles" not in data:
            print(f"Warning: 'mano_angles' not found in {file_path}. Skipping.")
            return

        mano_angles = data["mano_angles"]
        if mano_angles.ndim == 3:  # (B, T, 45) -> take first sample
             mano_angles = mano_angles[0]
             
        cache["mano_angles"] = mano_angles
        cache["keypoint_traj"] = transform_joint3d(keypoint_traj)
        
        if "betas" in data:
            b = np.array(data["betas"])
            if b.ndim == 1:
                b = b[None, :].repeat(keypoint_traj.shape[0], axis=0)
            cache["betas"] = torch.tensor(b, dtype=torch.float32)
        else:
            cache["betas"] = torch.zeros((keypoint_traj.shape[0], 10), dtype=torch.float32)
            
        cache["side"] = side
        cache["T"] = keypoint_traj.shape[0]
        cache["file_idx"] = file_idx
        
        frame_slider.max = max(1, cache["T"] - 1)
        frame_slider.value = 0

    def render_frame():
        if cache["mano_angles"] is None:
            return
            
        frame_idx = int(frame_slider.value)
        if frame_idx >= cache["T"]:
            frame_idx = cache["T"] - 1
            
        mano_angles = cache["mano_angles"][frame_idx]  # shape (45,)
        
        mano_angles_t = torch.tensor(mano_angles, dtype=torch.float32).unsqueeze(0).to(device)
        betas = cache["betas"][frame_idx:frame_idx+1].to(device)
        
        _, recon_joint3d = mano_recon(mano_angles_t, side=cache["side"], betas=betas)
        
        if hasattr(model.network, 'finger_scaling'):
            with torch.no_grad():
                recon_joint3d = scale_hand_fingers(recon_joint3d, model.network.finger_scaling)
                
        recon_joint3d = transform_joint3d(recon_joint3d).cpu().numpy()[0]
        
        keypoint_traj = cache["keypoint_traj"][frame_idx]
        
        with server.atomic():
            # Predicted joints (blue)
            server.scene.add_point_cloud(
                name="/joints",
                points=recon_joint3d,
                colors=np.array([0.0, 0.0, 1.0]),
                point_size=float(args.point_size),
            )
            server.scene.add_line_segments(
                name="/bones",
                points=_make_bone_segments(recon_joint3d),
                colors=np.array([0.0, 0.0, 1.0]),
                line_width=float(args.line_width),
            )
            
            # Input keypoints (red)
            server.scene.add_point_cloud(
                name="/joints_input",
                points=keypoint_traj,
                colors=np.array([1.0, 0.0, 0.0]),
                point_size=float(args.point_size),
            )
            server.scene.add_line_segments(
                name="/bones_input",
                points=_make_bone_segments(keypoint_traj),
                colors=np.array([1.0, 0.0, 0.0]),
                line_width=float(args.line_width),
            )

    @file_slider.on_update
    def _(_evt) -> None:
        if getattr(_evt, "client_id", None) is None:
            return
        cache["is_playing"] = False
        load_file(int(file_slider.value))
        render_frame()

    @frame_slider.on_update
    def _(_evt) -> None:
        if getattr(_evt, "client_id", None) is None:
            return
        render_frame()

    @playing.on_click
    def _(_) -> None:
        cache["is_playing"] = not cache["is_playing"]

    server.gui.add_markdown(
        "help",
        """
Use **File Index** and **Frame** sliders to navigate the refined `.pkl` files.
**Blue**: joints generated using the saved `mano_angles`
**Red**: original input keypoints from `world_hand_joints`/`world_hand_joints_biased`
""".strip(),
    )

    # Initial load
    load_file(int(file_slider.value))
    render_frame()

    # Main loop
    dt = 1.0 / max(1e-6, float(args.fps))
    print("Viser viewer started! Connect to http://localhost:" + str(args.port))
    while True:
        if cache["is_playing"]:
            next_frame = int(frame_slider.value) + 1
            if next_frame >= cache["T"]:
                next_frame = 0
            frame_slider.value = next_frame
            render_frame()
        time.sleep(dt)


if __name__ == "__main__":
    main(tyro.cli(Args))
