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
from calibration_net.utils import mano_recon, transform_joint3d, Side, scale_hand_fingers
from visualization.visualize_keypoint_recon import _make_bone_segments
from calibration_net.policy.refine_inferencer import RefineModelInferencer

# Mapping of finger names to joint indices in the 45-dim MANO theta array
# (0-indexed w.r.t the 15 finger joints)
FINGER_JOINT_INDICES = {
    "Index": [0, 1, 2],
    "Middle": [3, 4, 5],
    "Pinky": [6, 7, 8],
    "Ring": [9, 10, 11],
    "Thumb": [12, 13, 14]
}


@dataclass
class Args:
    input_dir: str
    output_dir: str
    model_path: str
    port: int = 8087
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
    model, norm_stats, cfg = load_model(args.model_path, device=device)
    model.eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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

    # Dictionary to hold user modifications
    # The shape of mano theta for fingers is (15, 3). This array matches that.
    manual_offsets = np.zeros((15, 3), dtype=np.float32)

    cache: dict[str, Any] = {
        "file_idx": None,
        "is_playing": False,
        "updating_sliders": False,
        "raw_data": None,
        "inferencer": None,
        "pred_theta": None, 
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

    finger_selector = server.gui.add_dropdown(
        "Select Finger",
        ("Index", "Middle", "Pinky", "Ring", "Thumb"),
        initial_value="Index"
    )

    sliders = []

    def load_file(file_idx: int) -> None:
        file_path = pkl_files[file_idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        cache["raw_data"] = data
        file_info.content = f"**File:** `{file_path.name}`"
        
        # Determine keypoint key
        keypoint_key = "world_hand_joints"
        if 'world_hand_joints_biased' in data:
            keypoint_key = 'world_hand_joints_biased'
            
        keypoint_traj = data[keypoint_key]
        if keypoint_traj.ndim == 4:
            keypoint_traj = keypoint_traj[0] # taking first sample if batched

        side = _get_str(data.get("mano_side", "left"))
        
        paired_poses = None
        has_preference_extractor = hasattr(model, 'preference_extractor') and model.preference_extractor is not None
        if has_preference_extractor:
            biased_key = 'world_hand_joints_biased'
            true_key_candidates = ['world_hand_joints', 'world_hand_joints_true']
            true_key = next((k for k in true_key_candidates if k in data.keys()), None)
            
            if biased_key in data.keys() and true_key is not None:
                biased_seq_raw = data[biased_key]
                if biased_seq_raw.ndim == 4:
                    biased_seq_raw = biased_seq_raw[0]
                    
                true_seq_raw = data[true_key]
                if true_seq_raw.ndim == 4:
                    true_seq_raw = true_seq_raw[0]
                    
                paired_poses = {
                    biased_key: biased_seq_raw,
                    true_key: true_seq_raw
                }

        # Initialize or update inferencer
        if cache.get("inferencer") is None:
            cache["inferencer"] = RefineModelInferencer(
                model_dir=args.model_path,
                side=side,
                device=device,
                paired_poses=paired_poses,
                fixed_noise=True,
                loaded_model=(model, norm_stats, cfg)
            )
            status = cache["inferencer"].canonicalizer_status()
            if status is not None:
                print(status)
        else:
            cache["inferencer"].set_side(side)
            cache["inferencer"].update_condition(paired_poses)
            
        pred_theta = cache["inferencer"].inference(keypoint_traj, batched=(keypoint_traj.ndim == 3))
        cache["pred_theta"] = pred_theta # shape should be (T, 45)
        
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
        
        reset_offsets_gui()

    def reset_offsets_gui():
        manual_offsets.fill(0.0)
        update_sliders_from_offsets()

    def update_sliders_from_offsets():
        cache['updating_sliders'] = True
        finger_name = finger_selector.value
        joint_indices = FINGER_JOINT_INDICES[finger_name]
        for i, j, s in sliders:
            joint_global_idx = joint_indices[i]
            s.value = float(manual_offsets[joint_global_idx, j])
        cache['updating_sliders'] = False

    def render_frame():
        if cache["pred_theta"] is None:
            return
            
        frame_idx = int(frame_slider.value)
        if frame_idx >= cache["T"]:
            frame_idx = cache["T"] - 1
            
        pred_theta = cache["pred_theta"][frame_idx]  # shape (45,)
        offset = manual_offsets.flatten()            # shape (45,)
        modified_theta = pred_theta + offset
        
        modified_theta_t = torch.tensor(modified_theta, dtype=torch.float32).unsqueeze(0).to(device)
        betas = cache["betas"][frame_idx:frame_idx+1].to(device)
        
        _, recon_joint3d = mano_recon(modified_theta_t, side=cache["side"], betas=betas)
        
        if hasattr(model.network, 'finger_scaling'):
            with torch.no_grad():
                recon_joint3d = scale_hand_fingers(recon_joint3d, model.network.finger_scaling)
                
        recon_joint3d = transform_joint3d(recon_joint3d).cpu().numpy()[0]
        
        keypoint_traj = cache["keypoint_traj"][frame_idx]
        
        with server.atomic():
            # Predicted joints + offset (blue)
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

    # Attach callbacks for sliders
    def create_slider_callback(joint_local_idx, axis_idx, slider):
        @slider.on_update
        def _(_) -> None:
            if cache['updating_sliders']: return
            finger_name = finger_selector.value
            joint_global_idx = FINGER_JOINT_INDICES[finger_name][joint_local_idx]
            manual_offsets[joint_global_idx, axis_idx] = slider.value
            cache["is_playing"] = False
            render_frame()

    for i in range(3):
        for j, axis in enumerate(["X", "Y", "Z"]):
            s = server.gui.add_slider(f"Joint {i} {axis}", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
            sliders.append((i, j, s))
            create_slider_callback(i, j, s)

    @finger_selector.on_update
    def _(_) -> None:
        update_sliders_from_offsets()

    reset_btn = server.gui.add_button("Reset Offsets")
    @reset_btn.on_click
    def _(_) -> None:
        reset_offsets_gui()
        render_frame()
        
    save_btn = server.gui.add_button("Save Data")
    @save_btn.on_click
    def _(_) -> None:
        if cache["pred_theta"] is None:
            return
        
        file_path = pkl_files[int(file_slider.value)]
        out_path = output_dir / file_path.name
        
        pred_theta = cache["pred_theta"] # shape (T, 45)
        offset = manual_offsets.flatten()
        modified_theta = pred_theta + offset
        
        data_copy = cache["raw_data"].copy()
        data_copy["mano_angles"] = modified_theta
        
        with open(out_path, 'wb') as f:
            pickle.dump(data_copy, f)
            
        print(f"Saved refined data to {out_path}")

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
Use **File Index** and **Frame** sliders to navigate.
Select a finger to edit its joints (Offsets apply to the entire trajectory).
**Blue**: predicted joints + offset (model output)
**Red**: input keypoints (ground truth)
Hit **Save Data** to save the modified MANO thetas.
""".strip(),
    )

    # Initial load
    load_file(int(file_slider.value))
    render_frame()

    # Main loop
    dt = 1.0 / max(1e-6, float(args.fps))
    print("Viser server started! Connect to http://localhost:" + str(args.port))
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
