#!/usr/bin/env python3
"""
Visualize the paired pose HDF5 (world_hand_joints and world_hand_joints_biased).
"""

import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import h5py
import torch
import viser
import time

from calibration_net.utils import transform_joint3d

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def _make_bone_segments(joints: np.ndarray) -> np.ndarray:
    """(21,3) -> (E,2,3) for viser line segments."""
    segs = []
    for a, b in MANO_CONNECTIONS:
        segs.append(np.stack([joints[a], joints[b]], axis=0))
    return np.stack(segs, axis=0)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize paired poses HDF5")
    parser.add_argument('hdf5_path', type=str, help='Path to paired poses HDF5 file (e.g., paired_poses.hdf5)')
    parser.add_argument('--port', type=int, default=8087, help='Viser server port')
    parser.add_argument('--fps', type=float, default=30.0, help='Playback FPS')
    parser.add_argument('--point_size', type=float, default=0.002, help='Point size for visualization')
    parser.add_argument('--line_width', type=float, default=1.0, help='Line width for visualization')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading HDF5 file: {args.hdf5_path}")
    data = {}
    with h5py.File(args.hdf5_path, 'r') as f:
        episodes = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
        for ep in episodes:
            grp = f[ep]
            
            # Extract world_hand_joints
            if 'world_hand_joints' in grp:
                jt = np.array(grp['world_hand_joints'])
                # Pre-transform
                jt_t = transform_joint3d(jt)
                if torch.is_tensor(jt_t):
                    jt = jt_t.cpu().numpy()
                else:
                    jt = np.array(jt_t)
            else:
                jt = None
                
            # Extract world_hand_joints_biased
            if 'world_hand_joints_biased' in grp:
                jb = np.array(grp['world_hand_joints_biased'])
                # Pre-transform
                jb_t = transform_joint3d(jb)
                if torch.is_tensor(jb_t):
                    jb = jb_t.cpu().numpy()
                else:
                    jb = np.array(jb_t)
            else:
                jb = None
                
            data[ep] = {
                'true': jt,
                'biased': jb
            }

    if not data:
        print("No episodes found in the HDF5 file.")
        return

    server = viser.ViserServer(port=args.port)
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.25, 0.0, 0.25)
        client.camera.look_at = (0.0, 0.0, 0.0)

    server.scene.add_grid("/grid", 1.0, 1.0)
    
    ep_slider = server.gui.add_slider("episode", min=0, max=max(0, len(episodes)-1), step=1, initial_value=0)
    
    # We will update frame_slider.max dynamically
    frame_slider = server.gui.add_slider("frame", min=0, max=100, step=1, initial_value=0)
    playing = server.gui.add_button("▶ Play/Pause")
    is_playing = {"value": False}
    
    info_markdown = server.gui.add_markdown("info", "")
    
    def render(ep_idx, frame_idx):
        ep = episodes[ep_idx]
        ep_data = data[ep]
        
        jt = ep_data['true']
        jb = ep_data['biased']
        
        # Determine number of frames
        T = 0
        if jt is not None:
            T = jt.shape[0]
        elif jb is not None:
            T = jb.shape[0]
            
        if T == 0:
            return
            
        if frame_idx >= T:
            frame_idx = T - 1
            frame_slider.value = frame_idx
            
        frame_slider.max = max(1, T - 1)
        
        with server.atomic():
            # True joints - Red
            if jt is not None:
                server.scene.add_point_cloud(
                    name="/true_joints",
                    points=jt[frame_idx],
                    colors=np.array([1.0, 0.0, 0.0]),
                    point_size=args.point_size,
                )
                server.scene.add_line_segments(
                    name="/true_bones",
                    points=_make_bone_segments(jt[frame_idx]),
                    colors=np.array([1.0, 0.0, 0.0]),
                    line_width=args.line_width,
                )
            else:
                # Remove if not exists
                server.scene.add_point_cloud(name="/true_joints", points=np.zeros((0,3)), colors=np.zeros((0,3)), point_size=args.point_size)
                server.scene.add_line_segments(name="/true_bones", points=np.zeros((0,2,3)), colors=np.zeros((0,3)), line_width=args.line_width)

            # Biased joints - Blue
            if jb is not None:
                server.scene.add_point_cloud(
                    name="/biased_joints",
                    points=jb[frame_idx],
                    colors=np.array([0.0, 0.0, 1.0]),
                    point_size=args.point_size,
                )
                server.scene.add_line_segments(
                    name="/biased_bones",
                    points=_make_bone_segments(jb[frame_idx]),
                    colors=np.array([0.0, 0.0, 1.0]),
                    line_width=args.line_width,
                )
            else:
                # Remove if not exists
                server.scene.add_point_cloud(name="/biased_joints", points=np.zeros((0,3)), colors=np.zeros((0,3)), point_size=args.point_size)
                server.scene.add_line_segments(name="/biased_bones", points=np.zeros((0,2,3)), colors=np.zeros((0,3)), line_width=args.line_width)

        info_markdown.content = f"**episode**: `{ep}`\n**frames**: `{T}`\n**current frame**: `{frame_idx}`"

    @ep_slider.on_update
    def _(_evt):
        if getattr(_evt, "client_id", None) is None:
            return
        frame_slider.value = 0
        render(int(ep_slider.value), 0)
        
    @frame_slider.on_update
    def _(_evt):
        if getattr(_evt, "client_id", None) is None:
            return
        render(int(ep_slider.value), int(frame_slider.value))

    @playing.on_click
    def _(_evt):
        is_playing["value"] = not is_playing["value"]

    # legend
    server.gui.add_markdown("help", "Legend:\n- **red**: world_hand_joints (True)\n- **blue**: world_hand_joints_biased (Biased)")

    # Initial render
    render(0, 0)
    print(f"Viser server running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    
    dt = 1.0 / max(1e-6, args.fps)
    try:
        while True:
            if is_playing["value"]:
                next_frame = int(frame_slider.value) + 1
                if next_frame > int(frame_slider.max):
                    next_frame = 0
                frame_slider.value = next_frame
                render(int(ep_slider.value), next_frame)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nShutting down viser server...")

if __name__ == '__main__':
    main()
