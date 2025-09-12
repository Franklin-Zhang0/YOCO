import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from manopth.manolayer import ManoLayer
import zmq
import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import threading
import time
import multiprocessing

# ----------------------------
# Utilities
# ----------------------------
MEDIAPIPE_21_ORDER = list(range(21))  # MP order = [0..20]; MANO-21 commonly uses the same order.
FINGERTIP_IDXS = [4, 8, 12, 16, 20]   # MP fingertip indices
KEEP16_FROM_21 = [i for i in range(21) if i not in FINGERTIP_IDXS]  # drop tips to match MANO-16

WEBXR_TO_SMPLX = [
     0,   # wrist                                          → 0
     1, 2, 3, 4,                                           # thumb_1-4
     6, 7, 8, 9,                                           # index_1-4  (skip 5 = metacarpal)
    11,12,13,14,                                           # middle_1-4 (skip 10)
    16,17,18,19,                                           # ring_1-4   (skip 15)
    21,22,23,24                                            # pinky_1-4  (skip 20)
]

JOINT_WEIGHT = [
    1.0,
    1.0, 1.0, 1.0, 3.0,
    1.0, 1.0, 1.0, 3.0,
    1.0, 1.0, 1.0, 3.0,
    1.0, 1.0, 1.0, 3.0,
    1.0, 1.0, 1.0, 3.0,
]

def webxr_to_smplx(joints_xyz: np.ndarray) -> np.ndarray:
    """
    Convert a (25, 3) WebXR-ordered hand to the (21, 3) MANO ordering.

    Parameters
    ----------
    joints_xyz : np.ndarray
        Hand joint coordinates in WebXR order, shape (25, 3).

    Returns
    -------
    np.ndarray
        Same coordinates reordered for MANO, shape (21, 3).
    """
    if joints_xyz.shape[0] != 25 or joints_xyz.shape[1] != 3:
        raise ValueError("Expected (25,3) array in WebXR joint order.")
    return joints_xyz[WEBXR_TO_SMPLX]

def robust_l1(x, delta=0.01):
    # Huber loss with small delta (nearly L1 but smooth near 0)
    absx = torch.abs(x)
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device))
    lin = absx - quad
    return 0.5 * quad**2 / delta + lin

def to_torch(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class PoseCollector():
    def __init__(self, frame_rate=30):
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE,   1) 
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.connect("tcp://localhost:5555")
        self.frame_rate=frame_rate

    def collect(self, timesteps:int=1, sides:list[str]=['left','right']):
        assert sides in [['left'], ['right'], ['left', 'right']]
        hand_pose={
            'left':[],
            'right':[]
        }
        while True:
            stop_flag = True
            for side in sides:
                if len(hand_pose[side]) < timesteps:
                    stop_flag = False
            if stop_flag:
                break

            msg = self._sub.recv_json()
            obj = msg['data']
            sides = ['left', 'right']
            for side in sides:
                matrix = obj[side]
                if not isinstance(matrix, list):
                    print(f"Invalid joint data for {side} hand, skipping")
                    continue
                joints = np.asarray(matrix, dtype=np.float32) 
                if joints.shape != (400,):
                    print(f"Bad joint array shape for {side} hand, skipping")
                    continue
                joints = joints.reshape(25, 4, 4)
                joints_xyz = joints[:, 3, :3]  # Extract XYZ positions from the 4x4 matrices
                joints_xyz = webxr_to_smplx(joints_xyz)  # Reorder to MANO format
                hand_pose[side].append(joints_xyz)
            print(f"Number of frames collected, left:{len(hand_pose['left'])}/{timesteps}, right:{len(hand_pose['right'])}/{timesteps}", end='\r')

        for side in sides:
            hand_pose[side] = np.stack(hand_pose[side], axis=0)
        return hand_pose



# ----------------------------
# Fitter
# ----------------------------
class ManoSequenceFitter:
    def __init__(
        self,
        mano_model_dir="models/mano",
        mano_side="right",              # "left" or "right"
        use_pca=False,                  # If True, fit PCA pose with ncomps
        ncomps=15,                      # PCA components if use_pca=True
        lr=0.02,
        iters=300,
        w_joint=1.0,
        w_pose_prior=0.001,
        w_shape_prior=0.1,
        w_trans_smooth=0.0,            # e.g., 0.1 to encourage smooth translation over time
        w_pose_smooth=0.0,             # e.g., 0.01 to encourage smooth pose over time
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.use_pca = use_pca
        self.ncomps = ncomps
        self.iters = iters
        self.w_joint = w_joint
        self.w_pose_prior = w_pose_prior
        self.w_shape_prior = w_shape_prior
        self.w_trans_smooth = w_trans_smooth
        self.w_pose_smooth = w_pose_smooth
        self.lr = lr

        # MANO layer
        self.mano_layer = ManoLayer(
            mano_root=mano_model_dir,
            side=mano_side,
            use_pca=use_pca,
            ncomps=ncomps if use_pca else 45,  # ncomps is ignored when use_pca=False
            flat_hand_mean=False,               # better prior
        ).to(device)

        # Freeze shape by default; you can set requires_grad=True to also fit betas
        self.betas = nn.Parameter(torch.zeros(10, device=device), requires_grad=True)


    def fit_sequence(self, joints3d_world, optimize_scale=False):
        """
        Args:
          joints3d_world: np.ndarray (T, 21, 3) target joints in a common 3D coord system.
                          If you only care per-frame, any rigid frame is fine.
        Returns dict with per-frame parameters:
          - global_orient: (T,3), hand_pose: (T,45) or (T,ncomps)
          - transl: (T,3), betas: (10,) (shared)
        """
        T = joints3d_world.shape[0]
        target_21 = to_torch(joints3d_world, self.device)  # (T,21,3)

        # Center targets at wrist to reduce translation mismatch (still fit transl)
        target_root = target_21[:, 0:1, :]  # (T,1,3) wrist
        target_centered = target_21 - target_root

        target_used = target_centered  # (T,21,3)

        # Per-frame parameters
        if self.use_pca:
            hand_pose = nn.Parameter(torch.zeros(T, self.ncomps, device=self.device))
        else:
            hand_pose = nn.Parameter(torch.zeros(T, 45, device=self.device))  # 15*3 axis-angle

        global_orient = nn.Parameter(torch.zeros(T, 3, device=self.device))  # axis-angle
        transl = nn.Parameter(torch.zeros(T, 3, device=self.device))
        scale = nn.Parameter(torch.ones(1, device=self.device), requires_grad=optimize_scale)

        params = [hand_pose, global_orient, transl]
        if optimize_scale:
            params.append(scale)
        if self.betas.requires_grad:
            params.append(self.betas)

        opt = Adam(params, lr=self.lr)

        tqdm_bar= tqdm.tqdm(range(self.iters), desc="Fitting")
        for it in tqdm_bar:
            opt.zero_grad()

            # MANO forward per frame
            concatenated_pose = torch.cat([global_orient, hand_pose], dim=1) 
            if self.use_pca:
                out = self.mano_layer(
                    th_betas=self.betas.unsqueeze(0).expand(T, -1),
                    th_pose_coeffs=concatenated_pose,
                    th_trans=transl
                )
            else:
                # Concatenate 3 (global_orient) + 45 (hand_pose) if using axis-angle API variant
                out = self.mano_layer(
                    th_pose_coeffs=concatenated_pose,            # 45D axis-angle (15 joints*3)
                    th_betas=self.betas.unsqueeze(0).expand(T, -1),
                    th_trans=transl
                )

            verts, J_mano = out  # (T, n_verts, 3)

            # Center MANO joints at wrist too for a fair comparison
            J_centered = (J_mano - J_mano[:, 0:1, :])  # convert mm to M

            # Optional per-frame scale (to absorb size mismatch if betas are fixed)
            if optimize_scale:
                J_centered = J_centered * (scale**2).view(1, 1, 1).expand(T, -1, -1)

            # Joint loss
            joint_residual = J_centered - target_used * 1000.0  # convert M to mm
            loss_j = robust_l1(joint_residual).mean()

            # Small pose prior (keeps fingers near the MANO mean pose)
            loss_prior = (hand_pose**2).mean() 
            loss_shape = (self.betas**2).mean()

            # Smoothness (optional)
            loss_smooth = torch.tensor(0.0, device=self.device)
            if T > 1:
                if self.w_trans_smooth > 0:
                    loss_smooth = loss_smooth + (transl[1:] - transl[:-1]).pow(2).mean()
                if self.w_pose_smooth > 0:
                    loss_smooth = loss_smooth + (hand_pose[1:] - hand_pose[:-1]).pow(2).mean()

            loss = self.w_joint * loss_j + self.w_pose_prior * loss_prior + self.w_shape_prior * loss_shape \
                   + self.w_trans_smooth * loss_smooth + self.w_pose_smooth * 0.0  # already included above

            loss.backward()
            opt.step()
            tqdm_bar.set_description(f"loss {loss_j.item():.4f}, {loss_prior.item():.4f}, {loss_shape.item():.4f}")


        with torch.no_grad():
            result = {
                "global_orient": global_orient.detach().cpu().numpy(),  # (T,3) axis-angle
                "hand_pose": hand_pose.detach().cpu().numpy(),          # (T,45) or (T,ncomps)
                "transl": transl.detach().cpu().numpy(),                # (T,3)
                "betas": self.betas.detach().cpu().numpy(),             # (10,)
                "scale": scale.detach().cpu().numpy() if optimize_scale else None
            }
        return result
    
    def initialize_single_frame_fitting(self):
        self.mano_pose = nn.Parameter(torch.zeros(self.ncomps if self.use_pca else 45, device=self.device, dtype=torch.float32))
        self.global_orient = nn.Parameter(torch.zeros(3, device=self.device))
        self.transl = nn.Parameter(torch.zeros(3, device=self.device))
        self.optim = torch.optim.Adam([self.mano_pose, self.global_orient, self.transl], lr=0.1)

    def fit_single_frame(self, joints3d, prev_result=None, betas=None, max_steps=20):
        if betas is None:
            betas = torch.zeros(10, device=self.device, dtype=torch.float32)
        joints3d = to_torch(joints3d, self.device)  # (21,3)
        joints3d = joints3d - joints3d[0:1]

        def closure():
            self.optim.zero_grad()
            pose = torch.cat([self.global_orient, self.mano_pose])
            vert, J_mano = self.mano_layer(
                th_betas=betas.unsqueeze(0),
                th_pose_coeffs=pose.unsqueeze(0),
                th_trans=self.transl.unsqueeze(0),
            )
            J_mano = J_mano - J_mano[:, 0:1, :]  # Center at wrist
            J_mano = J_mano.squeeze()
            pose_prior = (self.mano_pose**2).mean()
            loss = robust_l1(J_mano[1:]-joints3d[1:] * 1000.0).mean() + self.w_pose_prior * pose_prior
            if prev_result is not None: # add smoothness loss
                smooth_loss = ((self.mano_pose - prev_result['hand_pose'])**2).mean() + ((self.global_orient - prev_result['global_orient'])**2).mean() + ((self.transl - prev_result['transl'])**2).mean()
                loss += smooth_loss
            loss.backward()
            return loss
        
        for i in range(max_steps):
            self.optim.zero_grad()
            loss = closure()
            if loss < 3: break
            self.optim.step()

        result = {
            "global_orient": self.global_orient.detach(),  # (T,3) axis-angle
            "hand_pose": self.mano_pose.detach(),          # (T,45) or (T,ncomps)
            "transl": self.transl.detach(),                # (T,3)
            "betas": betas.detach(),             # (10,)
            "loss": loss.detach()
        }
        return result


# MANO-21 (SMPL-X hand) kinematic edges in the order you used (wrist=0, then thumb/index/middle/ring/pinky 1..4)
EDGES_21 = [
    # Thumb (0-1-2-3-4)
    (0,1), (1,2), (2,3), (3,4),
    # Index (0-5-6-7-8)
    (0,5), (5,6), (6,7), (7,8),
    # Middle (0-9-10-11-12)
    (0,9), (9,10), (10,11), (11,12),
    # Ring (0,13,14,15,16)
    (0,13), (13,14), (14,15), (15,16),
    # Pinky (0,17,18,19,20)
    (0,17), (17,18), (18,19), (19,20),
]

def _axis_equal_3d(ax, xyz):
    """
    Make 3D axes have equal scale based on a (N,3) array of points.
    """
    if xyz.size == 0:
        return
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    ranges = maxs - mins
    max_range = ranges.max()
    if max_range == 0:
        max_range = 1.0
    centers = (maxs + mins) / 2.0
    for i, c in enumerate(centers):
        ax.set_xlim(centers[0] - max_range/2, centers[0] + max_range/2)
        ax.set_ylim(centers[1] - max_range/2, centers[1] + max_range/2)
        ax.set_zlim(centers[2] - max_range/2, centers[2] + max_range/2)

def reconstruct_mano_sequence(out_dict, mano_layer, use_pca=False, device=torch.device("cpu")):
    """
    Re-run MANO forward in batch to get verts and joints for all frames.

    Parameters
    ----------
    out_dict : dict
        Result from fitter.fit_sequence(...).
        Keys: "global_orient"(T,3), "hand_pose"(T,45 or T,ncomps), "transl"(T,3), "betas"(10,)
    mano_layer : manopth.manolayer.ManoLayer
        The same layer (same side, PCA setting) used in fitting.
    use_pca : bool
        Must match the layer setup.

    Returns
    -------
    verts_np : (T, V, 3) float32, in millimeters
    joints_np: (T, 21, 3) float32, in millimeters
    """
    global_orient = torch.from_numpy(out_dict["global_orient"]).float().to(device)  # (T,3)
    hand_pose     = torch.from_numpy(out_dict["hand_pose"]).float().to(device)      # (T,45) or (T,ncomps)
    transl        = torch.from_numpy(out_dict["transl"]).float().to(device)         # (T,3)
    betas         = torch.from_numpy(out_dict["betas"]).float().to(device)          # (10,)
    T = global_orient.shape[0]

    concat = torch.cat([global_orient, hand_pose], dim=1)  # (T, 3+pose_dim)

    out = mano_layer(
        th_betas=betas.unsqueeze(0).expand(T, -1),
        th_pose_coeffs=concat,
        th_trans=transl
    )  # returns (verts_mm, joints_mm)
    verts_mm, joints_mm = out
    verts_np  = verts_mm.detach().cpu().numpy()
    joints_np = joints_mm.detach().cpu().numpy()
    return verts_np, joints_np

def animate_hand_sequence(
    joints_seq_mano_mm,
    edges=EDGES_21,
    target_seq_m=None,
    title="MANO Fit",
    fps=30,
    save_path=None,  # e.g., "mano_fit.mp4" or "mano_fit.gif"
):
    """
    Animate a MANO joint sequence (mm). Optionally overlay target joints (meters).

    joints_seq_mano_mm : (T,21,3) in millimeters
    target_seq_m       : (T,21,3) in meters (optional)
    """
    # Convert MANO mm → meters for shared scale if overlaying target (which is in meters)
    joints_seq_m = joints_seq_mano_mm / 1000.0
    T = joints_seq_m.shape[0]

    # Precompute scene bounds for consistent view
    all_pts = joints_seq_m.reshape(-1,3)
    if target_seq_m is not None:
        all_pts = np.vstack([all_pts, target_seq_m.reshape(-1,3)])

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")

    # Init scatters and line segments
    pred_scatter = ax.scatter([], [], [], s=15)
    bone_lines = [ax.plot([], [], [], linewidth=2)[0] for _ in edges]

    tgt_scatter = None
    tgt_bone_lines = None
    if target_seq_m is not None:
        tgt_scatter = ax.scatter([], [], [], s=12, marker="^")
        tgt_bone_lines = [ax.plot([], [], [], linewidth=1, linestyle="--")[0] for _ in edges]

    _axis_equal_3d(ax, all_pts)

    def draw_skeleton(joints_xyz, scatter_handle, line_handles):
        scatter_handle._offsets3d = (joints_xyz[:,0], joints_xyz[:,1], joints_xyz[:,2])
        for (i,(a,b)) in enumerate(edges):
            seg = np.stack([joints_xyz[a], joints_xyz[b]], axis=0)
            line_handles[i].set_data(seg[:,0], seg[:,1])
            line_handles[i].set_3d_properties(seg[:,2])

    def init():
        # Frame 0
        draw_skeleton(joints_seq_m[0], pred_scatter, bone_lines)
        if target_seq_m is not None:
            draw_skeleton(target_seq_m[0], tgt_scatter, tgt_bone_lines)
        return [pred_scatter, *bone_lines] + (([tgt_scatter] + tgt_bone_lines) if target_seq_m is not None else [])

    def update(frame):
        draw_skeleton(joints_seq_m[frame], pred_scatter, bone_lines)
        if target_seq_m is not None:
            draw_skeleton(target_seq_m[frame], tgt_scatter, tgt_bone_lines)
        ax.set_title(f"{title} | frame {frame+1}/{T}")
        return [pred_scatter, *bone_lines] + (([tgt_scatter] + tgt_bone_lines) if target_seq_m is not None else [])

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, interval=1000.0/max(fps,1), blit=False)

    if save_path is not None:
        if save_path.lower().endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            # Requires FFMPEG installed
            ani.save(save_path, writer=animation.FFMpegWriter(fps=fps))
        print(f"Saved animation to {save_path}")

    plt.show()
    plt.close(fig)

def _axis_equal_3d(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    rng  = (maxs - mins).max()
    if rng == 0: rng = 1.0
    ctr  = (maxs + mins) / 2.0
    ax.set_xlim(ctr[0]-rng/2, ctr[0]+rng/2)
    ax.set_ylim(ctr[1]-rng/2, ctr[1]+rng/2)
    ax.set_zlim(ctr[2]-rng/2, ctr[2]+rng/2)

def animate_mano_mesh(
    verts_seq_mm,              # (T, V=778, 3) MANO verts in millimeters
    faces,                     # (F, 3) triangle indices (np.int32/64)
    title="MANO Mesh",
    fps=30,
    save_path=None,            # e.g. "mano_mesh.mp4" or ".gif"
    show_wireframe=True
):
    """
    Simple matplotlib animation of MANO mesh over time.
    """
    # Convert to meters so it matches your target-joint units if you overlay later
    verts_seq = verts_seq_mm / 1000.0
    T, V, _ = verts_seq.shape
    F = faces.shape[0]

    # Precompute scene bounds
    all_pts = verts_seq.reshape(-1, 3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    _axis_equal_3d(ax, all_pts)

    # Initial mesh
    tris0 = verts_seq[0][faces]   # (F, 3, 3)
    mesh_collection = Poly3DCollection(
        tris0,
        alpha=0.7,
        edgecolor='k' if show_wireframe else 'none',
        linewidths=0.2
    )
    mesh_collection.set_facecolor((0.75, 0.75, 0.85, 1.0))  # soft gray-blue
    ax.add_collection3d(mesh_collection)

    # Optional: show root/wrist point to inspect drift
    root_scatter = ax.scatter([verts_seq[0,0,0]],[verts_seq[0,0,1]],[verts_seq[0,0,2]], s=15)

    def init():
        mesh_collection.set_verts(verts_seq[0][faces])
        root_scatter._offsets3d = ( [verts_seq[0,0,0]],
                                    [verts_seq[0,0,1]],
                                    [verts_seq[0,0,2]] )
        ax.set_title(f"{title} | frame 1/{T}")
        return [mesh_collection, root_scatter]

    def update(f):
        tris = verts_seq[f][faces]            # (F, 3, 3)
        mesh_collection.set_verts(tris)
        root_scatter._offsets3d = ( [verts_seq[f,0,0]],
                                    [verts_seq[f,0,1]],
                                    [verts_seq[f,0,2]] )
        ax.set_title(f"{title} | frame {f+1}/{T}")
        return [mesh_collection, root_scatter]

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=T, interval=1000.0/max(1, fps), blit=False
    )

    if save_path is not None:
        if save_path.lower().endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            ani.save(save_path, writer=animation.FFMpegWriter(fps=fps))
        print(f"Saved mesh animation to {save_path}")

    plt.show()
    plt.close(fig)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    pose_collector = PoseCollector()
    pose = pose_collector.collect(1000)
    result = {}
    for side in ['left', 'right']:
        fitter = ManoSequenceFitter(
            mano_model_dir="/home/franklin/project/hand_teleop/manopth/mano/models",           # <-- change to your MANO folder
            mano_side=side,
            use_pca=True,                          # set True if you want PCA pose
            ncomps=21,
            iters=200,
            w_joint=1.0,
            w_pose_prior=0.001,
            w_shape_prior=0.01,
            w_trans_smooth=0.1,                    # try >0 for sequences
            w_pose_smooth=0.01
        )

        out = fitter.fit_sequence(pose[side], optimize_scale=False)
        result[side] = out
        prev_result = None

        # fitter.initialize_single_frame_fitting()
        
        # try:
        #     tqdm_bar = tqdm.tqdm(range(10000))
        #     for i in tqdm_bar:
        #         single_pose = pose_collector.collect(1)[side][0]
        #         single_out = fitter.fit_single_frame(single_pose,  betas=to_torch(out["betas"], fitter.device), prev_result=prev_result)
        #         prev_result = single_out
        #         print(prev_result['hand_pose'][:5])
        #         # tqdm_bar.set_description(f"Single-frame loss {single_out['loss'].item():.4f}")
        # except:
        #     pass
            # print(prev_result)
        # except:
        #     pass
        print('betas:', out["betas"])
        print("scale:", None if out["scale"] is None else out["scale"]**2)
        # verts_mm, joints_mm = reconstruct_mano_sequence(out, fitter.mano_layer, use_pca=fitter.use_pca, device=fitter.device)
        # animate_hand_sequence(joints_mm, title=f"MANO Fit ({side})", fps=60, save_path="./visualizations/keypoints.gif")
        # faces = fitter.mano_layer.th_faces.detach().cpu().numpy()

        # # Animate the mesh
        # animate_mano_mesh(verts_mm, faces, title=f"MANO Mesh ({side})", fps=60, save_path="./visualizations/mesh.gif")
    breakpoint()
