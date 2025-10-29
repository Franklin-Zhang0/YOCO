import pyroki as pk
import yourdfpy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from calibration_net.utils import AngleNormalizer, transform_joint3d
from calibration_net.policy.preprocessing import bone_normalize
import tyro

link_mapping = [
    0, 
    3, 4, 5, 6,
    8, 9, 10, 11,
    13, 14, 15, 16,
    17, 18, 19, 20,
    21, 22, 23, 24
]

def main(urdf_path:str, n_samples:int, output_path:str):    
    def filename_handler(fname: str) -> str:
        base_path = urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    urdf_path = Path(urdf_path)

    try:
        urdf = yourdfpy.URDF.load(urdf_path, filename_handler=filename_handler)
    except Exception as e:
        print(f"Error loading URDF from {urdf_path}: {e}")
        return

    hand = pk.Robot.from_urdf(urdf)
    
    angle_normalizer = AngleNormalizer(urdf_path)
    n_joints = len(angle_normalizer.joint_upper_limits)

    all_data_qpos = []
    all_data_normed_qpos = []
    all_data_keypoint = []
    all_data_local_keypoint = []
    
    for _ in tqdm(range(n_samples)):
        normed_qpos = np.random.uniform(0, 1, n_joints)
        qpos = angle_normalizer.denormalize(normed_qpos)
        keypoint = hand.forward_kinematics(cfg=qpos)[:, -3:]
        keypoint = keypoint[np.array(link_mapping)]
        transformed_keypoint = transform_joint3d(keypoint)

        all_data_qpos.append(qpos)
        all_data_normed_qpos.append(normed_qpos)
        all_data_keypoint.append(keypoint)
        all_data_local_keypoint.append(transformed_keypoint)

    transformed_keypoint = np.array(all_data_local_keypoint)
    transformed_keypoint = bone_normalize(transformed_keypoint)

    dataset = {"qpos": all_data_qpos, "normed_qpos": all_data_normed_qpos, "world_hand_joints": all_data_keypoint, "local_hand_joints": transformed_keypoint}
    np.savez(output_path, **dataset)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    tyro.cli(main)