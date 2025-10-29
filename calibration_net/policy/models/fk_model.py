import torch
from torch import nn
from .resmlp import ResMLPBlock
import numpy as np
from tqdm import tqdm
import os

TIP_INDICIES = [4, 8, 12, 16, 20]

def get_finger_fk(n_joint=4, hidden=128):
    return nn.Sequential(
        nn.Linear(n_joint, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, 3)
    ) 

class FKModel(nn.Module):
    def __init__(self, keypoint_joints):
        # keypoint_joints: a list of list.
        # keypoint[i] is the indices of joints that drive the i-th keypoint.
        # Example: For allegro, [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

        super().__init__()
        num_fingers = len(keypoint_joints)
        
        self.nets = []
        self.n_total_joint = 0

        for joint in keypoint_joints:
            net = get_finger_fk(n_joint=len(joint))
            self.nets.append(net)
            self.n_total_joint += len(joint)

        self.nets = nn.ModuleList(self.nets)

        self.keypoint_joints = keypoint_joints

    def forward(self, joint):
        # x: [B, DOF], joint values. normalized to [-1, 1]. 
        # out:   [B, N, 3], sequence of keypoint.
        keypoints = []
        for i, net in enumerate(self.nets):
            joint_ids = self.keypoint_joints[i]
            keypoint = net(joint[:, joint_ids])
            keypoints.append(keypoint)

        return torch.stack(keypoints, dim=1)


class FKTrainer:
    def __init__(self, fk_model:FKModel, fk_ds_path:str, fs_model_path:str, device:str):
        self.fk_model = fk_model
        self.fk_ds_path = fk_ds_path
        self.fs_model_path = fs_model_path
        self.data = np.load(self.fk_ds_path)
        self.robot_point_cloud = self.get_robot_point_cloud()
        self.device = device

    def get_robot_point_cloud(self):
        all_keypoint_data = self.data["world_hand_joints"][:, TIP_INDICIES, :].transpose(1, 0, 2)
        return all_keypoint_data

    def train(self):
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.data["normed_qpos"], dtype=torch.float32),
            torch.tensor(self.data["world_hand_joints"], dtype=torch.float32)
        )
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        optimizer = torch.optim.Adam(self.fk_model.parameters(), lr=5e-4)
        criterion_fk = nn.MSELoss()

        print("Train Neural Forward Kinematics (FK) from Scratch")
        total_epochs = 20
        for epoch in range(total_epochs):
            all_fk_error = 0
            tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            tqdm_bar.set_postfix_str(f"Epoch {epoch}/{total_epochs}")
            for batch_idx, batch in tqdm_bar:
                qpos, keypoint = batch
                keypoint = keypoint[:, TIP_INDICIES, :]
                qpos = qpos.float().to(self.device)
                keypoint = keypoint.float().to(self.device)
                predicted_keypoint = self.fk_model(qpos)
                optimizer.zero_grad()
                loss = criterion_fk(predicted_keypoint, keypoint)
                loss.backward()
                optimizer.step()

                all_fk_error += loss.item()
            avg_fk_error = all_fk_error / (batch_idx + 1)
            print(f"Neural FK Training Epoch: {epoch}; Training Loss: {avg_fk_error}")
        
        dir_path = os.path.dirname(self.fs_model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.fk_model.state_dict(), self.fs_model_path)
        
        self.fk_model.eval()
        return self.fk_model