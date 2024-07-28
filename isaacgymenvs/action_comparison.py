import torch
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import onnxruntime as ort
from IPython import embed
import onnx
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs

def compare_actions():

    # Load the latest observation and action history
    list_of_files = glob.glob('isaacgymenvs/history/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    data = torch.load(latest_file)

    pos_history = data["pos_history"]
    pos = [t[0].cuda() for t in pos_history]  # Move tensors to CUDA device
    # print(pos)
    ori_history = data["ori_history"]
    ori = [t[0].cuda() for t in ori_history]  # Move tensors to CUDA device
    ori_history_quat = data["ori_history_quat"]
    ori_quat = [t[0].cuda() for t in ori_history_quat]  # Move tensors to CUDA device
    lin_vel_history = data["lin_vel_history"]
    lin_vel = [t[0].cuda() for t in lin_vel_history]  # Move tensors to CUDA device
    ang_vel_history = data["ang_vel_history"]
    ang_vel = [t[0].cuda() for t in ang_vel_history]  # Move tensors to CUDA device
    dof_pos_history = data["dof_pos_history"]
    dof_pos = [t[0][0].cuda() for t in dof_pos_history]  # Move tensors to CUDA device
    action_history = data["action_history"]
    actions = [t[0].cuda() for t in action_history]  # Move tensors to CUDA device
    obs_history = data["obs_history"]
    obs = [t[0].cuda() for t in obs_history]  # Move tensors to CUDA device


    # Load the trained model with ONNX format :
    ort_model = ort.InferenceSession("/home/m4pc/Documents/IsaacGymEnvs/isaacgymenvs/MorphingLander.onnx")

    # Loop over the observations and get the actions from the model
    onnx_actions = []
    obs_onnx_hist = []

    # Initial previous action :
    prev_actions = actions[0]
    # add a dimension to the prev_actions
    first_actions = prev_actions.unsqueeze(0)
    print("prev_actions: ", first_actions.cpu().numpy())
    onnx_actions.append(first_actions.cpu().numpy())
    for i in range(len(pos)):
        # print("pos[i]: ", pos[i])
        # print("ori_quat[i]: ", ori_quat[i])
        # print("lin_vel[i]: ", lin_vel[i])
        # print("ang_vel[i]: ", ang_vel[i])
        # print("dof_pos[i]: ", dof_pos[i])
        # print("prev_actions: ", prev_actions)
        dof_pos_tensor = dof_pos[i].view(1)
        # print("dof_pos: ", dof_pos)
        obs_onnx = torch.cat([pos[i], ori_quat[i], lin_vel[i], ang_vel[i], dof_pos_tensor/np.pi/2], dim=0).float()
        obs_onnx = obs_onnx.unsqueeze(0)

        obs_numpy = obs_onnx.cpu().numpy()
        print("obs_numpy: ", obs_numpy)
        print("obs_numpy shape: ", obs_numpy.shape)
        print("obs type: ", type(obs_numpy))
        obs_onnx_hist.append(obs_onnx)
        
        outputs = ort_model.run(
            None,
            {"obs": obs_numpy},
        )
        # Clip the outputs to be in the range [-1, 1]. But the outputs aren't tensor so the function used can't be trch.clamp
        outputs[0][0][0] = np.clip(outputs[0][0][0], -1, 1)
        outputs[0][0][1] = np.clip(outputs[0][0][1], -1, 1)
        outputs[0][0][2] = np.clip(outputs[0][0][2], -1, 1)
        outputs[0][0][3] = np.clip(outputs[0][0][3], -1, 1)
        outputs[0][0][4] = np.clip(outputs[0][0][4], -1, 1)
        
        # The 4 first actions should be modified to be in the range [0, 1 ] instead of [-1, 1]
        outputs[0][0][0] = (outputs[0][0][0] + 1) / 2
        outputs[0][0][1] = (outputs[0][0][1] + 1) / 2
        outputs[0][0][2] = (outputs[0][0][2] + 1) / 2
        outputs[0][0][3] = (outputs[0][0][3] + 1) / 2
        print("outputs[0]: ", outputs[0])
        onnx_actions.append(outputs[0])
        prev_actions = torch.tensor(outputs[0], device='cuda:0').squeeze(0)
        # print("prev_actions: ", prev_actions)

    actions = torch.stack(actions).cpu().numpy()  
    print(actions.shape)  
    print("onnx shape: ", np.array(onnx_actions).shape)
    # delete the second dimension of the onnx_actions
    onnx_actions = np.array(onnx_actions).squeeze(1)
    
    #  Plot the actions :
    plt.figure()
    plt.plot(actions[:, 0], label="Real actions")
    plt.plot(onnx_actions[:, 0], label="ONNX actions")
    plt.legend()
    plt.show()

    
    



if __name__ == "__main__":
    compare_actions()

