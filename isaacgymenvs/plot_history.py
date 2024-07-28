import torch
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def plot_history():
    # Get the latest file
    kt = 28.15
    list_of_files = glob.glob('isaacgymenvs/history/*')
    # Print the current directory:
    latest_file = max(list_of_files, key=os.path.getctime)
    data = torch.load(latest_file)
    pos_history = data["pos_history"]
    x = [t[0][0].item() for t in pos_history]
    y = [t[0][1].item() for t in pos_history]
    z = [t[0][2].item() for t in pos_history]
    ori_history = data["ori_history"]
    yaw = [t[0][0].item() for t in ori_history]
    pitch = [t[0][1].item() for t in ori_history]
    roll = [t[0][2].item() for t in ori_history]
    lin_vel_history = data["lin_vel_history"]
    x_vel = [t[0][0].item() for t in lin_vel_history]
    y_vel = [t[0][1].item() for t in lin_vel_history]
    z_vel = [t[0][2].item() for t in lin_vel_history]
    ang_vel_history = data["ang_vel_history"]
    roll_vel = [t[0][0].item() for t in ang_vel_history]
    pitch_vel = [t[0][1].item() for t in ang_vel_history]
    yaw_vel = [t[0][2].item() for t in ang_vel_history]
    dof_pos_history = data["dof_pos_history"]
    dof_pos = [t[0][0].item() for t in dof_pos_history] 
    thrust_history = data["thrust_history"]
    rotor0 = [t[0][0].item()/kt for t in thrust_history]
    rotor1 = [t[0][1].item()/kt for t in thrust_history]
    rotor2 = [t[0][2].item()/kt for t in thrust_history]
    rotor3 = [t[0][3].item()/kt for t in thrust_history]

    # observation_history = data["obs_history"]
    # obs_tilt_vel = [t[0][17].item() for t in observation_history]

    ori_history_quat = data["ori_history_quat"]
    q_x = [t[0][0].item() for t in ori_history_quat]
    q_y = [t[0][1].item() for t in ori_history_quat]
    q_z = [t[0][2].item() for t in ori_history_quat]
    q_w = [t[0][3].item() for t in ori_history_quat]

    action_history = data["action_history"]
    tilt_vel = [t[0][4].item() for t in action_history]

    # disturbance = data["disturbance_history"]
    # dist_f_z = [t[0][0].item() for t in disturbance]
    # dist_f_x = [t[0][1].item() for t in disturbance]
    # dist_f_y = [t[0][2].item() for t in disturbance]
    # dist_tau_x = [t[0][3].item() for t in disturbance]
    # dist_tau_y = [t[0][4].item() for t in disturbance]
    
    # cg = data["cg_history"]
    # cg = [t.item() for t in cg]

    # ktact = data["ktact_history"]
    # ktact = [t.item() for t in ktact]
    # # Plotting Position
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(x, label="x", color="red")
    plt.plot(y, label="y", color="green")
    plt.plot(z, label="z", color="blue")
    plt.title("Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.legend()

    # Plotting Orientation represented by quaternion
    plt.subplot(2, 2, 2)
    plt.plot(q_x, label="q_x", color="red")
    plt.plot(q_y, label="q_y", color="green")
    plt.plot(q_z, label="q_z", color="blue")
    plt.plot(q_w, label="q_w", color="orange")
    plt.title("Quaternion")
    plt.xlabel("Time (s)")
    plt.ylabel("Quaternion")
    plt.grid(True)
    plt.legend()
    
    # Plotting Linear Velocity
    plt.subplot(2, 2, 3)
    plt.plot(x_vel, label="x", color="red")
    plt.plot(y_vel, label="y", color="green")
    plt.plot(z_vel, label="z", color="blue")
    plt.title("Linear Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend()

    # Plotting Angular Velocity
    plt.subplot(2, 2, 4)
    plt.plot(yaw_vel, label="yaw", color="red")
    plt.plot(pitch_vel, label="pitch", color="green")
    plt.plot(roll_vel, label="roll", color="blue")
    plt.title("Angular Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Plotting Thrusts separately
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(rotor0, label="rotor0", color="red")
    plt.title("Thrust (Rotor 0)")
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust (N)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(rotor1, label="rotor1", color="green")
    plt.title("Thrust (Rotor 1)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(rotor2, label="rotor2", color="blue")
    plt.title("Thrust (Rotor 2)")
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust (N)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(rotor3, label="rotor3", color="orange")
    plt.title("Thrust (Rotor 3)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Plotting all Thrusts together
    plt.figure(figsize=(10, 6))
    plt.plot(rotor0, label="rotor0", color="red")
    plt.plot(rotor1, label="rotor1", color="green")
    plt.plot(rotor2, label="rotor2", color="blue")
    plt.plot(rotor3, label="rotor3", color="orange")
    plt.title("Thrusts")
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust (N)")
    plt.grid(True)
    plt.legend()

    
    # # Plotting Disturbance as well as the tilt angle (dof pos) and altitude on the third plot and the the cos of the tilt angle
    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 2, 1)
    # plt.plot(dist_f_z, label="f_z", color="red")
    # plt.plot(dist_f_x, label="f_x", color="green")
    # plt.plot(dist_f_y, label="f_y", color="blue")
    # plt.title("Disturbance Forces")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Force (N)")
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(2, 2, 2)
    # plt.plot(dist_tau_x, label="tau_x", color="red")
    # plt.plot(dist_tau_y, label="tau_y", color="green")
    # plt.title("Disturbance Torques")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Torque (Nm)")
    # plt.grid(True)
    # plt.legend()

    # Have two differenty y axes for the tilt angle and the altitude
    plt.subplot(2, 2, 3)
    ax1 = plt.gca()  # Get the current axis
    ax2 = ax1.twinx()  # Create a twin axis sharing the x-axis
    ax1.plot(dof_pos, label="dof_pos", color="red")
    ax1.set_ylabel("dof_pos (rad)", color="red")
    ax1.tick_params(axis='y', labelcolor="red")
    ax2.plot(z, label="z", color="blue")
    ax2.set_ylabel("z (m)", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    plt.title("Tilt Angle")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Plotting tilt velocity
    plt.figure(figsize=(10, 6))
    plt.plot(tilt_vel, label="tilt_vel", color="red")
    plt.title("Tilt Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Tilt Velocity (rad/s)")
    plt.grid(True)
    plt.legend()
    
    # plot yaw pitch roll :
    plt.figure(figsize=(10, 6))
    plt.plot(yaw, label="yaw", color="red")
    plt.plot(pitch, label="pitch", color="green")
    plt.plot(roll, label="roll", color="blue")
    plt.title("Yaw Pitch Roll")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.legend()



    
    # plt.subplot(2, 2, 4)
    # cos_dof_pos = [np.cos(t) for t in dof_pos]
    # plt.plot(cos_dof_pos, label="cos(dof_pos)", color="red")
    # plt.title("Cosine of Tilt Angle")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Cosine")
    # plt.grid(True)
    # plt.legend()

    # # plot cg 
    # plt.figure(figsize=(10, 6))
    # plt.plot(cg, label="cg", color="red")
    # plt.title("Cg")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Cg ")
    # plt.grid(True)
    # plt.legend()

    # # plot ktact
    # plt.figure(figsize=(10, 6))
    # plt.plot(ktact, label="ktact", color="red")
    # plt.title("ktact")
    # plt.xlabel("Time (s)")
    # plt.ylabel("ktact ")    
    # plt.grid(True)
    # plt.legend()

    


    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    plot_history()
