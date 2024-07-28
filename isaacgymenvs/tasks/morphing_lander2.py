import numpy as np
import os
import torch
import datetime
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.terrain_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask
from PIL import Image as im

class MorphingLander2(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        # self.reset_dist = self.cfg["env"]["resetDist"]
        # self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = 500
        # self.random_env = self.cfg["env"]["terrain"]["randomTerrain"]
        # self.min_terrain_height = self.cfg["env"]["terrain"]["minHeight"]
        # self.max_terrain_height = self.cfg["env"]["terrain"]["maxHeight"]
        # self.terrain_step = self.cfg["env"]["terrain"]["step"]
        # self.downsampled_scale = self.cfg["env"]["terrain"]["downsampledScale"]

        self.obst = self.cfg["env"]["terrain"]["discreteObstacles"]["obst"]
        self.max_height = self.cfg["env"]["terrain"]["discreteObstacles"]["maxHeight"]
        self.min_size = self.cfg["env"]["terrain"]["discreteObstacles"]["minSize"]
        self.max_size = self.cfg["env"]["terrain"]["discreteObstacles"]["maxSize"]
        self.num_rectangles = self.cfg["env"]["terrain"]["discreteObstacles"]["numRectangles"]
        self.plat_size = self.cfg["env"]["terrain"]["discreteObstacles"]["platformSize"]
        self.terrain_width = self.cfg["env"]["terrain"]["discreteObstacles"]["terrainWidth"]
        self.terrain_length = self.cfg["env"]["terrain"]["discreteObstacles"]["terrainLength"]

        self.randomize  = self.cfg["task"]["randomize"]

        self.center_x = self.terrain_width / 2
        self.center_y = self.terrain_length / 2

        self.add_vision = self.cfg["env"]["vision"]["addVision"]
        self.save_frames = self.cfg["env"]["vision"]["saveFrames"]

        dofs_per_env = 6
        bodies_per_env = 7

        self.cfg["env"]["numObservations"] = 14
        self.cfg["env"]["numActions"] = 4
        self.cfg["env"]["maxRPM"] = 1250
        self.cfg["env"]["kt"] = 28.15
        self.cfg["env"]["km"] = 0.018
        self.prop_radius = 0.1016
        self.maxHeight = self.cfg["env"]["initialStates"]["maxHeight"]
        self.minHeight = self.cfg["env"]["initialStates"]["minHeight"]
        self.minPitchRoll = self.cfg["env"]["initialStates"]["minPitchRoll"]
        self.maxPitchRoll = self.cfg["env"]["initialStates"]["maxPitchRoll"]
        self.minTiltAngle = self.cfg["env"]["initialStates"]["minTiltAngle"]
        
        self.max_z_force = self.cfg["task"]["disturbance"]["max_z_force"]
        self.std_z_force = self.cfg["task"]["disturbance"]["std_z_force"]
        self.std_xy_force = self.cfg["task"]["disturbance"]["std_xy_force"]
        self.std_torque = self.cfg["task"]["disturbance"]["std_torque"]
        self.max_tilt_rate = self.cfg["task"]["tilt"]["max_tilt_rate"]
        self.tau_up = self.cfg["task"]["motor"]["tau_up"]
        self.tau_down = self.cfg["task"]["motor"]["tau_down"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        runs_dir = os.path.join(parent_dir, "isaacgymenvs/runs")

        latest_dir = None
        latest_time = 0

        for dir_name in os.listdir(runs_dir):
            if dir_name.startswith("MorphingLander") and os.path.isdir(os.path.join(runs_dir, dir_name)):
                dir_time = os.path.getmtime(os.path.join(runs_dir, dir_name))
                if dir_time > latest_time:
                    latest_time = dir_time
                    latest_dir = dir_name

        if latest_dir:
            latest_run_dir = os.path.join(runs_dir, latest_dir)
            print("latest_run_dir:", latest_run_dir)
        else:
            print("No directory found matching the criteria.")

        # Save the compute_morphing_lander_reward function in the latest run dir :
        torch.jit.script(compute_morphing_lander_reward).save(os.path.join(latest_run_dir, "compute_morphing_lander_reward.pt"))



        if not headless:
            self.pos_history = []
            self.ori_history = []
            self.ori_history_quat = []
            self.lin_vel_history = []
            self.ang_vel_history = []
            self.dof_pos_history = []
            self.action_history = []
            self.thrust_history = []
            self.obs_history = []
            if self.randomize:
                self.disturbance_history = []
                self.cg_history = []
                self.ktact_history = []
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)  
        _dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim) 
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        # self.root_tensor = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, 13)
        # self.dof_tensor = gymtorch.wrap_tensor(_dof_tensor).view(self.num_envs, dofs_per_env, 2)

        self.root_tensor = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, 13)
        self.dof_tensor = gymtorch.wrap_tensor(_dof_tensor).view(self.num_envs, dofs_per_env, 2)
        self.contact_forces = gymtorch.wrap_tensor(_contact_tensor).view(self.num_envs, bodies_per_env, 3)

        self.root_states = self.root_tensor
        self.root_pos = self.root_states[..., :3]

        self.root_ori = self.root_states[..., 3:7]
        self.root_ori_euler = torch.tensor(Rotation.from_quat(self.root_ori.cpu()).as_euler('zyx', degrees=True), device=self.device)
        self.root_lin_vel = self.root_states[..., 7:10]
        self.root_ang_vel = self.root_states[..., 10:13]
        self.init_pos = self.root_pos.clone()

        self.dof_states = self.dof_tensor
        self.dof_pos = self.dof_states[..., 0]
        self.dof_vel = self.dof_states[..., 1]

        self.initial_root_states = self.root_tensor.clone()
        #  Set the dof states 0 and 3 to 50 degrees in radians :
        self.initial_dof_states = self.dof_tensor.clone()
        self.max_rpm = self.cfg["env"]["maxRPM"]

        if self.randomize:
            self.kt = self.cfg["env"]["kt"] * (1 + 0.1 *torch.randn((self.num_envs,), device=self.device))
            self.km = self.cfg["env"]["km"] * (1 + 0.1 *torch.randn((self.num_envs,), device=self.device))

            self.z_force_coeff = torch.rand((self.num_envs,), device=self.device) 
            self.stds_z_force = torch.randn((self.num_envs,), device=self.device) * self.std_z_force
            self.stds_xy_force = torch.randn((self.num_envs,), device=self.device) * self.std_xy_force
            self.stds_torque = torch.randn((self.num_envs,), device=self.device) * self.std_torque
            
            self.tilt_rates = (1 + 0.1 * (torch.randn((self.num_envs,), device=self.device))) * self.max_tilt_rate
        else:
            self.kt = self.cfg["env"]["kt"]
            self.km = self.cfg["env"]["km"]

            self.z_force_coeff = torch.zeros((self.num_envs,), device=self.device)
            self.stds_z_force = torch.zeros((self.num_envs,), device=self.device)
            self.stds_xy_force = torch.zeros((self.num_envs,), device=self.device)
            self.stds_torque = torch.zeros((self.num_envs,), device=self.device)

            self.tilt_rates = self.max_tilt_rate*torch.ones((self.num_envs, 1), device=self.device)

        self.dof_lower_limit = torch.tensor([-np.pi/2, 0, 0, -np.pi/2, 0, 0], device=self.device, dtype=torch.float32)
        self.dof_upper_limit = torch.tensor([0, 0, 0, 0, 0, 0], device=self.device, dtype=torch.float32)
        self.actions = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32, requires_grad=False)
        self.dof_position_targets = torch.zeros((self.num_envs,dofs_per_env), device=self.device, dtype=torch.float32, requires_grad=False)
        self.thrusts = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.moments = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        self.frame_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)

        self.terminal_cost_added = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        if self.viewer:
            cam_pos = gymapi.Vec3(7, 7, 5)
            cam_target = gymapi.Vec3(0, 0, 3)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, bodies_per_env, 13)
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]
    
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81 
        if self.randomize:
            self.sim_params.dt = np.random.uniform(0.001, 0.015)
        else:  
            self.dt = self.sim_params.dt
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane(self.obst)
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self, random_terrain):
        if random_terrain:
            
            vertical_scale = 0.05
            horizontal_scale = 0.2
            num_rows = int(self.terrain_width / horizontal_scale)
            num_cols = int(self.terrain_length / horizontal_scale)
            subterrain = SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
            terrain = discrete_obstacles_terrain(subterrain, self.max_height, self.min_size, self.max_size, self.num_rectangles, self.plat_size).height_field_raw
            vertices, triangles = convert_heightfield_to_trimesh(terrain, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = vertices.shape[0]
            tm_params.nb_triangles = triangles.shape[0]
            tm_params.transform.p.x = -1.
            tm_params.transform.p.y = -1.
            self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)
        else:
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self, num_envs, spacing, num_per_row, center_x=None, center_y=None):
        if center_x is not None and center_y is not None:
            lower = gymapi.Vec3(center_x - spacing*num_per_row/2, center_y - spacing*num_per_row/2, 0)
            upper = gymapi.Vec3(center_x + spacing*num_per_row/2, center_y + spacing*num_per_row/2, spacing*num_per_row)
        else:
            lower = gymapi.Vec3(-spacing, -spacing, 0)
            upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/morphing_lander/morphing_lander.urdf"
     
        asset_options = gymapi.AssetOptions()
        lander_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(lander_asset)
        

        self.envs = []
        self.lander_handles = []

        for i in range(self.num_envs):
            rand_pitch = np.random.uniform(self.minPitchRoll, self.maxPitchRoll)*180/np.pi
            rand_roll = np.random.uniform(self.minPitchRoll, self.maxPitchRoll)*180/np.pi
            rand_height = np.random.uniform(self.minHeight, self.maxHeight)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, rand_height)
            pose.r = gymapi.Quat.from_euler_zyx(0, rand_pitch, rand_roll)
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            lander_handle = self.gym.create_actor(env_ptr, lander_asset, pose, "morphinglander", i, 1, 0)
            dof_props = self.gym.get_actor_dof_properties(env_ptr, lander_handle)
            # Only set the first and fourth dof to vel drive mode and the others to none :
            dofs = [0, 3]
            for j in range(self.num_dof):
                if j in dofs:
                    dof_props['driveMode'][j] = gymapi.DOF_MODE_POS
                    dof_props['stiffness'][j] = 1000000
                    dof_props['damping'][j] = 10
                else:
                    dof_props['driveMode'][j] = gymapi.DOF_MODE_NONE
                    dof_props['stiffness'][j] = 0
                    dof_props['damping'][j] = 0
                    
            self.gym.set_actor_dof_properties(env_ptr, lander_handle, dof_props)
            self.envs.append(env_ptr)
            self.lander_handles.append(lander_handle)
                    
        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros((self.num_envs, 4, 3), device=self.device)
            self.base_link_offset = torch.zeros((self.num_envs, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z
                # Create a tensor :
                self.base_link_offset[i, 0] = env_origin.x
                self.base_link_offset[i, 1] = env_origin.y
                self.base_link_offset[i, 2] = env_origin.z

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.terminal_cost_added[:] = compute_morphing_lander_reward(
            self.root_pos,
            self.root_ori,
            self.root_ori_euler,
            self.root_lin_vel,
            self.root_ang_vel,
            self.dof_pos,
            self.reset_buf,
            self.progress_buf,
            self.thrusts,
            self.terminal_cost_added,
            self.contact_forces,
            self.init_pos,
            torch.tensor(self.max_episode_length)
        )

    def reset_idx(self, env_idx):
        env_idx = env_idx.to(self.device)
        num_reset = len(env_idx)

        self.dof_states[env_idx] = self.initial_dof_states[env_idx]

        env_ids_int32 = env_idx.to(torch.int32)

        self.root_states[env_idx] = self.initial_root_states[env_idx]
        
        if self.randomize:
            # self.root_states[env_idx, 0] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
            # self.root_states[env_idx, 1] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 0] = torch_rand_float(-0.5, 0.5, (num_reset, 1), device=self.device).flatten()
            self.root_states[env_idx, 1] = torch_rand_float(-0.5, 0.5, (num_reset, 1), device=self.device).flatten()
            self.root_states[env_idx, 2] = torch_rand_float(0.3, 1, (num_reset, 1), device=self.device).flatten()
            
            rand_tilt = np.random.uniform(self.minTiltAngle, 0)
            self.dof_pos[env_idx, 0] = rand_tilt*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.dof_pos[env_idx, 3] = rand_tilt*torch.ones(num_reset, device=self.device, dtype=torch.float32)

            rand_yaw = 0
            rand_pitch = np.random.uniform(self.minPitchRoll, self.maxPitchRoll)*180/np.pi
            rand_roll = np.random.uniform(self.minPitchRoll, self.maxPitchRoll)*180/np.pi
            init_quat = Rotation.from_euler('zyx', [rand_yaw, rand_pitch, rand_roll], degrees=True).as_quat()
    
            self.root_states[env_idx, 3] = init_quat[0]*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 4] = init_quat[1]*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 5] = init_quat[2]*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 6] = init_quat[3]*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 7] = torch_rand_float(-0.5, 0.5, (num_reset, 1), device=self.device).flatten()
            self.root_states[env_idx, 8] = torch_rand_float(-0.5, 0.5, (num_reset, 1), device=self.device).flatten()
            self.root_states[env_idx, 9] = torch_rand_float(-0.5, 0.5, (num_reset, 1), device=self.device).flatten()
            
            self.root_states[env_idx, 10] = torch_rand_float(-2, 2, (num_reset, 1), device=self.device).flatten()
            self.root_states[env_idx, 11] = torch_rand_float(-2, 2, (num_reset, 1), device=self.device).flatten()
            self.root_states[env_idx, 12] = torch_rand_float(-2, 2, (num_reset, 1), device=self.device).flatten()


            self.kt[env_idx] = self.cfg["env"]["kt"]*(1 + 0.1 * torch.randn((num_reset,), device=self.device))
            self.km[env_idx] = self.cfg["env"]["km"]*(1 + 0.1 * torch.randn((num_reset,), device=self.device))

            # the disturbance in x and y should be between -max_xy_force and max_xy_force :
            self.z_force_coeff[env_idx] = torch.rand((num_reset,), device=self.device) 
            self.stds_z_force[env_idx] = torch.randn((num_reset,), device=self.device) * self.std_z_force
            self.stds_xy_force[env_idx] = torch.randn((num_reset,), device=self.device) * self.std_z_force
            self.stds_torque[env_idx] = torch.randn((num_reset,), device=self.device) * self.std_torque

            self.tilt_rates[env_idx] = (1 + 0.3 * (torch.randn((num_reset,), device=self.device))) * self.max_tilt_rate
        else:
            init_yaw = -0
            init_pitch = 0
            init_roll = -0
            init_quat = Rotation.from_euler('zyx', [-init_yaw, -init_pitch, -init_roll]).as_quat()
            self.root_states[env_idx, 0] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 1] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 0] = 0.0*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 1] = -0.0*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 2] = 0.75*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.dof_pos[env_idx, 0] = np.deg2rad(0)*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.dof_pos[env_idx, 3] = np.deg2rad(0)*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 3] = init_quat[0]*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 4] = init_quat[1]*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 5] = init_quat[2]*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 6] = init_quat[3]*torch.ones(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 7] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 8] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 9] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 10] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 11] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
            self.root_states[env_idx, 12] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)

            self.z_force_coeff[env_idx] = torch.zeros(num_reset, device=self.device)
            self.stds_z_force[env_idx] = torch.zeros(num_reset, device=self.device)
            self.stds_xy_force[env_idx] = torch.zeros(num_reset, device=self.device)
            self.stds_torque[env_idx] = torch.zeros(num_reset, device=self.device)

            self.tilt_rates[env_idx] = self.max_tilt_rate

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        #  Only the first and fourth dof are set to -1.54 and the others to 0 :
        # self.dof_pos[env_idx] = torch.zeros((num_reset, 6), device=self.device, dtype=torch.float32)
        # self.dof_pos[env_idx, 0] = 0
        # self.dof_pos[env_idx, 3] = 0
        self.dof_vel[env_idx] = torch.zeros((num_reset, 6), device=self.device, dtype=torch.float32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_tensor), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # reset forces thrusts moments torques and dof position targets
        self.forces[env_idx] = 0.0
        self.thrusts[env_idx] = 0.0
        self.moments[env_idx] = 0.0
        self.torques[env_idx] = 0.0

        self.dof_position_targets[env_idx] = self.dof_pos[env_idx]
        # self.dof_velocity_targets[env_idx] = self.dof_vel[env_idx]
        self.terminal_cost_added[env_idx] = False
        self.reset_buf[env_idx] = 0
        self.progress_buf[env_idx] = 0

        if not self.headless:
            # First save the history of the previous episode as pickles in the worskpace :
            if len(self.pos_history) > 0:
                #  Create a folder in the history folder with simplz the date and hour as the name after "history" :
                if not os.path.exists("history"):
                    os.makedirs("history")
                
                #  Now create the folder inside the history folder :
                
                now = datetime.datetime.now()
                file_name = "history/" + now.strftime("%Y-%m-%d-%H-%M-%S")
                #  Now save the history of the previous episode all at once as torch tensor, but there should be only one file saved :
                if self.randomize:
                    torch.save({"pos_history": self.pos_history, "ori_history": self.ori_history, "ori_history_quat": self.ori_history_quat, "lin_vel_history": self.lin_vel_history, "ang_vel_history": self.ang_vel_history, "dof_pos_history": self.dof_pos_history, "action_history": self.action_history, "thrust_history": self.thrust_history, "obs_history": self.obs_history, "disturbance_history": self.disturbance_history, "cg_history" : self.cg_history, "ktact_history" : self.ktact_history}, file_name + ".pt")
                else:
                    torch.save({"pos_history": self.pos_history, "ori_history": self.ori_history, "ori_history_quat": self.ori_history_quat, "lin_vel_history": self.lin_vel_history, "ang_vel_history": self.ang_vel_history, "dof_pos_history": self.dof_pos_history, "action_history": self.action_history, "thrust_history": self.thrust_history, "obs_history": self.obs_history}, file_name + ".pt")
                
                                
            self.pos_history = []
            self.ori_history = []
            self.ori_history_quat = []
            self.lin_vel_history = []
            self.ang_vel_history = []
            self.dof_pos_history = []
            self.action_history = []
            self.thrust_history = []
            self.obs_history = []
            if self.randomize:
                self.disturbance_history = []
                self.cg_history = []   
                self.ktact_history = [] 

    def pre_physics_step(self, _actions):
        actions = _actions.clone().to(self.device)
             
        # actions is a tensor of shape (num_envs, 5) where the first 4 elements are the RPMs of the 4 rotors and the last element is the tilt propeller anglular speed
        # we will convert the RPMs to thrusts using the following formula:
        # thrust = k_f * u where u is the action
        # moment = k_t*k_m * u where u is the action
        # Add disturbances forces and torques to make learning more robust :
        # the disturbance can correspond to ground effect when the robot gets close to the ground :
        # First 4 actions : between -1 and 1  => needs to be between 0 and 1
        new_actions = self.postprocess_actions(actions, self.actions)
        self.actions = new_actions
        if self.randomize:
            self.kt = self.kt.unsqueeze(1)

        # Low pass filter on the thrust : thrust = thrust * exp(-dt/tau) + new_thrust * (1 - exp(-dt/tau))
        self.thrusts[:, :4] = new_actions[:, :4] * self.kt
        if self.randomize:
            self.kt = self.kt.squeeze(1)
        self.moments[:, 0] = -self.kt * self.km * new_actions[:, 0]
        self.moments[:, 1] = -self.kt * self.km * new_actions[:, 1]
        self.moments[:, 2] = self.kt * self.km * new_actions[:, 2]
        self.moments[:, 3] = self.kt * self.km * new_actions[:, 3]

        tilt_actions = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)

        tilt_actions[:, 0] = 0.95
        tilt_actions[:, 3] = 0.95
        if self.randomize:
            self.dof_position_targets += tilt_actions * self.dt * (-self.tilt_rates.unsqueeze(1))
        else:
            self.dof_position_targets += tilt_actions * self.dt * (-self.max_tilt_rate)
        
        self.dof_position_targets[:] = tensor_clamp(self.dof_position_targets, self.dof_lower_limit, self.dof_upper_limit)

        # Print the link info :
        rotors_indx = [5, 2, 3, 6]

        for rotor, i in zip(rotors_indx, range(4)):
            self.forces[:, rotor, 2] = self.thrusts[:, i]
            self.torques[:, rotor, 2] = self.moments[:, i]

        # the actual force we're gonna add aren't only the max force itself, it should be inversely proportionnal to the squre of the altitude z :
        if self.randomize:
            # We start by clipping height : if z <= 0.5, z = 0.5
            clipped_height = torch.clamp(self.root_pos[:, 2], self.prop_radius/4, 1000)
            self.cgs =  1/(1 - (self.prop_radius/(4*(clipped_height - 0.2)).pow(2)))
            #  Clip the cg between 1 and 2 :
            self.cgs = torch.clamp(self.cgs, 1, 2)
            # disturbance formula : dist = cg*((sum prop RPM)*kt). It is also inversely proportionnal to the tilt angle :
            self.means_z_force = self.z_force_coeff*(self.cgs - 1)*(self.kt*(new_actions[:, :4].sum(dim=1)))*torch.cos(self.dof_pos[:, 0])
            self.disturbance_z_forces = (self.means_z_force + self.stds_z_force * (torch.randn((self.num_envs,), device=self.device)))
            self.disturbance_xy_forces = self.stds_xy_force.unsqueeze(1) * (torch.randn((self.num_envs, 2), device=self.device))
            self.disturbance_roll_pitch_torques = self.stds_torque.unsqueeze(1) * (torch.randn((self.num_envs, 2), device=self.device))

            self.force_z_disturbances = self.disturbance_z_forces 
            self.force_xy_disturbances = self.disturbance_xy_forces 
            self.roll_pitch_torque_disturbances = self.disturbance_roll_pitch_torques 
            
            base_link_index = 0 
            self.forces[:, base_link_index, 2] = self.force_z_disturbances
            self.forces[:, base_link_index, :2] = self.force_xy_disturbances    
            self.torques[:, base_link_index, :2] = self.roll_pitch_torque_disturbances
            # self.disturbance_z_forces = torch.zeros((self.num_envs, 1), device=self.device)
            # self.disturbance_xy_forces = torch.zeros((self.num_envs, 2), device=self.device)
            # self.disturbance_roll_pitch_torques = torch.zeros((self.num_envs, 2), device=self.device)

        else:
            # clipped_height = torch.clamp(self.root_pos[:, 2], self.prop_radius/4, 1000)
            # self.cgs =  1/(1 - (self.prop_radius/(4*(clipped_height - 0.2)).pow(2)))
            # self.cgs = torch.clamp(self.cgs, 1, 2)
            # self.means_z_force = (self.cgs - 1)*(self.kt*(new_actions[:, :4].sum(dim=1)))*torch.cos(self.dof_pos[:, 0])

            # self.disturbance_z_forces = self.means_z_force
            # self.disturbance_xy_forces = torch.zeros((self.num_envs, 2), device=self.device)
            # self.disturbance_roll_pitch_torques = torch.zeros((self.num_envs, 2), device=self.device)

            # base_link_index = 0
            # self.forces[:, base_link_index, 2] = self.disturbance_z_forces
            # self.forces[:, base_link_index, :2] = self.disturbance_xy_forces
            # self.torques[:, base_link_index, :2] = self.disturbance_roll_pitch_torques

            self.disturbance_z_forces = torch.zeros((self.num_envs, 1), device=self.device)
            self.disturbance_xy_forces = torch.zeros((self.num_envs, 2), device=self.device)
            self.disturbance_roll_pitch_torques = torch.zeros((self.num_envs, 2), device=self.device)


        euler_angles = Rotation.from_quat(self.root_ori.cpu()).as_euler('zyx', degrees=True)

        self.root_ori_euler = torch.tensor(euler_angles, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)

    def postprocess_actions(self, actions, previous_action):
        new_actions = actions.clone()
        new_actions = (new_actions + 1) / 2

        mask_up = new_actions > previous_action
        mask_down = new_actions < previous_action
        new_actions[mask_up] = self.first_order_filter(new_actions[mask_up], previous_action[mask_up], self.tau_up)
        new_actions[mask_down] = self.first_order_filter(new_actions[mask_down], previous_action[mask_down], self.tau_down)
        return new_actions
    
    def first_order_filter(self, x, y, tau):
        dt = torch.tensor(self.dt, device=self.device)
        tau = torch.tensor(tau, device=self.device)
        alpha = torch.exp(-dt / tau)

        return alpha * x + (1 - alpha) * y
    
    def post_physics_step(self):
        self.progress_buf += 1
        
        if not self.headless:
            self.pos_history.append(self.root_pos.clone())
            euler_angles = Rotation.from_quat(self.root_ori.cpu()).as_euler('zyx', degrees=True)
            self.ori_history.append(torch.tensor(euler_angles, device=self.device))
            self.ori_history_quat.append(self.root_ori.clone())
            self.lin_vel_history.append(self.root_lin_vel.clone())
            self.ang_vel_history.append(self.root_ang_vel.clone())
            self.dof_pos_history.append(self.dof_pos.clone())
            self.action_history.append(self.actions.clone())
            self.thrust_history.append(self.thrusts.clone())
            self.obs_history.append(self.obs_buf.clone())
            # Since the dimension of each disturbance is different, we need to concatenate them before saving them :
            # if self.randomize:
                # self.disturbance_history.append(torch.cat((self.force_z_disturbances.unsqueeze(1), self.force_xy_disturbances, self.roll_pitch_torque_disturbances), dim=1))
                # self.cg_history.append(self.cgs.clone())
                # self.ktact_history.append(self.kt.clone()*self.actions[:, :4].sum(dim=1))

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_reward()
        self.compute_observations()
        
    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        target_phi = np.pi/2
        # self.obs_buf[..., 0:3] = self.root_pos
        # self.obs_buf[..., 3:6] = self.root_ori_euler*np.pi/180
        # self.obs_buf[..., 6:9] = self.root_lin_vel
        # self.obs_buf[..., 9:12] = self.root_ang_vel
        # self.obs_buf[..., 12] = self.dof_pos[:, 0] / np.pi/2
        # self.obs_buf[..., 13:] = self.actions

        self.obs_buf[..., 0:3] = self.root_pos
        self.obs_buf[..., 3:7] = self.root_ori
        self.obs_buf[..., 7:10] = self.root_lin_vel
        self.obs_buf[..., 10:13] = self.root_ang_vel
        self.obs_buf[..., 13] = self.dof_pos[:, 0] / np.pi/2
        # self.obs_buf[..., 13:] = self.actions
        
        # self.obs_buf[..., 14:] = self.actions
        # Check if there's any nan value in the observation buffer :   
        if torch.any(torch.isnan(self.obs_buf)):
            print("Nan value in the observation buffer")
            # Print the index of the nan value :
            print(" Index of NaN :" , torch.isnan(self.obs_buf).nonzero(as_tuple=False))
            print(self.obs_buf)

        return self.obs_buf    
    
@torch.jit.script
def compute_morphing_lander_reward(root_pos, root_ori, root_ori_euler, root_lin_vel, root_ang_vel, dof_pos, reset_buf, progress_buf, thrusts, terminal_cost_added, contact_forces, init_pos, max_episode_length):
    
    # position reward
    target_dist = torch.sqrt((init_pos[..., 0] - root_pos[..., 0]) * (init_pos[..., 0] - root_pos[..., 0]) +
                             (init_pos[..., 1] - root_pos[..., 1]) * (init_pos[..., 1] - root_pos[..., 1]) +
                             (0.0-root_pos[..., 2]) * (0.0-root_pos[..., 2]))
    # pos_reward = 1.0 / (1.0 + target_dist * target_dist)
    z_reward = 1.0 / (1.0 + (0.0-root_pos[..., 2]) * (0.0-root_pos[..., 2]))
    xy_reward = 1.0 / (1.0 + (init_pos[..., 0]-root_pos[..., 0]) * (init_pos[..., 0]-root_pos[..., 0]) + (init_pos[..., 1]-root_pos[..., 1]) * (init_pos[..., 1]-root_pos[..., 1]))
    # up reward
    ups = quat_axis(root_ori, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_ang_vel[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)


    #  landing reward, the lower the landing velocity, the higher the reward :
    landing_vel = torch.abs(root_lin_vel[..., 2])
    landing_reward = 1.0 / (1.0 + (0.0 - landing_vel) * (0.0 - landing_vel))

    # final reward function
    reward = 2*z_reward + 0.75*xy_reward + up_reward + spinnage_reward + 1.85*landing_reward

    # if contact_forces.sum() > 0.2:
    # #     print("contact detected")
    #     reward = reward - torch.sum(thrusts * thrusts)*10e-7
    #     reward = reward - xy_reward - 1.5*z_reward - 1.25*landing_reward
    # resets due to misbehavior and print whenever there's a reset :
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 3.0, ones, die)

    # Also die when the pitch or roll get higher than 90 deg but use of quaternion representation :
    die = torch.where(tiltage > 0.9, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset, terminal_cost_added
            
@torch.jit.script
def compute_morphing_lander_reward_old(root_pos, root_ori, root_ori_euler, root_lin_vel, root_ang_vel, dof_pos, reset_buf, progress_buf, thrusts, terminal_cost_added, contact_forces, init_pos, max_episode_length):
    target_dist = torch.sqrt((init_pos[..., 0] - root_pos[..., 0]) * (init_pos[..., 0] - root_pos[..., 0]) +
                             (init_pos[..., 1] - root_pos[..., 1]) * (init_pos[..., 1] - root_pos[..., 1]) +
                             (0.3-root_pos[..., 2]) * (0.3-root_pos[..., 2]))
    
    # target_dist = torch.sqrt((root_pos[..., 0]) * (root_pos[..., 0]) +
    #                          (root_pos[..., 1]) * (root_pos[..., 1]) +
    #                          (0.3-root_pos[..., 2]) * (0.3-root_pos[..., 2]))

    pos_reward = 1.0 / (1.0 + target_dist * target_dist)
    # update terminal cost added:
        # terminal_cost_added = terminal_cost_added | add_terminal_cost
    # print("Reward after and terminal cost added : ", reward, terminal_cost_added)
    # # uprightness
    ups = quat_axis(root_ori, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)
    # spinning
    spinnage = torch.abs(root_ang_vel[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)
    # Tilt reward
    tilt = dof_pos[..., 0]
    target_tilt = torch.sqrt((-np.pi/2-tilt)**2)
    tilt_reward = 1.0 / (1.0 + target_tilt * target_tilt)
    #  landing reward, the lower the landing velocity, the higher the reward :
    landing_vel = torch.abs(root_lin_vel[..., 2])
    landing_reward = 1.0 / (1.0 + (0.4 - landing_vel) * (0.4 - landing_vel))

    # Ensure Euler angles stay close to 0 :
    attitude_term = torch.abs(root_ori_euler[..., 1]) + torch.abs(root_ori_euler[..., 2])
    # reward = pos_reward + pos_reward * (up_reward + spinnage_reward)
    # combined reward
    # uprigness and spinning only matter when close to the target
    # reward = 100*pos_reward + pos_reward * (up_reward + 10*spinnage_reward) + 100*tilt_reward+landing_reward
    # reward = 30*pos_reward + pos_reward * (up_reward + 15*spinnage_reward+ 90*tilt_reward*landing_reward) + 15*landing_reward*tilt_reward*root_pos[..., 2] + 10*landing_reward
    # reward = 20*pos_reward + pos_reward * (up_reward + 15*spinnage_reward+30*tilt_reward*landing_reward) + 15*landing_reward*tilt_reward*root_pos[..., 2] + 30*up_reward
    # reward = 20*pos_reward + pos_reward * (up_reward + 15*spinnage_reward+30*tilt_reward*landing_reward) + 15*landing_reward*tilt_reward*root_pos[..., 2] + 30*up_reward+landing_reward
    # reward = 10*pos_reward + pos_reward * (up_reward + 10*spinnage_reward+ 100*tilt_reward*landing_reward) + landing_reward
    # reward = 10*pos_reward + pos_reward * (up_reward + 10*spinnage_reward+ 90*tilt_reward) + 15*tilt_reward*landing_reward*root_pos[..., 2]
    reward = 2.5*pos_reward + 0.6*pos_reward * (1.5*up_reward + spinnage_reward) + 6*tilt_reward + 0.6*landing_reward*root_pos[..., 2] - attitude_term
    # If there is a non null contact force on both arms, we add reward inversely proportional to the 4 first input actions (thrusts) :
    # We first need to get the contact forces on the arms :
    
    # if contact_forces.sum() > 0:
    #     reward = reward + 1.0 / (1.0 + torch.sum(thrusts * thrusts))

    # resets due to misbehavior and print whenever there's a reset :
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 3.0, ones, die)
    # Also die when the pitch or roll get higher than 90 deg but use of quaternion representation :
    die = torch.where(tiltage > 0.9, ones, die)
    # Also end the episode if the robot is below a certain height and the tilt angle is higher than -np.pi/2:
    # die = torch.where((root_pos[..., 2] < 0.25) & (torch.abs(dof_pos[..., 0] - (-1.55)) > 1.1), ones, die)


    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset, terminal_cost_added
            




        

        




