import numpy as np
import os
import torch
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.terrain_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

class MorphingLander(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        # self.reset_dist = self.cfg["env"]["resetDist"]
        # self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = 500
        self.random_env = self.cfg["env"]["terrain"]["randomTerrain"]
        self.min_terrain_height = self.cfg["env"]["terrain"]["minHeight"]
        self.max_terrain_height = self.cfg["env"]["terrain"]["maxHeight"]
        self.terrain_step = self.cfg["env"]["terrain"]["step"]
        self.downsampled_scale = self.cfg["env"]["terrain"]["downsampledScale"]
        dofs_per_env = 6
        bodies_per_env = 7

        self.cfg["env"]["numObservations"] = 19
        self.cfg["env"]["numActions"] = 5
        self.cfg["env"]["maxRPM"] = 1250
        self.cfg["env"]["kt"] = 29
        self.cfg["env"]["km"] = 0.09
        self.maxHeight = self.cfg["env"]["initialStates"]["maxHeight"]
        self.minHeight = self.cfg["env"]["initialStates"]["minHeight"]
        self.minPitchRoll = self.cfg["env"]["initialStates"]["minPitchRoll"]
        self.maxPitchRoll = self.cfg["env"]["initialStates"]["maxPitchRoll"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)  
        _dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim) 
        
        # self.root_tensor = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, 13)
        # self.dof_tensor = gymtorch.wrap_tensor(_dof_tensor).view(self.num_envs, dofs_per_env, 2)

        self.root_tensor = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, 13)
        self.dof_tensor = gymtorch.wrap_tensor(_dof_tensor).view(self.num_envs, dofs_per_env, 2)

        self.root_states = self.root_tensor
        self.root_pos = self.root_states[..., :3]
        self.root_ori = self.root_states[..., 3:7]
        self.root_lin_vel = self.root_states[..., 7:10]
        self.root_ang_vel = self.root_states[..., 10:13]

        self.dof_states = self.dof_tensor
        self.dof_pos = self.dof_states[..., 0]
        self.dof_vel = self.dof_states[..., 1]
        
        self.initial_root_states = self.root_tensor.clone()
        self.initial_dof_states = self.dof_tensor.clone()

        self.max_rpm = self.cfg["env"]["maxRPM"]
        self.kt = self.cfg["env"]["kt"]
        self.km = self.cfg["env"]["km"]

        self.rpm_lower_limit = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.rpm_upper_limit = self.max_rpm * torch.ones(4, device=self.device, dtype=torch.float32)

        self.thrust_lower_limit = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.thrust_upper_limit = self.kt * torch.ones(4, device=self.device, dtype=torch.float32)

        self.torque_lower_limit = -self.kt * self.km * torch.ones(4, device=self.device, dtype=torch.float32)
        self.torque_upper_limit = self.kt * self.km * torch.ones(4, device=self.device, dtype=torch.float32)

        self.dof_lower_limit = torch.tensor([-np.pi/8, 0, 0, -np.pi/8, 0, 0], device=self.device, dtype=torch.float32)
        self.dof_upper_limit = torch.tensor([np.pi/8, 0, 0, np.pi/8, 0, 0], device=self.device, dtype=torch.float32)
        
        # self.dof_position_targets = torch.zeros((self.num_envs,dofs_per_env), device=self.device, dtype=torch.float32, requires_grad=False)
        self.dof_velocity_targets = torch.zeros((self.num_envs,dofs_per_env), device=self.device, dtype=torch.float32, requires_grad=False)
        self.thrusts = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.moments = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

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
        self.dt = self.sim_params.dt
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane(self.random_env)
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self, random_terrain):
        if random_terrain:
            terrain_width = 12
            terrain_length = 12
            vertical_scale = 0.005
            horizontal_scale = 0.5
            num_rows = int(terrain_width / horizontal_scale)
            num_cols = int(terrain_length / horizontal_scale)
            print("num rows : ", num_rows)
            print("num cols : ", num_cols)
            print("vertical scale : ", vertical_scale)
            print("horizontal scale : ", horizontal_scale)
            subterrain = SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
            terrain = random_uniform_terrain(subterrain, self.min_terrain_height, self.max_terrain_height, self.terrain_step, self.downsampled_scale).height_field_raw
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
    
    def _create_envs(self, num_envs, spacing, num_per_row):
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
            dof_props['driveMode'].fill(gymapi.DOF_MODE_VEL)
            dof_props['stiffness'].fill(0)
            dof_props['damping'].fill(100000)
            self.gym.set_actor_dof_properties(env_ptr, lander_handle, dof_props)
            dof_names = self.gym.get_actor_dof_dict(env_ptr, lander_handle)
            print("dof names : ", dof_names)            
            self.envs.append(env_ptr)
            self.lander_handles.append(lander_handle)
        
        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros((self.num_envs, 4, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z
               

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_morphing_lander_reward(
            self.root_pos,
            self.root_ori,
            self.root_lin_vel,
            self.root_ang_vel,
            self.dof_pos,
            self.reset_buf,
            self.progress_buf,
            torch.tensor(self.max_episode_length)
        )

    def reset_idx(self, env_idx):
        env_idx = env_idx.to(self.device)
        num_reset = len(env_idx)

        self.dof_states[env_idx] = self.initial_dof_states[env_idx]

        env_ids_int32 = env_idx.to(torch.int32)

        self.root_states[env_idx] = self.initial_root_states[env_idx]
        self.root_states[env_idx, 0] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
        self.root_states[env_idx, 1] = torch.zeros(num_reset, device=self.device, dtype=torch.float32)
        self.root_states[env_idx, 2] = torch_rand_float(1, 3, (num_reset, 1), device=self.device).flatten()

        # just tensors of dim 1 :
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
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.dof_pos[env_idx] = torch.zeros((num_reset, 6), device=self.device, dtype=torch.float32)
        self.dof_vel[env_idx] = torch.zeros((num_reset, 6), device=self.device, dtype=torch.float32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_tensor), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # reset forces thrusts moments torques and dof position targets
        self.forces[env_idx] = 0.0
        self.thrusts[env_idx] = 0.0
        self.moments[env_idx] = 0.0
        self.torques[env_idx] = 0.0

        # self.dof_position_targets[env_idx] = self.dof_pos[env_idx]
        self.dof_velocity_targets[env_idx] = self.dof_vel[env_idx]
        self.reset_buf[env_idx] = 0
        self.progress_buf[env_idx] = 0
    
    def pre_physics_step(self, _actions):
        actions = _actions.clone().to(self.device)

        # actions is a tensor of shape (num_envs, 5) where the first 4 elements are the RPMs of the 4 rotors and the last element is the tilt propeller anglular speed
        # we will convert the RPMs to thrusts using the following formula:
        # thrust = k_f * u where u is the action
        # moment = k_t*k_m * u where u is the action

        
        self.thrusts[:, :4] = self.kt * actions[:, :4]
        self.thrusts[:] = tensor_clamp(self.thrusts, self.thrust_lower_limit, self.thrust_upper_limit)
        self.moments[:, 0] = -self.kt * self.km * actions[:, 0]
        self.moments[:, 1] = self.kt * self.km * actions[:, 1]
        self.moments[:, 2] = -self.kt * self.km * actions[:, 2]
        self.moments[:, 3] = self.kt * self.km * actions[:, 3]
        self.moments[:] = tensor_clamp(self.moments, self.torque_lower_limit, self.torque_upper_limit)

        max_dof_speed_scale = 22.5

        # Since the dof_positions_target has the shape num_envs * 6, the tilt action should be applied to both tilt motors. so the action needs to have six elements instead of one (the four last elements being simply 0 since we never actuate the rotors) :
        tilt_actions = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        # tilt_actions = torch.ones((self.num_envs, 6), device=self.device, dtype=torch.float32)


        tilt_actions[:, 0] = actions[:, 4]
        tilt_actions[:, 3] = actions[:, 4]
        self.dof_velocity_targets = tilt_actions * max_dof_speed_scale
        # self.dof_position_targets[:] = tensor_clamp(self.dof_position_targets, self.dof_lower_limit, self.dof_upper_limit)
        print("dof_velocity_targets after : ", self.dof_velocity_targets)
        # The names to indey dictionnary is the following :
        # Rigid body indices :  {'arml': 1, 'armr': 4, 'base_link': 0, 'rotor0': 5, 'rotor1': 2, 'rotor2': 3, 'rotor3': 6}

        rotors_indx = [5, 2, 3, 6]

        for rotor, i in zip(rotors_indx, range(4)):
            self.forces[:, rotor, 2] = self.thrusts[:, i]
            self.torques[:, rotor, 2] = self.moments[:, i]
        
        # print("dof positions targets after :, ", self.dof_position_targets)
        # Convert the gpu target tensors to cpu target tensors :

        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_velocity_targets))
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)
        # print("dof pos before step : ", self.dof_pos)
        print("dof vel before step : ", self.dof_vel)
    
    def post_physics_step(self):
        # print("Dof pos after step : ", self.dof_pos)
        print("Dof vel after step : ", self.dof_vel)

        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)


        print("Dof target vel after step : ", self.dof_velocity_targets)
        print("Dof vel after step : ", self.dof_vel)

        self.compute_reward()
        self.compute_observations()

        if self.viewer and self.debug_viz:

            rotor_indices = torch.LongTensor([5, 2, 3 ,6])
            quats = self.rb_quats[:, rotor_indices]
            dirs = quat_axis(quats.view(self.num_envs * 4, 4), 2).view(self.num_envs, 4, 3)
            starts = self.rb_positions[:, rotor_indices] + self.rotor_env_offsets
            ends = starts + 0.1 * self.thrusts.view(self.num_envs, 4, 1) * dirs

            # submit debug line geometry
            verts = torch.stack([starts, ends], dim=2).cpu().numpy()
            colors = np.zeros((self.num_envs * 4, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_envs * 4, verts, colors)
    
    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        target_x = 0.0
        target_y = 0.0
        target_z = 0.15
        target_phi = np.pi/2
        target_dof = [target_phi, 0, 0, target_phi, 0, 0]
        # self.obs_buf[..., 0] = (target_x - self.root_pos[..., 0]) / 3
        # self.obs_buf[..., 1] = (target_y - self.root_pos[..., 1]) / 3
        # self.obs_buf[..., 2] = (target_z - self.root_pos[..., 2]) / 3
        self.obs_buf[..., 0:3] = self.root_pos
        self.obs_buf[..., 3:7] = self.root_ori
        self.obs_buf[..., 7:10] = self.root_lin_vel
        self.obs_buf[..., 10:13] = self.root_ang_vel
        # self.obs_buf[..., 13:] = (self.dof_pos - target_dof) / np.pi/2
        self.obs_buf[..., 13:] = self.dof_vel

        print("Observations : ", self.obs_buf)
        return self.obs_buf    
    
@torch.jit.script
def compute_morphing_lander_reward(root_pos, root_ori, root_lin_vel, root_ang_vel, dof_pos, reset_buf, progress_buf, max_episode_length):
    target_dist = torch.sqrt(root_pos[..., 0] * root_pos[..., 0] +
                             root_pos[..., 1] * root_pos[..., 1] +
                             (0.5 - root_pos[..., 2]) * (0.5 - root_pos[..., 2]))
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)
    # uprightness
    ups = quat_axis(root_ori, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)
    # spinning
    spinnage = torch.abs(root_ang_vel[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)
    # Tilt reward
    tilt = dof_pos[..., 0]
    target_tilt = torch.sqrt((-np.pi/2)**2- tilt*tilt)
    tilt_reward = 1.0 / (1.0 + target_tilt * target_tilt)
    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward) + tilt_reward

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 3.0, ones, die)
    die = torch.where(root_pos[..., 2] < 0.2, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset
            







        



