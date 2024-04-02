import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi, gymtorch
from math import sqrt
import os

# initialize gym
gym = gymapi.acquire_gym()

# # parse arguments
# args = gymutil.parse_arguments(
#     description="Visualize Morphing Lander URDF",
#     custom_parameters=[
#         {"name": "--urdf_path", "type": str, "default": "../../assets/urdf/morphing_lander/morphing_lander.urdf",
#          "help": "Path to the URDF file of the morphing lander"}])

print("CREATE SIMULATION")
# configure sim
sim_params = gymapi.SimParams()
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.up_axis = gymapi.UP_AXIS_Z
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
print("ADD GROUND PLANE")
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load morphing lander asset
print("LOAD MORPHING LANDER ASSET")
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
asset_file = "urdf/morphing_lander/morphing_lander.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# set up the env grid
print("SET UP ENV GRID")
num_envs = 64
num_per_row = int(sqrt(num_envs))
env_spacing = 1.25
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []


# set random seed
np.random.seed(17)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # create morphing lander
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 2)  # Set position of the morphing lander
    # set it to be upright
    pose.r = gymapi.Quat(0, 0, 0, 1)  
    # pose.r = gymapi.Quat(-0.7071, 0, 0, 0.7071)  
    ahandle = gym.create_actor(env, asset, pose, "morphing_lander")
    dof_props = gym.get_actor_dof_properties(env, ahandle)
    dof_props["driveMode"] = gymapi.DOF_MODE_POS
    dof_props["stiffness"] = 0.0
    dof_props["damping"] = 0.0
    gym.set_actor_dof_properties(env, ahandle, dof_props)
# print number of envs
print("Number of envs: ", len(envs))

_root_tensor = gym.acquire_actor_root_state_tensor(sim)
_dof_tensor = gym.acquire_dof_state_tensor(sim)

gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)

root_tensor = gymtorch.wrap_tensor(_root_tensor)
dof_tensor = gymtorch.wrap_tensor(_dof_tensor)

root_states = root_tensor
root_pos =  root_states[:, 0:3]
root_quats = root_states[:, 3:7]
dof_states = dof_tensor
dof_pos = dof_states[:, 0]
dof_vel = dof_states[:, 1]

print("root_pos: ", root_pos)
print("root_quats: ", root_quats)
print("dof_pos: ", dof_pos)
print("dof_vel: ", dof_vel)


# gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 0, 1))
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(10, 10, 2), gymapi.Vec3(0, 0, 2))

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    # gym.simulate(sim)
    # gym.fetch_results(sim, True)

    # # update the viewer
    # gym.step_graphics(sim)
    # Get the DOF states of the morphing lander
    dof_states = gym.get_actor_dof_states(envs[0], ahandle, gymapi.STATE_POS)
    # print("dof_states: ", dof_states)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
            