import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
import os

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(
    description="Visualize Morphing Lander URDF",
    custom_parameters=[
        {"name": "--urdf_path", "type": str, "default": "../../assets/urdf/morphing_lander/morphing_lander.urdf",
         "help": "Path to the URDF file of the morphing lander"}])

# Configure sim
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Load morphing lander asset
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
asset_file = "urdf/morphing_lander/morphing_lander.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# Create actor for the morphing lander
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)  # Set position of the morphing lander
# Set orientation of the morphing lander twith a roll of 90 °
pose.r = gymapi.Quat(0, 0, 0.7071, 0.7071)
env_ptr = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 0)
ahandle = gym.create_actor(env_ptr, asset, pose, "morphing_lander")

# Set up camera
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))
# dof_pos = gym.get_actor_dof_pos(env_ptr, ahandle)
# print("dof_pos: ", dof_pos)
while not gym.query_viewer_has_closed(viewer):
    # Update the viewer
    gym.draw_viewer(viewer, sim, True)
    

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
