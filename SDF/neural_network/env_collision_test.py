from srmt.planning_scene import PlanningScene, VisualSimulator
from math import pi, cos, sin
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import torch
from env_collision_model_ver1 import EnvCollNet

np.printoptions(precision=3, suppress=True, linewidth=100, threshold=10000)

import rospy
import tf2_ros

def get_link_transform(tf_buffer, target_frame, source_frame):
    try:
        transform = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
        return transform.transform
    except tf2_ros.LookupException as e:
        rospy.logwarn(f"Could not find the transform from {source_frame} to {target_frame}: {e}")
        return None
    except tf2_ros.ConnectivityException as e:
        rospy.logwarn(f"Connectivity error while looking up transform: {e}")
        return None
    except tf2_ros.ExtrapolationException as e:
        rospy.logwarn(f"Extrapolation error while looking up transform: {e}")
        return None


joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],
                        [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]])

workspace = np.array([[-pi/3, 0.4, 0],         # min (Angle, Radius, Height)
                      [pi/3,  0.9, 1.0]])      # max (Angle, Radius, Height)
obs_size_limit = np.array([[0.1, 0.1, 0.1],    # min (Width, Depth, Height)
                           [0.2, 0.2, 0.2]])   # max (Width, Depth, Height)
obs_num_limit = np.array([5,     # min
                          10])   # max

pc = PlanningScene(arm_names=['panda'], arm_dofs=[7], base_link="panda_link0")
vs = VisualSimulator(width=640, height=576, focal_length_x=504.118, focal_length_y=504.12, z_near=0.25, z_far=2.21) # Azure Kinect SDK (mode: NFOV_UNBINNED)

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

base2panda_tf = get_link_transform(tf_buffer, "base_link", "panda_link0")
base2panda = np.array([base2panda_tf.translation.x, base2panda_tf.translation.y, base2panda_tf.translation.z])
base2cam_tf = get_link_transform(tf_buffer, "base_link", "depth_camera_link")
base2cam = np.array([base2cam_tf.translation.x, base2cam_tf.translation.y, base2cam_tf.translation.z])
cam2view = np.array([np.cos(35/180*np.pi), 0, -np.sin(35/180*np.pi)]) # 35 degree below view

vs.set_cam_and_target_pose(cam_pos=base2cam, target_pos=base2cam+cam2view) # 35 degree below view
scene_bound = np.array([[-0.9, -0.9, -0.4],
                        [0.9, 0.9, 1.4]]) # rough panda workspace
scene_bound = scene_bound + base2panda
vs.set_scene_bounds(scene_bound[0,:], scene_bound[1,:])

voxel_res = 0.05 # [m]
vs.set_grid_resolutions((scene_bound[1,:] - scene_bound[0,:]) / voxel_res)


# NN model load
date = "2024_08_26_22_32_27/"
model_file_name = "env_collision.pkl"

model_dir = "model/env_collsion_ver1/" + date + model_file_name
device = torch.device('cpu')

model = EnvCollNet(dof=7).to(device)

model_state_dict = torch.load(model_dir, map_location=device)
model.load_state_dict(model_state_dict)


# Plot
plt.ion()
fig, ax = plt.subplots(1, 1, figsize=(6, 2))
lines1 = []
lines2 = []

line1, = ax.plot([],[], label='ans', color="blue", linewidth=4.0, linestyle='--')
line2, = ax.plot([],[], label='pred', color = "red", linewidth=2.0)
ax.legend()
ax.grid()


def plt_func(fig, line1, line2, x_data, y_data, y_hat_data):
    if len(x_data) > 10:
        x_data = x_data[-10:]
        y_data = y_data[-10:]
        y_hat_data = y_hat_data[-10:]
    line1.set_data(x_data, y_data)
    line2.set_data(x_data, y_hat_data)
    ax.set_xlim(x_data[0], x_data[-1])
    ax.set_ylim(min(min(y_data), min(y_hat_data))-5, max(max(y_data), max(y_hat_data))+5)
    fig.canvas.draw()
    fig.canvas.flush_events()

x_data = []
y_data = []
y_hat_data = []


for env_iter in range(1000):
    num_obs = np.random.randint(obs_num_limit[0].item(), obs_num_limit[1].item())
    for obs_idx in range(num_obs):
        obs_pos = np.random.uniform(workspace[0], workspace[1]) # angle, radius, height
        pc.add_box(name = "obs_"+str(obs_idx), 
                   dim = np.random.uniform(obs_size_limit[0], obs_size_limit[1]),
                   pos = [obs_pos[1]*cos(obs_pos[0]), obs_pos[1]*sin(obs_pos[0]), obs_pos[2]],
                   quat=R.random().as_quat()
                   )
    vs.load_scene(pc)
    depth = vs.generate_depth_image()
    voxel_grid = vs.generate_voxel_occupancy()

    for q_iter in range(20):
        q = np.random.uniform(low=joint_limit[0,:], high=joint_limit[1,:])
        pc.display(q)
        env_min_dist = pc.min_distance(q, False, True)*100
        with torch.no_grad():
            model.eval()
            x_q = torch.from_numpy(q.reshape(1, -1).astype(np.float32)).to(device)
            x_occ = torch.from_numpy(voxel_grid.reshape(1,1,36,36,36).astype(np.float32)).to(device)
            # jac = torch.autograd.functional.jacobian(model, x)
            NN_output = model(x_q, x_occ)
        env_min_dist_pred = NN_output.cpu().detach().numpy().item()
        print("=====================================")
        print(f'env_true : {env_min_dist}')
        print(f'env_pred : {env_min_dist_pred}')
        print("=====================================")

        x_data.append(env_iter*20+q_iter)
        y_data.append(env_min_dist)
        y_hat_data.append(env_min_dist_pred)
        plt_func(fig, line1, line2, x_data, y_data, y_hat_data)

        keyboard_input = input()
        if keyboard_input == '':
            pass