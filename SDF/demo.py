from srmt.planning_scene import PlanningScene, VisualSimulator
from math import pi, cos, sin
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

np.printoptions(precision=3, suppress=True, linewidth=100, threshold=10000)

joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],
                        [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]])

workspace = np.array([[-pi/3, 0.4, 0],      # min (Angle, Radius, Height)
                      [pi/3,  0.9, 1.0]]) # max (Angle, Radius, Height)
obs_size_limit = np.array([[0.1, 0.1, 0.1], # min (Width, Depth, Height)
                           [0.2, 0.2, 0.2]])   # max (Width, Depth, Height)
obs_num_limit = np.array([5,     # min
                          10])  # max

base2panda = np.array([0.3, 0, 0.256])
base2cam = np.array([-0.170, 0, 1.5825])
cam2view = np.array([np.cos(20/180*np.pi), 0, -np.sin(20/180*np.pi)])

pc = PlanningScene(arm_names=['panda'], arm_dofs=[7], base_link="panda_link0")

vs = VisualSimulator(width=640, height=576) # Azure Kinect SDK (mode: NFOV_UNBINNED)
vs.set_cam_and_target_pose(cam_pos=-base2panda+base2cam, target_pos=-base2panda+base2cam+cam2view)
scene_bound = np.array([[-0.9, -0.9, -0.4],
                        [0.9, 0.9, 1.4]])
scene_bound = scene_bound + base2panda
vs.set_scene_bounds(scene_bound[0,:], scene_bound[1,:])
voxel_res = 0.05 # [m]

vs.set_grid_resolutions((scene_bound[1,:] - scene_bound[0,:]) / voxel_res)


title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}


for i in range(1000):
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

    ax1 = plt.figure(1).add_subplot()
    ax1.set_title("depth image", fontsize=16, fontweight='bold', pad=20)
    ax1.imshow(depth)

    ax2 = plt.figure(2).add_subplot(projection='3d')
    ax2.voxels(voxel_grid)
    axis_res = 0.2 # [m]
    ax2.set_xticks(np.arange(0,voxel_grid.shape[0],axis_res/voxel_res))
    ax2.set_yticks(np.arange(0,voxel_grid.shape[1],axis_res/voxel_res))
    ax2.set_zticks(np.arange(0,voxel_grid.shape[2],axis_res/voxel_res))
    ax2.set_xticklabels(np.round(np.arange(scene_bound[0,0],scene_bound[1,0],axis_res),2))
    ax2.set_yticklabels(np.round(np.arange(scene_bound[0,1],scene_bound[1,1],axis_res),2))
    ax2.set_zticklabels(np.round(np.arange(scene_bound[0,2],scene_bound[1,2],axis_res),2))
    ax2.set_title("voxel grid", fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.show()

    for _ in range(50):
        q = np.random.uniform(low=joint_limit[0,:], high=joint_limit[1,:])
        pc.display(q)
        self_min_dist = pc.min_distance(q, True, False)
        env_min_dist = pc.min_distance(q, False, True)
        print(f'self: {self_min_dist}')
        print(f'env : {env_min_dist}')
        pc.print_current_collision_infos()    
        keyboard_input = input()
        if keyboard_input == '':
            pass