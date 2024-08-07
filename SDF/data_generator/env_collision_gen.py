from srmt.planning_scene import PlanningScene, VisualSimulator
import numpy as np
from math import pi, cos, sin
from scipy.spatial.transform import Rotation as R
import time
import os
import datetime as dt
import sys
import pickle
import argparse
from multiprocessing import Process, Queue

np.printoptions(precision=3, suppress=True, linewidth=100, threshold=10000)
np.set_printoptions(threshold=sys.maxsize)
title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}


def main(args):
    # Number of threads
    if args.num_th > os.cpu_count():
        args.num_th = os.cpu_count()
    print("core: {}".format(args.num_th))

    # Parameters
    joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],  # min
                            [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]]) # max

    workspace = np.array([[-pi/3, 0.4, 0],      # min (Angle, Radius, Height)
                          [pi/3,  0.9, 1.0]])   # max (Angle, Radius, Height)
    obs_size_limit = np.array([[0.1, 0.1, 0.1],  # min (Width, Depth, Height)
                               [0.2, 0.2, 0.2]]) # max (Width, Depth, Height)
    obs_num_limit = np.array([5,     # min
                              10])   # max

    base2panda = np.array([0.3, 0, 0.256])
    base2cam = np.array([-0.170, 0, 1.5825])
    cam2view = np.array([np.cos(20/180*np.pi), 0, -np.sin(20/180*np.pi)])

    scene_bound = np.array([[-0.9, -0.9, -0.4],
                            [0.9, 0.9, 1.4]])
    scene_bound = scene_bound + base2panda


    data_dir = "env_data"
    if not os.path.exists(data_dir): os.mkdir(data_dir)

    num_envs_per_thread = [args.num_env // args.num_th + (1 if i < args.num_env % args.num_th else 0) for i in range(args.num_th)]

    def work(id, result):
        t0 = time.time()
        np.random.seed(id)

        dataset = []
    
        # Create Planning Scene
        pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="panda_link0", topic_name="planning_scene_" + str(id))

        # Create cameras
        vs = VisualSimulator(width=640, height=576) # Azure Kinect SDK (mode: NFOV_UNBINNED)
        vs.set_cam_and_target_pose(cam_pos=-base2panda+base2cam, target_pos=-base2panda+base2cam+cam2view)
        vs.set_scene_bounds(scene_bound[0,:], scene_bound[1,:])
        vs.set_grid_resolutions((scene_bound[1,:] - scene_bound[0,:]) / args.voxel_res)


        for env_iter in range(num_envs_per_thread[id]):
            env_idx = (sum(num_envs_per_thread[0:id]) if id != 0 else 0) + env_iter

            q_set = []
            min_dist_set = []

            # Create random obstacle
            num_obs = np.random.randint(obs_num_limit[0].item(), obs_num_limit[1].item())
            for obs_idx in range(num_obs):
                obs_pos = np.random.uniform(workspace[0], workspace[1]) # angle, radius, height
                pc.add_box(name = "obs_"+str(obs_idx), 
                           dim = np.random.uniform(obs_size_limit[0], obs_size_limit[1]),
                           pos = [obs_pos[1]*cos(obs_pos[0]), obs_pos[1]*sin(obs_pos[0]), obs_pos[2]],
                           quat = R.random().as_quat()
                           )
            vs.load_scene(pc)
            depth = vs.generate_depth_image()
            voxel_grid = vs.generate_voxel_occupancy()

            # random configuration
            for joint_iter in range(args.num_q_per_env):
                q = np.random.uniform(low=joint_limit[0,:], high=joint_limit[1,:])
                pc.display(q)
                env_min_dist = pc.min_distance(q, False, True)*100 # [m] -> [cm]
                q_set.append(q)
                min_dist_set.append(env_min_dist)

            env_dataset = {"env_idx": env_idx,
                           "q": np.array(q_set),
                           "min_dist": np.array(min_dist_set),
                           "depth": depth,
                           "occupancy": voxel_grid}
            dataset.append(env_dataset)

            if ((env_iter+1) / num_envs_per_thread[id]) % 0.1 == 0 : 
                print("{0:.1f} sec {1:.1f}% completed on th:{2}.".format(time.time()-t0, 
                                                                         (env_iter+1) / num_envs_per_thread[id] * 100, 
                                                                         id))
        result.put(dataset)
        return
    
    result = Queue()
    threads = []
    dataset = []
    
    for i in range(args.num_th):
        th = Process(target=work, args=(i,result))
        threads.append(th)
    
    print("Start multi-threading!")
    for i in range(args.num_th):
        threads[i].start()

    for i in range(args.num_th):
        data = result.get()
        dataset = dataset + data

    for i in range(args.num_th):
        threads[i].join()


    date = dt.datetime.now()
    data_dir = data_dir + "/{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    os.mkdir(data_dir)

    with open(data_dir + "/dataset.pickle", "wb") as f:
        pickle.dump(dataset,f)

    with open(data_dir + "/param_setting.txt", "w", encoding='UTF-8') as f:
        params = {"num_env": args.num_env,
                  "num_q_per_env": args.num_q_per_env,
                  "voxel_res": args.voxel_res,
                  "seed": args.seed}
        for param, value in params.items():
            f.write(f'{param} : {value}\n')


    # import shutil
    # folder_path = "data/"
    # num_save = 3
    # order_list = sorted(os.listdir(folder_path), reverse=True)[1:]
    # remove_folder_list = order_list[num_save:]
    # for rm_folder in remove_folder_list:
    #     shutil.rmtree(folder_path+rm_folder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_th", type=int, default=30)
    parser.add_argument("--num_env", type=int, default=2000)
    parser.add_argument("--num_q_per_env", type=int, default=500)
    parser.add_argument("--voxel_res", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
