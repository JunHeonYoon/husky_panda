from srmt.planning_scene import PlanningScene
import numpy as np
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

"""
This version acquires [q, min_dist] dataset in unoform distribution in q space.
"""


def main(args):
    # Number of threads
    if args.num_th > os.cpu_count():
        args.num_th = os.cpu_count()
    print("core: {}".format(args.num_th))

    # Parameters
    joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],  # min 
                            [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]]) # max
    
    data_dir = "self_data"
    if not os.path.exists(data_dir): os.mkdir(data_dir)
    
    
    def work(id, result):
        dataset = {}
        q_set = []
        min_dist_set = []

        # Create Planning Scene
        pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="base_link", topic_name="planning_scene_" + str(id))

        np.random.seed(id)
        t0 = time.time()
        for iter in range(int(args.num_q / args.num_th)):
            q = np.random.uniform(low=joint_limit[0], high=joint_limit[1], size=7)
            pc.display(q)
            min_dist = pc.min_distance(q)*100 # [m] -> [cm]
                
            q_set.append(q)
            min_dist_set.append(min_dist)

            if (iter / int(args.num_q / args.num_th)*100) % 10 == 0 :
                t1 = time.time()
                print("{0:.1f}% of dataset accomplished on th:{1}! (Time: {2:.02f})".format(iter / int(args.num_q / args.num_th)*100, id, t1-t0))

        dataset["q"] = q_set
        dataset["min_dist"] = min_dist_set
        dataset["id"] = id

        result.put(dataset)
        return


    result = Queue()
    threads =[]

    dataset = {}
    dataset["q"] = []
    dataset["min_dist"] = []

    for i in range(args.num_th):
        th = Process(target=work, args=(i,result))
        threads.append(th)
    
    print("Start multi-threading!")
    for i in range(args.num_th):
        threads[i].start()

    for i in range(args.num_th):
        data = result.get()
        dataset["q"] = dataset["q"] + data["q"]
        dataset["min_dist"] = dataset["min_dist"] + data["min_dist"]

    for i in range(args.num_th):
        threads[i].join()


    date = dt.datetime.now()
    data_dir = data_dir + "/{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    os.mkdir(data_dir)

    with open(data_dir + "/dataset.pickle", "wb") as f:
        dataset["q"] = np.array(dataset["q"])
        dataset["min_dist"] = np.array(dataset["min_dist"])
        pickle.dump(dataset,f)
    print("Total number of data: {} (coll: {}, close:{}, free: {})".format(dataset["min_dist"].size, 
                                                                           np.sum(dataset["min_dist"]<=0), 
                                                                           np.sum((dataset["min_dist"]>0) & (dataset["min_dist"]<=5)), 
                                                                           np.sum(dataset["min_dist"]>5)))

    with open(data_dir + "/param_setting.txt", "w", encoding='UTF-8') as f:
        params = {"number of dataset": dataset["min_dist"].size,
                  "number of collsion data": np.sum(dataset["min_dist"]<=0),
                  "number of close data": np.sum((dataset["min_dist"]>0) & (dataset["min_dist"]<=5)),
                  "number of free data": np.sum(dataset["min_dist"]>5)
                  }
        f.write("Husky-Panda Self collision dataset:\n")
        for param, value in params.items():
            f.write(f'\t\t{param} : {value}\n')

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
    parser.add_argument("--num_q", type=int, default=10000000)

    args = parser.parse_args()
    main(args)
