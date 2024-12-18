#! /usr/bin/env python 
import rospy
from std_msgs.msg import Int8MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import threading
import torch
import sys
import torch.autograd.functional as F
import time


sys.path.append("/home/yoonjunheon/catkin_ws/src/husky_panda/SDF/neural_network")
from env_collision_model_ver1 import EnvCollNet

N = 1
torch.set_num_threads(20) 
print(torch.backends.mkldnn.enabled)

class env_dist_check:
    def __init__(self) -> None:
        rospy.init_node('array_listener_node', anonymous=True)
        rospy.Subscriber('/masked_voxel', Int8MultiArray, self.voxel_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)

        self.load_model("../configs/env_collision.pkl")
        self.lock = threading.Lock()
        self.joint = None
        self.voxel = None


    def voxel_callback(self, data: Int8MultiArray):
        # 메시지로부터 배열의 크기 복원
        dim_x = data.layout.dim[0].size
        dim_y = data.layout.dim[1].size
        dim_z = data.layout.dim[2].size
        
        # 수신된 1차원 데이터를 3차원 배열로 변환
        with self.lock:
            self.voxel = np.array(data.data, dtype=np.float32).reshape((dim_x, dim_y, dim_z))
            # self.voxel = np.zeros([dim_x, dim_y, dim_z], dtype=np.float32)

    def joint_callback(self, data: JointState):
        conf_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        with self.lock:
            self.joint = np.array([
                data.position[joint_idx] 
                for joint_idx, joint_name in enumerate(data.name) 
                if joint_name in conf_joint_names
            ], dtype=np.float32)


    def load_model(self, file_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model = EnvCollNet(dof=7).to(self.device)
        model_state_dict = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(model_state_dict)

    def model_forward(self):
        if self.joint is not None and self.voxel is not None:
            with self.lock:
                x_q = torch.from_numpy(self.joint.reshape(1, -1)).to(self.device).requires_grad_(True)
                x_occ = torch.from_numpy(self.voxel.reshape(1, 1, 36, 36, 36)).to(self.device)
            # with torch.no_grad():
            self.model.eval()

            x_q = x_q.repeat(N, 1)
            start = time.time()
            x_occ = x_occ.repeat(N, 1, 1, 1, 1)
            NN_output = self.model(x_q, x_occ)

            # Define a lambda function that calculates the output with respect to x_q
            model_output_fn = lambda x_q: self.model(x_q, x_occ)

            # Calculate the Jacobian of the model's output with respect to x_q
            jacobian = F.jacobian(model_output_fn, x_q)

            env_min_dist_pred = NN_output.cpu().detach().numpy()
            jacobian_pred = jacobian.cpu().detach().numpy()
            print(f'hz: {1/(time.time() - start)}')

            
            return env_min_dist_pred, jacobian_pred
        return None, None

    

if __name__ == '__main__':
    m = env_dist_check()
    rate = rospy.Rate(10)  # 10Hz 루프
    try:
        while not rospy.is_shutdown():
            ans, jac = m.model_forward()
            if ans is not None:
                print(ans.shape)
                print(f'env_pred: {ans[0]} cm')
                print(jac.shape)
                print(jac[0,0,0])
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
