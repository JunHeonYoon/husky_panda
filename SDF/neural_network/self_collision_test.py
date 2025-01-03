from srmt.planning_scene import PlanningScene
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from SDF.neural_network.self_collision_model import SelfCollNet
import torch



# Parameters
joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],
                        [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]])

# Create Planning Scene
pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="base_linl")

# NN model load
date = "2024_09_10_15_58_59/"
model_file_name = "self_collision.pkl"

model_dir = "model/self_collision_ver1/" + date + model_file_name
device = torch.device('cpu')

model = SelfCollNet(
    fc_layer_sizes=[7, 256, 64, 1],
    batch_size=1,
    nerf=True,
    device=device).to(device)

model_state_dict = torch.load(model_dir, map_location=device)
model.load_state_dict(model_state_dict)


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



# q_set = np.loadtxt("/home/yoonjunheon/git/MPCC/C++/debug.txt")
# print(q_set.shape)

for iter in range(1,1000):
# for iter, q in enumerate(q_set):
    joint_state =  np.random.uniform(low=joint_limit[0], high=joint_limit[1], size=7)
    # joint_state =  qq
    print(joint_state)
    # joint_state =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    pc.display(joint_state)
    min_dist = pc.min_distance(joint_state)*100
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(joint_state.reshape(1, -1).astype(np.float32)).to(device)
        # jac = torch.autograd.functional.jacobian(model, x)
        NN_output = model(x)
    min_dist_pred = NN_output.cpu().detach().numpy().item()
    print("=================================")
    print(f'sel_true : {min_dist}')
    print(f'sel_pred : {min_dist_pred}')
    print("=================================")
    
    x_data.append(iter)
    y_data.append(min_dist)
    y_hat_data.append(min_dist_pred)
    

    plt_func(fig, line1, line2, x_data, y_data, y_hat_data)
    

    keyboard_input = input()
    if keyboard_input == '':
        pass
