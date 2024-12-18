
import torch
import numpy as np
import os
import pandas as pd


device = torch.device('cpu', 0)
tensor_args = {'device': device, 'dtype': torch.float32}

# NN model load
date = "2024_09_12_10_46_43/"
model_file_name = "self_collision.pkl"

model_dir = "model/self_collision_ver1/" + date + model_file_name
model = torch.load(model_dir)

n_layers = 3 # [7, 256, 64, 1]


for layer in range(n_layers):
    globals()['weight_{}'.format(str(layer))] = pd.DataFrame(model["fc.MLP.L{}.weight".format(layer)].cpu().numpy())
    globals()['bias_{}'.format(str(layer))] = pd.DataFrame(model["fc.MLP.L{}.bias".format(layer)].cpu().numpy())

if not os.path.exists('parameter'): os.makedirs('parameter')
for layer in range(n_layers):
    eval("weight_" + str(layer) + ".to_csv('parameter/weight_{}.txt'.format(layer), sep = ' ', index=False, header=False)")
    eval("bias_" + str(layer) + ".to_csv('parameter/bias_{}.txt'.format(layer), sep = ' ', index=False, header=False)")