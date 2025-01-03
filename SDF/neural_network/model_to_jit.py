import torch
from SDF.neural_network.env_collision_model import EnvCollNet

# NN model load
date = "2024_08_02_17_50_33/"
model_file_name = "env_collision.pkl"

model_dir = "model/env_collsion_ver1/" + date + model_file_name
device = torch.device('cuda')

model = EnvCollNet(dof=7).to(device)

model_state_dict = torch.load(model_dir, map_location=device)
model.load_state_dict(model_state_dict)
model.eval()

dummy_x_q = torch.randn(1,7).to(device)
dummy_x_occ = torch.randn(1,1,36,36,36).to(device)
traced_model = torch.jit.trace(model, (dummy_x_q, dummy_x_occ))

torch.jit.save(traced_model, "model/env_collsion_ver1/" + date+"env_collision.pt")