from __future__ import division
import os
import time
import torch
import torch.nn as nn
import argparse
import pickle
import numpy as np
from env_collision_model import EnvCollNet
import datetime as dt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import wandb

"""
This version predict minimum distance btw robot itself.
input: joint angle(q), occupancy grid(occ)
output: minimum distance(d) [unit: cm]
"""

class CollisionNetDataset(Dataset):
    """
    data pickle contains data list 
        data = [env_data1, env_data2, ...]
    env_data is dict which contains
        'env_idx'  : index of environment (int)
        'q'        : joint angle (np.array, shape: num_q*dof)
        'min_dist' : minimum distance (np.array, shape: num_q)
        'depth'    : depth image (np.array, shape: 576*640)
        'occupancy': Occupancy Voxel (np.array, shape: 36*36*36)
    """

    def __init__(self, file_name,):
        with open(file_name, 'rb') as f:
            self.dataset = pickle.load(f)
        self.num_env = len(self.dataset)
        self.num_q_per_env = self.dataset[0]["q"].shape[0]
        self.dof = self.dataset[0]["q"].shape[1]
        self.voxel_shape = self.dataset[0]["occupancy"].shape

        print('Total number of data: ', self.num_env*self.num_q_per_env)
        print('Number of Env       : ', self.num_env)
        print('Number of q per Env : ', self.num_q_per_env)
        print('Occupancy shape     : ', self.voxel_shape)

    def __len__(self):
        return self.num_env*self.num_q_per_env

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        
        env_idx = [i//self.num_q_per_env for i in idx]
        q_idx = [i%self.num_q_per_env for i in idx]

        q = torch.tensor(np.array([self.dataset[env_idx[i]]["q"][q_idx[i]] for i in range(len(idx))]), dtype=torch.float32)
        voxel = torch.tensor(np.array([self.dataset[i]["occupancy"] for i in env_idx]), dtype=torch.float32)
        min_dist = torch.tensor(np.array([self.dataset[env_idx[i]]["min_dist"][q_idx[i]] for i in range(len(idx))]), dtype=torch.float32)
        min_dist_label = (min_dist < 1).float() # free(larger than 1 cm):0, collision: 1
        return q, voxel, min_dist, min_dist_label


def main(args):
    file_name = "../data_generator/env_data/2025_01_02_22_00_27/dataset.pickle"
    train_ratio = 0.999
    val_ratio = 0.0005
    test_ratio = 1 - (train_ratio + val_ratio)
    
    date = dt.datetime.now()
    data_dir = "{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}/".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    log_dir = 'log/env_collsion/' + data_dir
    chkpt_dir = 'model/checkpoints/env_collsion/' + data_dir
    model_dir = 'model/env_collsion/' + data_dir

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    suffix = 'rnd{}'.format(args.seed)

    log_file_name = log_dir + 'log_{}'.format(suffix)
    model_name = '{}'.format(suffix)

    wandb.init(project='Panda env collision')
    wandb.run.name = data_dir
    wandb.run.save()
    wandb.config.update(args)


    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('loading data ...')
    read_time = time.time()
    dataset = CollisionNetDataset(file_name=file_name)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(
        dataset=val_dataset, batch_size=len(val_dataset))
    test_data_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset))
    end_time = time.time()
    
    print('data load done. time took {0}'.format(end_time-read_time))
    print('[data len] total: {} train: {}, val: {}, test: {}'.format(len(dataset), len(train_dataset), len(val_dataset), len(test_dataset)))
    
    collnet = EnvCollNet(dof=7, latent_dim=256, mu=args.mu, beta=args.beta).to(device)
    print(collnet)

    optimizer = torch.optim.Adam(collnet.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # clear log
    with open(log_file_name, 'w'):
        pass

    min_loss = 1e100
    e_notsaved = 0

    for q, voxel, _, label in val_data_loader:
        val_q, val_voxel, val_label = q.to(device, dtype=torch.float32).squeeze(), voxel.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32).squeeze()

    for q, voxel, _, label in test_data_loader:
        test_q, test_voxel, test_label = q.to(device, dtype=torch.float32).squeeze(), voxel.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32).squeeze()

    for epoch in range(args.epochs):
        loader_tqdm = tqdm.tqdm(train_data_loader)

        # for training
        for q, voxel, _, label in loader_tqdm:
            train_q, train_voxel, train_label = q.to(device, dtype=torch.float32).squeeze(), voxel.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32).squeeze()

            collnet.train()
            with torch.cuda.amp.autocast():
                train_label_pred, train_voxel, train_recon_voxel, mean, log_var = collnet.forward(train_q, train_voxel)
                train_losses = collnet.loss(env=train_voxel,
                                            recon_env=train_recon_voxel, 
                                            y=train_label,
                                            y_hat=train_label_pred,
                                            mean=mean,
                                            log_var=log_var,
                                            prefix="train_")

            optimizer.zero_grad()
            scaler.scale(train_losses["train_loss"]).backward()
            scaler.step(optimizer)
            scaler.update()


        # for validation
        collnet.eval()
        with torch.cuda.amp.autocast():
            val_label_pred, val_voxel, val_recon_voxel, mean, log_var = collnet.forward(val_q, val_voxel)
            val_losses = collnet.loss(env=val_voxel,
                                        recon_env=val_recon_voxel, 
                                        y=val_label,
                                        y_hat=val_label_pred,
                                        mean=mean,
                                        log_var=log_var,
                                        prefix="val_")
            
        # for test
        collnet.eval()
        with torch.cuda.amp.autocast():
            test_label_hat = collnet.inference(test_q, test_voxel)
            test_loss_fc = collnet.loss_fc(test_label, test_label_hat)

        if epoch == 0:
            min_loss = val_losses["val_loss"]

        scheduler.step()

        if val_losses["val_loss"] < min_loss:
            e_notsaved = 0
            print('saving model', val_losses["val_loss"].item())
            checkpoint_model_name = chkpt_dir + 'loss_{}_{}_checkpoint_{:02d}_{}_self'.format(val_losses["val_loss"].item(), model_name, epoch, args.seed) + '.pkl'
            torch.save(collnet.state_dict(), os.path.join(model_dir, "env_collision.pkl"))
            torch.save(collnet.state_dict(), checkpoint_model_name)
            min_loss = val_losses["val_loss"]
        print("Epoch: {} (Saved at {})".format(epoch, epoch-e_notsaved))
        print("[Train] Total loss : {:.3f}".format(train_losses["train_loss"].item()))
        print("[Train] FC loss    : {:.3f}".format(train_losses["train_loss_fc"].item()))
        print("[Train] VAE loss   : {:.3f}".format(train_losses["train_loss_vae"].item()))
        print("[Train] BCE loss   : {:.3f}".format(train_losses["train_loss_bce"].item()))
        print("[Train] KLD loss   : {:.3f}".format(train_losses["train_loss_kld"].item()))
        print("[Valid] Total loss : {:.3f}".format(val_losses["val_loss"].item()))
        print("[Valid] FC loss    : {:.3f}".format(val_losses["val_loss_fc"].item()))
        print("[Valid] VAE loss   : {:.3f}".format(val_losses["val_loss_vae"].item()))
        print("[Valid] BCE loss   : {:.3f}".format(val_losses["val_loss_bce"].item()))
        print("[Valid] KLD loss   : {:.3f}".format(val_losses["val_loss_kld"].item()))
        print("[Test]  FC loss    : {:.3f}".format(test_loss_fc.item()))
        print("=========================================================================================")

        wandb.log({"Train loss":{
                        "total loss": train_losses["train_loss"],
                        "fc loss": train_losses["train_loss_fc"],
                        "vae loss": train_losses["train_loss_vae"],
                        "bce loss": train_losses["train_loss_bce"],
                        "kld loss": train_losses["train_loss_kld"],
                        },
                   "Val loss":{
                        "total loss": val_losses["val_loss"],
                        "fc loss": val_losses["val_loss_fc"],
                        "vae loss": val_losses["val_loss_vae"],
                        "bce loss": val_losses["val_loss_bce"],
                        "kld loss": val_losses["val_loss_kld"],
                        },
                    "Test loss":{
                        "fc loss": test_loss_fc,
                        },
                   })

        with open(log_file_name, 'a') as f:
            f.write("Epoch: {} (Saved at {}) / Train total Loss: {} / Train FC loss: {} / Train VAE loss: {} / Train BCE loss: {} / Train KLD loss: {} / Valid total Loss: {} / Valid FC loss: {} / Valid VAE loss: {} / Valid BCE loss: {} / Valid KLD loss: {} / Test FC loss\n".format(epoch,
                                                                                                                                                                                                                                                                                          epoch - e_notsaved,
                                                                                                                                                                                                                                                                                          train_losses["train_loss"],
                                                                                                                                                                                                                                                                                          train_losses["train_loss_fc"],
                                                                                                                                                                                                                                                                                          train_losses["train_loss_vae"],
                                                                                                                                                                                                                                                                                          train_losses["train_loss_bce"],
                                                                                                                                                                                                                                                                                          train_losses["train_loss_kld"],
                                                                                                                                                                                                                                                                                          val_losses["val_loss"],
                                                                                                                                                                                                                                                                                          val_losses["val_loss_fc"],
                                                                                                                                                                                                                                                                                          val_losses["val_loss_vae"],
                                                                                                                                                                                                                                                                                          val_losses["val_loss_bce"],
                                                                                                                                                                                                                                                                                          val_losses["val_loss_kld"],
                                                                                                                                                                                                                                                                                          test_loss_fc))

        e_notsaved += 1
    torch.save





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--mu', type=float, default=10)
    
    args = parser.parse_args()
    main(args)