from __future__ import division
import os
import time
import torch
import torch.nn as nn
import argparse
import pickle
import numpy as np
from SDF.neural_network.env_collision_model_ver2 import EnvCollNet
import datetime as dt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import wandb

"""
This version use Depth image for input data.
input: joint angle(q), depth image(depth)
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
        self.depth_shape = self.dataset[0]["depth"].shape
        # self.occ_shape = self.dataset[0]["occupancy"].shape

        print('Total number of data: ', self.num_env*self.num_q_per_env)
        print('Number of Env       : ', self.num_env)
        print('Number of q per Env : ', self.num_q_per_env)
        print('Depth shape         : ', self.depth_shape)
        # print('Occupancy shape     : ', self.occ_shape)

    def __len__(self):
        return self.num_env*self.num_q_per_env

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        
        env_idx = [i//self.num_q_per_env for i in idx]
        q_idx = [i%self.num_q_per_env for i in idx]

        return torch.tensor(np.array([self.dataset[env_idx[i]]["q"][q_idx[i]] for i in range(len(idx))]), dtype=torch.float32), \
               torch.tensor(np.array([self.dataset[i]["depth"] for i in env_idx]), dtype=torch.float32), \
               torch.tensor(np.array([self.dataset[env_idx[i]]["min_dist"][q_idx[i]] for i in range(len(idx))]), dtype=torch.float32)



def main(args):
    file_name = "../data_generator/env_data/2024_08_01_17_42_04/dataset.pickle"
    train_ratio = 0.9999
    test_ratio = 1 - train_ratio
    
    date = dt.datetime.now()
    data_dir = "{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}/".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    log_dir = 'log/env_collsion_ver2/' + data_dir
    chkpt_dir = 'model/checkpoints/env_collsion_ver2/' + data_dir
    model_dir = 'model/env_collsion_ver2/' + data_dir

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
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset))
    end_time = time.time()
    
    print('data load done. time took {0}'.format(end_time-read_time))
    print('[data len] total: {} train: {}, test: {}'.format(len(dataset), len(train_dataset), len(test_dataset)))
    
    class RankingLoss(nn.Module):
        def __init__(self, margin):
            super(RankingLoss, self).__init__()
            self.margin = margin

            if self.margin is None:
                self.margin_loss = nn.SoftMarginLoss()
            else:
                self.margin_loss = nn.MarginRankingLoss(margin=self.margin)

        def forward(self, preds, y):
            assert len(preds) % 2 == 0, 'the batch size is not even.'

            preds_i = preds[:preds.size(0) // 2]
            preds_j = preds[preds.size(0) // 2:]
            y_i = y[:y.size(0) // 2]
            y_j = y[y.size(0) // 2:]
            labels = torch.sign(y_i - y_j)

            if self.margin is None:
                return self.margin_loss(preds_i-preds_j, labels)
            else:
                return self.margin_loss(preds_i, preds_j, labels)
        
    class RankingAccuracy(nn.Module):
        def __init__(self):
            super(RankingAccuracy, self).__init__()

        def forward(self, preds, y, size_average=True):
            n = preds.size(0)
            gt_diff_mat = y.expand(n, n) - y.expand(n, n).t()
            gt_comparison = torch.sign(gt_diff_mat)
            pred_diff_mat = preds.expand(n, n) - preds.expand(n, n).t()
            pred_comparison = torch.sign(pred_diff_mat)
            acc_mat = (gt_comparison == pred_comparison) + (gt_comparison == 0)
            m = (gt_comparison == 0).sum().float()
            acc_mat = torch.sign(acc_mat).float()
            acc = (acc_mat.sum() - m) / (n * n - m)

            return acc
            
    mse_criterion = torch.nn.MSELoss()
    ranking_criterion = RankingLoss(margin=0.0)
    accuracy_criterion = RankingAccuracy()
    
    collnet = EnvCollNet(dof=7).to(device)
    # import torchsummary
    # torchsummary.summary(collnet, (1,576, 640))
    print(collnet)

    optimizer = torch.optim.Adam(collnet.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # clear log
    with open(log_file_name, 'w'):
        pass

    min_loss = 1e100
    e_notsaved = 0

    for q, depth, min_dist in test_data_loader:
        test_q, test_depth, test_min_dist = q.to(device, dtype=torch.float32).squeeze(), depth.to(device, dtype=torch.float32), min_dist.to(device, dtype=torch.float32).squeeze()

    for epoch in range(args.epochs):
        loader_tqdm = tqdm.tqdm(train_data_loader)

        # for training
        for q, depth, min_dist in loader_tqdm:
            train_q, train_depth, train_min_dist = q.to(device, dtype=torch.float32).squeeze(), depth.to(device, dtype=torch.float32), min_dist.to(device, dtype=torch.float32).squeeze()

            collnet.train()
            with torch.cuda.amp.autocast():
                train_min_dist_pred = collnet.forward(train_q, train_depth).squeeze()
                train_ranking_loss = mse_criterion(train_min_dist_pred, train_min_dist)
                train_mse_loss = ranking_criterion(train_min_dist_pred, train_min_dist)
                train_loss = train_mse_loss + args.mu * train_ranking_loss

            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()


        # for test
        collnet.eval()
        with torch.cuda.amp.autocast():
            test_min_dist_pred = collnet.forward(test_q, test_depth).squeeze()
            test_ranking_loss = mse_criterion(test_min_dist_pred, test_min_dist)
            test_mse_loss = ranking_criterion(test_min_dist_pred, test_min_dist)
            test_loss = test_mse_loss + args.mu * test_ranking_loss
            acc = accuracy_criterion(test_min_dist_pred, test_min_dist)

        if epoch == 0:
            min_loss = test_loss

        scheduler.step(test_loss)

        if test_loss < min_loss:
            e_notsaved = 0
            print('saving model', test_loss.item())
            checkpoint_model_name = chkpt_dir + 'loss_{}_{}_checkpoint_{:02d}_{}_self'.format(test_loss.item(), model_name, epoch, args.seed) + '.pkl'
            torch.save(collnet.state_dict(), os.path.join(model_dir, "env_collision.pkl"))
            torch.save(collnet.state_dict(), checkpoint_model_name)
            min_loss = test_loss
        print("Epoch: {} (Saved at {})".format(epoch, epoch-e_notsaved))
        print("[Train] MSE loss    : {:.3f}".format(train_mse_loss.item()))
        print("[Train] Ranking loss: {:.3f}".format(train_ranking_loss.item()))
        print("[Test]  MSE loss    : {:.3f}".format(test_mse_loss.item()))
        print("[Test]  Ranking loss: {:.3f}".format(test_ranking_loss.item()))
        print("[Test]  Ranking acc : {:.3f}".format(acc.item()))
        print("=========================================================================================")

        wandb.log({"Total loss":{
                        "Training loss": train_loss,
                        "Validation loss": test_loss,
                        },
                   "MSE loss":{
                        "Training loss": train_mse_loss,
                        "Validation loss": test_mse_loss,
                        },
                    "Ranking loss":{
                        "Training loss": train_ranking_loss,
                        "Validation loss": test_ranking_loss,
                        },
                    "Ranking accuracy":{
                        "Validation accuracy": acc
                    }
                   })

        with open(log_file_name, 'a') as f:
            f.write("Epoch: {} (Saved at {}) / Train total Loss: {} / Train MSE loss: {} / Train Ranking loss: {} / Test total Loss: {} / Test MSE loss: {} / Test Ranking loss: {} / Test Ranking accuracy\n".format(epoch,
                                                                                                                                                                                                                     epoch - e_notsaved,
                                                                                                                                                                                                                     train_loss,
                                                                                                                                                                                                                     train_mse_loss,
                                                                                                                                                                                                                     train_ranking_loss,
                                                                                                                                                                                                                     test_loss,
                                                                                                                                                                                                                     test_mse_loss,
                                                                                                                                                                                                                     test_ranking_loss,
                                                                                                                                                                                                                     acc))

        e_notsaved += 1
    torch.save





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument('--mu', type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)