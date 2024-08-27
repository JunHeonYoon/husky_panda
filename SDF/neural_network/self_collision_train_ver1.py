from __future__ import division
import os
import time
import torch
import argparse
import pickle
import numpy as np
from self_collision_model_ver1 import SelfCollNet
import datetime as dt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import wandb

"""
This version predict minimum distance btw robot itself.
input: joint angle(q)
output: minimum distance(d) [unit: cm]
"""

class CollisionNetDataset(Dataset):
    """
    data pickle contains dict
        'q'          : joint angle
        'min_dist'   : minimum distance btw robot links
    """
    def __init__(self, file_name,):
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
            self.q = dataset['q']
            self.min_dist = dataset['min_dist']
        print('q shape: ', self.q.shape)
        print('min_dist shape: ', self.min_dist.shape)

    def __len__(self):
        return len(self.min_dist)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
    
        return np.array(self.q[idx],dtype=np.float32), np.array(self.min_dist[idx],dtype=np.float32)

def main(args):
    file_name = "../data_generator/self_data/2024_07_30_15_46_23/dataset.pickle"
    train_ratio = 0.99
    validation_ratio = 0.005
    test_ratio = 1 - (train_ratio + validation_ratio)
    
    date = dt.datetime.now()
    data_dir = "{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}/".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    log_dir = 'log/self_collsion_ver1/' + data_dir
    chkpt_dir = 'model/checkpoints/self_collsion_ver1/' + data_dir
    model_dir = 'model/self_collsion_ver1/' + data_dir

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    suffix = 'rnd{}'.format(args.seed)


    log_file_name = log_dir + 'log_{}'.format(suffix)
    model_name = '{}'.format(suffix)

    wandb.init(project='Husky-Panda Minimum distance Regression')
    wandb.run.name = data_dir
    wandb.run.save()
    wandb.config.update(args)


    """
    layer size = [7(joint angle), hidden1, hidden2, , ..., 1(mininum dist)]
    """
    layer_size = [7, 256, 64, 1]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('loading data ...')
    read_time = time.time()
    dataset = CollisionNetDataset(file_name=file_name)
    train_size = int(train_ratio * len(dataset))
    val_size = int(validation_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(
        dataset=val_dataset, batch_size=len(val_dataset))
    test_data_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset))
    end_time = time.time()
    
    print('data load done. time took {0}'.format(end_time-read_time))
    print('[data len] total: {} train: {}, test: {}'.format(len(dataset), len(train_dataset), len(test_dataset)))
    

    def loss_fn_fc(y_hat, y):
        ALL_MSE = torch.nn.functional.mse_loss(y_hat, y, reduction="mean")
        # free_mask = (y > 5)
        # FREE_MSE = torch.nn.functional.mse_loss(y_hat[free_mask], y[free_mask], reduction="mean") if free_mask.any() else 0
        close_mask = (y < 5) & (y > 0)
        CLOSE_MSE = torch.nn.functional.mse_loss(y_hat[close_mask], y[close_mask], reduction="mean") if close_mask.any() else 0
        # coll_mask = (y < 0)
        # COLL_MSE = torch.nn.functional.mse_loss(y_hat[coll_mask], y[coll_mask], reduction="mean") if coll_mask.any() else 0

        return ALL_MSE + CLOSE_MSE
    

    collnet = SelfCollNet(fc_layer_sizes=layer_size,
                          batch_size=args.batch_size,
                          device=device,
                          nerf=True).to(device)
    print(collnet)

    optimizer = torch.optim.Adam(collnet.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                           threshold=0.01, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-04, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # clear log
    with open(log_file_name, 'w'):
        pass

    min_loss = 1e100
    e_notsaved = 0

    for q, min_dist in val_data_loader:
        test_q, test_min_dist = q.to(device).squeeze(), min_dist.to(device).squeeze()
    for q, min_dist in test_data_loader:
        val_q, val_min_dist = q.to(device).squeeze(), min_dist.to(device).squeeze()

    for epoch in range(args.epochs):
        loader_tqdm = tqdm.tqdm(train_data_loader)
        
        # for training
        for q, min_dist in loader_tqdm:
            train_q, train_min_dist = q.to(device).squeeze(),  min_dist.to(device).squeeze()
            
            collnet.train()
            with torch.cuda.amp.autocast():
                min_dist_hat= collnet.forward(train_q)
                min_dist_hat = min_dist_hat.squeeze()
                loss_fc_train = loss_fn_fc(min_dist_hat, train_min_dist)
                loss_train = loss_fc_train

            scaler.scale(loss_train).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


        # for validation
        collnet.eval()
        with torch.cuda.amp.autocast():
            min_dist_hat = collnet.forward(val_q)
            min_dist_hat = min_dist_hat.squeeze()
            loss_fc_val = loss_fn_fc(min_dist_hat, val_min_dist)
            loss_val = loss_fc_val
        scheduler.step(loss_fc_val)

        # for test
        collnet.eval()
        with torch.cuda.amp.autocast():
            min_dist_hat = collnet.forward(test_q)
            min_dist_hat = min_dist_hat.squeeze()

            overall_rmse = torch.sqrt(torch.nn.functional.mse_loss(min_dist_hat, test_min_dist))

            mask_free = test_min_dist > 5
            rmse_free = torch.sqrt(torch.nn.functional.mse_loss(min_dist_hat[mask_free], test_min_dist[mask_free]))

            mask_close = (test_min_dist <= 5) & (test_min_dist >= 0)
            rmse_close = torch.sqrt(torch.nn.functional.mse_loss(min_dist_hat[mask_close], test_min_dist[mask_close]))

            mask_col = test_min_dist < 0
            rmse_col = torch.sqrt(torch.nn.functional.mse_loss(min_dist_hat[mask_col], test_min_dist[mask_col]))
        

        if epoch == 0:
            min_loss = loss_val

        scheduler.step(loss_val)

        if loss_val < min_loss:
            e_notsaved = 0
            print('saving model', loss_val.item())
            checkpoint_model_name = chkpt_dir + 'loss_{}_{}_checkpoint_{:02d}_{}_self'.format(loss_val.item(), model_name, epoch, args.seed) + '.pkl'
            torch.save(collnet.state_dict(), os.path.join(model_dir, "self_collision.pkl"))
            torch.save(collnet.state_dict(), checkpoint_model_name)
            min_loss = loss_val
        print("Epoch: {} (Saved at {})".format(epoch, epoch-e_notsaved))
        print("[Train] fc loss    : {:.3f}".format(loss_train.item()))
        print("[Valid] fc loss    : {:.3f}".format(loss_val.item()))
        print("[Test] RMSE (all)  : {:.3f}".format(overall_rmse.item()))
        print("[Test] RMSE (free) : {:.3f}".format(rmse_free.item()))
        print("[Test] RMSE (close): {:.3f}".format(rmse_close.item()))
        print("[Test] RMSE (col)  : {:.3f}".format(rmse_col.item()))
        print("min_dist           : {}".format(test_min_dist.detach().cpu().numpy()[:4]))
        print("min_dist_hat       : {}".format(min_dist_hat.detach().cpu().numpy()[:4]))
        print("=========================================================================================")

        wandb.log({"MSE loss":{
                                "Training loss": loss_train,
                                "Validation loss": loss_val,
                                },
                   "Test RMSE loss":{
                                "Over all": overall_rmse,
                                "Free": rmse_free,
                                "Close": rmse_close,
                                "Collision": rmse_col,
                                }
                   })

        with open(log_file_name, 'a') as f:
            f.write("Epoch: {} (Saved at {}) / Train Loss: {} / Valid Loss: {} / Test RMSE(all): {} / Test RMSE(free): {} / Test RMSE(close): {} / Test RMSE(col): {}\n".format(epoch,
                                                                                                                                                                            epoch - e_notsaved,
                                                                                                                                                                            loss_train,
                                                                                                                                                                            loss_val,
                                                                                                                                                                            overall_rmse,
                                                                                                                                                                            rmse_free,
                                                                                                                                                                            rmse_close,
                                                                                                                                                                            rmse_col))

        e_notsaved += 1
    torch.save





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2000000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    
    args = parser.parse_args()
    main(args)