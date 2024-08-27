from __future__ import division
import os
import time
import torch
import argparse
import pickle
import numpy as np
from self_collision_model_ver2 import SelfCollNet
import datetime as dt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import wandb

"""
This version predict virtual probability to be collide.
input: normalized joint angle(q)
output: virtual probability to be collide (p1, p2); [p1-p2 < 0] means collision
"""

class CollisionNetDataset(Dataset):
    """
    data pickle contains dict
        'q'           : joint angle
        'min_dist'    : minimum distance btw robot links
    Output data
        'normalized_q': normalized joint angle
        'coll'        : whether robot is collision(1) or free(0)
    """

    def __init__(self, file_name,):
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
            self.q = dataset['q']
            self.min_dist = dataset['min_dist']
        
        joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],  # min 
                                [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]]) # max

        # Normalizing q values
        self.normalized_q = (self.q - joint_limit[0]) / (joint_limit[1] - joint_limit[0])

        # Updating min_dist values
        self.coll = np.where(self.min_dist <= 1, 1, 0).astype(np.int64)

        print('normalized_q shape: ', self.normalized_q.shape)
        print('coll shape: ', self.coll.shape)

    def __len__(self):
        return len(self.min_dist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]

        return torch.tensor(self.normalized_q[idx], dtype=torch.float32), torch.tensor(self.coll[idx], dtype=torch.int64)



def main(args):
    file_name = "../data_generator/self_data/2024_07_31_01_31_52/dataset.pickle"
    train_ratio = 0.99
    validation_ratio = 0.005
    test_ratio = 1 - (train_ratio + validation_ratio)
    
    date = dt.datetime.now()
    data_dir = "{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}/".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    log_dir = 'log/self_collsion_ver2/' + data_dir
    chkpt_dir = 'model/checkpoints/self_collsion_ver2/' + data_dir
    model_dir = 'model/self_collsion_ver2/' + data_dir

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    suffix = 'rnd{}'.format(args.seed)

    log_file_name = log_dir + 'log_{}'.format(suffix)
    model_name = '{}'.format(suffix)

    wandb.init(project='Husky-Panda Collision detection')
    wandb.run.name = data_dir
    wandb.run.save()
    wandb.config.update(args)


    """
    layer size = [7(joint angle), hidden1, hidden2, , ..., 2(collision or not)]
    """
    layer_size = [7, 50, 30, 10, 2]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print('loading data ...')
    read_time = time.time()
    dataset = CollisionNetDataset(file_name=file_name)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(validation_ratio * total_size)
    test_size = total_size - train_size - val_size
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
    
    import torch.nn.functional as F
    def loss_fn_fc(y_hat, y):
        loss = F.cross_entropy(y_hat, y)
        return loss
    

    collnet = SelfCollNet(fc_layer_sizes=layer_size,
                          batch_size=args.batch_size,
                          device=device,
                          nerf=False).to(device)
    print(collnet)

    optimizer = torch.optim.Rprop(collnet.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                           threshold=0.01, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-04, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # clear log
    with open(log_file_name, 'w'):
        pass

    min_loss = 1e100
    e_notsaved = 0

    for normalized_q, coll in val_data_loader:
        val_normalized_q, val_coll = normalized_q.to(device, dtype=torch.float32).squeeze(), coll.to(device, dtype=torch.int64).squeeze()
    for normalized_q, coll in test_data_loader:
        test_normalized_q, test_coll = normalized_q.to(device, dtype=torch.float32).squeeze(), coll.to(device, dtype=torch.int64).squeeze()


    for epoch in range(args.epochs):
        loader_tqdm = tqdm.tqdm(train_data_loader)

        # for training
        for normalized_q, coll in loader_tqdm:
            train_normalized_q, train_coll = normalized_q.to(device, dtype=torch.float32).squeeze(), coll.to(device, dtype=torch.int64).squeeze()
            
            collnet.train()
            with torch.cuda.amp.autocast():
                train_coll_hat = collnet.forward(train_normalized_q).squeeze()
                loss_fc_train = loss_fn_fc(train_coll_hat, train_coll)
                loss_train = loss_fc_train

            scaler.scale(loss_train).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # for validation
        collnet.eval()
        with torch.cuda.amp.autocast():
            val_coll_hat = collnet.forward(val_normalized_q).squeeze()
            loss_fc_val = loss_fn_fc(val_coll_hat, val_coll)
            loss_val = loss_fc_val
        scheduler.step(loss_fc_val)


        from sklearn.metrics import accuracy_score, confusion_matrix

        # for test
        collnet.eval()
        with torch.cuda.amp.autocast():
            test_coll_hat = collnet.forward(test_normalized_q).squeeze()

            # Compute predictions
            pred_diff = test_coll_hat[:, 0] - test_coll_hat[:, 1]
            pred_labels = (pred_diff < 0).long()

            # Compute accuracy
            accuracy = accuracy_score(test_coll.cpu().numpy(), pred_labels.cpu().numpy())

            # Compute confusion matrix
            TN, FP, FN, TP = confusion_matrix(test_coll.cpu().numpy(), pred_labels.cpu().numpy()).ravel()
            TPR = TP / (TP + FN)
            TNR = TN / (TN + FP)

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
        print("[Test] Accuracy    : {:.3f}".format(accuracy))
        print("[Test] TPR         : {:.3f}".format(TPR))
        print("[Test] TNR         : {:.3f}".format(TNR))
        print("=========================================================================================")

        wandb.log({"NLL loss":{
                                "Training loss": loss_train,
                                "Validation loss": loss_val,
                                },
                   "Test metrics":{
                                "Accuracy": accuracy,
                                "TPR": TPR,
                                "TNR": TNR,
                                }
                   })

        with open(log_file_name, 'a') as f:
            f.write("Epoch: {} (Saved at {}) / Train Loss: {} / Valid Loss: {} / Test TPR: {} / Test TNR: {} / Test TP: {} / Test TN: {} / Test FP: {} / Test FN: {}\n".format(epoch,
                                                                                                                                                                               epoch - e_notsaved,
                                                                                                                                                                               loss_train,
                                                                                                                                                                               loss_val,
                                                                                                                                                                               TPR,
                                                                                                                                                                               TNR,
                                                                                                                                                                               TP,
                                                                                                                                                                               TN,
                                                                                                                                                                               FP,
                                                                                                                                                                               FN))

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