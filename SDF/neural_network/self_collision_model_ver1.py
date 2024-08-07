import torch
import torch.nn as nn

"""
This version predict minimum distance btw robot itself.
input: joint angle(q)
output: minimum distance(d) [unit: cm]
"""

class FullyConnectedNet(nn.Module):
    def __init__(self, layer_sizes, batch_size, nerf):
        nn.Module.__init__(self)
        
        def init_weights(m):
            # print(m)
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                # print(m.weight)

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if nerf and i==0:
                in_size = in_size*3
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+2 < len(layer_sizes):
                self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(0.1))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())


        self.MLP.apply(init_weights)
        self.nerf = nerf

    def forward(self, x):
        if(self.nerf):
            x_nerf = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        else:
            x_nerf = x

        y = self.MLP(x_nerf)
        return y

class SelfCollNet(nn.Module):
    def __init__(self, fc_layer_sizes, batch_size, device, nerf=True):
        
        nn.Module.__init__(self)

        self.fc = FullyConnectedNet(fc_layer_sizes,batch_size, nerf=nerf)

    def forward(self, x_q):
        """
        x_q: input q(7)
        """
        y = self.fc(x_q)
        
        return y
        
