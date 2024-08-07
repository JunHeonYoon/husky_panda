import torch
import torch.nn as nn

"""
This version use Depth image for input data.
input: joint angle(q), depth image(depth)
output: minimum distance(d) [unit: cm]
"""

class ManipulationConvNet(nn.Module):
    def __init__(self, dof):
        super(ManipulationConvNet, self).__init__()
        self.dof = dof

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
        )

        self.fc = nn.Sequential(
            nn.Linear((self.dof+2) * 128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EnvironmentEncoder(nn.Module):
    """Pytorch implementation of ResNet8 for processing the depth image.
    The output of the conv layers is flattened and compressed to a latent representation thr
    gh some FC layers.
    """
    def __init__(self, dropout_rate=0.2):
        super(EnvironmentEncoder, self).__init__()
        self.dropout_rate = dropout_rate
        # input size: batch, 1, 576, 640
        self.layers = torch.nn.ModuleDict({
            'input': torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=(5,5), stride=2, padding=(1,3)),
                torch.nn.MaxPool2d(kernel_size=(3,3), stride=2),
            ),
            'res_layer_1': torch.nn.Conv2d(32, 32, kernel_size=(1,1), stride=2, padding=0),
            'res_block_1': torch.nn.Sequential(
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
            ),
            'res_layer_2': torch.nn.Conv2d(32, 64, kernel_size=(1,1), stride=2, padding=0),
            'res_block_2': torch.nn.Sequential(
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1),
            ),
            'res_layer_3': torch.nn.Conv2d(64, 128, kernel_size=(1,1), stride=2, padding=0),
            'res_block_3': torch.nn.Sequential(
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            ),
            'output': torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.AvgPool2d(kernel_size=(3,3), stride=2),
                torch.nn.Flatten(),
                torch.nn.Linear(9216, 1024),
            )
        })

        self.init_conv_layer(self.layers)
    
    def init_conv_layer(self, layer):
        """Recursive function to initialize all conv2d layers with xavier_uniform_ througout the submodules."""
        if type(layer) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('conv2d'))
            torch.nn.init.zeros_(layer.bias)
        for ll in layer.children():
            self.init_conv_layer(ll)


    def forward(self, input):
        x = self.layers['input'](input)
        res = self.layers['res_layer_1'](x)
        x = self.layers['res_block_1'](x)
        x = x + res
        res = self.layers['res_layer_2'](x)
        x = self.layers['res_block_2'](x)
        x = x + res
        res = self.layers['res_layer_3'](x)
        x = self.layers['res_block_3'](x)
        x = x + res
        x = self.layers['output'](x)
        return x

    
class EnvCollNet(nn.Module):
    def __init__(self, dof):
        super(EnvCollNet, self).__init__()
        self.dof = dof

        self.extractor_conf = ManipulationConvNet(self.dof)
        self.extractor_env = EnvironmentEncoder()

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def load_pretrained(self, checkpoint):
    #     pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.extractor_env.state_dict()}
    #     self.extractor_env.load_state_dict(pretrained_dict)

    def get_env_feature(self, env):
        return self.extractor_env(env)

    def forward(self, conf, env):
        # if self.training is False:
        #     assert env.size(0) == 1, "You should pass only single environment map."
        #     env = env.repeat(conf.size(0), 1, 1, 1, 1)

        f_conf = self.extractor_conf(conf)
        # f_env = self.extractor_env(env)
        f_env = self.get_env_feature(env)
        # f_env = env
        f_env = f_env.view(f_env.size(0), -1)

        f = torch.cat([f_conf, f_env], dim=1)

        out = self.fc(f)

        return out
