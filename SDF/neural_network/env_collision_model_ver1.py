import torch
import torch.nn as nn

"""
This version use Occupancy Voxel grid for input data.
input: joint angle(q), occupancy grid(occ)
output: minimum distance(d) [unit: cm]
"""

class ManipulationConvNet(nn.Module):
    def __init__(self, dof):
        super(ManipulationConvNet, self).__init__()
        self.dof = dof

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=2),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU()
        # )

        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(64),
        # )

        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(128),
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear((self.dof+2) * 128, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024)
        # )

        # self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(self.dof, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
        )

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
        # x = x.view(x.size(0), 1, x.size(1))
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class EnvironmentEncoder(nn.Module):
    def __init__(self):
        super(EnvironmentEncoder, self).__init__()

        # # input size: batch, 1, 36, 36, 36
        # self.conv1 = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2), # batch, 32, 16, 16, 16
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3), # batch, 32, 14, 14, 14
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(),
        # )

        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # batch, 64, 7, 7, 7
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # batch, 64, 7, 7, 7
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # batch, 128, 4, 4, 4
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # batch, 128, 4, 4, 4
        #     nn.BatchNorm3d(128),
        # )

        # input size: batch, 1, 36, 36, 36
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2), # batch, 32, 16, 16, 16
            nn.BatchNorm3d(32),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3), # batch, 32, 14, 14, 14
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            # nn.ReLU(),
        )

        self.avgpool = nn.AvgPool3d(kernel_size=2,stride=2) # batch, 32, 7, 7, 7

        self.head = nn.Sequential(
            nn.Linear(32 * 7 * 7 * 7, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )

        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # batch, 64, 7, 7, 7
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # batch, 64, 7, 7, 7
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # batch, 128, 4, 4, 4
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # batch, 128, 4, 4, 4
        #     nn.BatchNorm3d(128),
        # )

        # self.head = nn.Sequential(
        #     nn.Linear(128 * 4 * 4 * 4, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        # )



        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)

        return x

    
class EnvCollNet(nn.Module):
    def __init__(self, dof):
        super(EnvCollNet, self).__init__()
        self.dof = dof

        self.extractor_conf = ManipulationConvNet(self.dof)
        self.extractor_env = EnvironmentEncoder()

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
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

    def forward(self, conf, env):
        f_conf = self.extractor_conf(conf)
        f_env = self.extractor_env(env)
        f_env = f_env.view(f_env.size(0), -1)

        f = torch.cat([f_conf, f_env], dim=1)

        out = self.fc(f)

        return out
