import torch
import torch.nn as nn

"""
This version use Occupancy Voxel grid for input data.
input: joint angle(q), occupancy grid(occ)
output: minimum distance(d) [unit: cm]
"""
    
class EnvironmentEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(EnvironmentEncoder, self).__init__()

        # input size: batch, 1, 36, 36, 36
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2), # batch, 32, 16, 16, 16
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3), # batch, 32, 14, 14, 14
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        # input size: batch, 32, 14, 14, 14
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # batch, 64, 7, 7, 7
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # batch, 64, 7, 7, 7
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # batch, 128, 4, 4, 4
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # batch, 128, 4, 4, 4
            nn.BatchNorm3d(128),
        )

        self.linear_means = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

        self.linear_log_var = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        means = self.linear_means(x)
        log_var = self.linear_log_var(x)

        return means, log_var
    
class EnvironmentDecoder(nn.Module):
    def __init__(self,latent_dim):
        super(EnvironmentDecoder, self).__init__()

        self.headTrans = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128 * 4 * 4 * 4),
        )
        # input size: batch, 128, 4, 4, 4
        self.conv2Trans = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # batch, 128, 4, 4, 4
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1), # batch, 64, 7, 7, 7
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # batch, 64, 7, 7, 7
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,  output_padding=1), # batch, 32, 14, 14, 14
            nn.BatchNorm3d(32),
        )
        # input size : batch, 32, 14, 14, 14
        self.conv1Trans = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3), # batch, 32, 16, 16, 16
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=5, stride=2, output_padding=1), # batch, 1, 36, 36, 36
        )

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.headTrans(x)
        x = x.view(x.size(0), 128, 4, 4, 4)
        x = self.conv2Trans(x)
        x = self.conv1Trans(x)
        x = self.sigmoid(x)

        return x
    
class EnvironmentVAE(nn.Module):
    def __init__(self, latent_dim):
        super(EnvironmentVAE, self).__init__()
        
        self.latent_dim = latent_dim

        self.encoder = EnvironmentEncoder(self.latent_dim)
        self.decoder = EnvironmentDecoder(self.latent_dim)

    def forward(self, x):
        batch_size = x.size(0)

        means, log_var = self.encoder(x)
        
        device = self.encoder.linear_means[-1].weight.device

        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn([batch_size, self.latent_dim]).to(device)

        z = eps * std + means

        recon_x = self.decoder(z)

        return recon_x, means, log_var, z
    
    def loss(self, recon_x, x, means, log_var, beta=1.0):
        bs = x.size(0)
        BCE = torch.nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum') / bs
        KLD = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp()) / bs

        return beta * BCE +  KLD, BCE, KLD   # VAE

    def encode(self, x):
        z, _ = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    
class EnvCollNet(nn.Module):
    def __init__(self, dof, latent_dim, beta=1.0, mu=1.0):
        super(EnvCollNet, self).__init__()
        self.dof = dof
        self.latent_dim = latent_dim
        self.beta = beta
        self.mu = mu

        self.env_vae = EnvironmentVAE(self.latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim + self.dof, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # for bool
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
        recon_env, mean, log_var, z = self.env_vae(env)

        x_fc = torch.cat([conf, z], dim=1)
        y = self.fc(x_fc)

        return y, env, recon_env, mean, log_var
    
    def inference(self, conf, env):
        z = self.env_vae.encode(env)

        x_fc = torch.cat([conf,z], dim=1)
        y = self.fc(x_fc)
        
        return y
    
    def loss_mse(self, y, y_hat):
        """
        y    : true minimum distance (cm)
        y_hat: predictied minimum distance (cm)
        """
        return torch.nn.functional.mse_loss(y_hat.squeeze(), y.squeeze(), reduction="mean")
    
    def loss_fc(self, y, y_hat):
        """
        y: bool for collision (free: 0, 1: collide)
        """
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat.squeeze(), y.squeeze(), reduction="mean")

    def loss(self, env, recon_env, y, y_hat, mean, log_var, prefix=''):
        loss_vae, loss_bce, loss_kld = self.env_vae.loss(recon_env, env, mean, log_var, self.beta)
        # loss_mse = self.loss_mse(y, y_hat)
        # loss = loss_vae + self.mu * loss_mse
        loss_fc = self.loss_fc(y, y_hat)
        loss = loss_vae + self.mu * loss_fc
        
        losses = {prefix+'loss': loss, 
                  prefix+'loss_vae': loss_vae,
                  prefix+'loss_bce': loss_bce,
                  prefix+'loss_kld': loss_kld,
                  prefix+'loss_fc': loss_fc, }
                #   prefix+'loss_mse': loss_mse, }

        return losses

if __name__ == '__main__':
    from torchinfo import summary
    model = EnvCollNet(latent_dim=128, dof=7)
    summary(model, input_size=[(1, 7), (1, 1, 36, 36, 36)])