#-*- coding : utf-8 -*-

"""
实现Conditional Neural Process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalNeuralProcess(nn.Module):
    def __init__(self, x_size, y_size, z_size):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.encoder = nn.Sequential(
            nn.Linear(x_size + y_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, z_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_size + x_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.softplus = nn.Softplus()

    def forward(self, context_x, context_y, target_x):
        """ context_x : (batch_size, n_context, x_size)
            context_y : (batch_size, n_context, y_size)
            target_x : (batch_size, n_target, x_size)
        """
        ### Encoder
        xy = torch.cat([context_x, context_y], dim=-1)      # (batch_size, n_context, x_size + y_size)
        bs, nc, xy_size = xy.shape

        xy = xy.view((bs * nc, xy_size))

        context_z = self.encoder(xy)                        # (batch_size * n_context, z_size)
        context_z = context_z.view((bs, nc, self.z_size))
        z = torch.mean(context_z, dim=1)                    # (bs, z_size)

        ### Decoder
        bs, nt, x_size = target_x.shape
        z = z.unsqueeze(dim=1).repeat((1, nt, 1))           # (bs, nt, z_size)

        z_tx = torch.cat([z, target_x], dim=-1)             # (bs, nt, z_size + x_size)
        z_tx = z_tx.view((bs * nt, self.z_size + x_size))
        out = self.decoder(z_tx)                            # (bs * nt, 2)
        out = out.view((bs, nt, 2))                         # (bs, nt, 2)

        mu = out[:, :, 0]
        log_sigma = out[:, :, 1]

        sigma = 0.1 + 0.9 * self.softplus(log_sigma)
        return mu, sigma

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma, target_y):
        """ mu : (bs, n_target)
            sigma : (bs, n_target)
            target_y : (bs, n_target)
        """
        loss = 0.0
        bs = mu.shape[0]
        for i in range(bs):
            dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu[i], covariance_matrix=torch.diag(sigma[i]))
            log_prob = dist.log_prob(target_y[i])
            loss = loss - 1.0 * torch.mean(log_prob)
        loss = loss / bs
        return loss


