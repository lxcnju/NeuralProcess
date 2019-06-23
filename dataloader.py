#-*- coding : utf-8 -*-

"""
数据生成
"""


import torch
from torch.utils.data import Dataset
from sklearn.gaussian_process.kernels import RBF
import numpy as np

class DatasetGP(Dataset):
    def __init__(self, n_tasks,
                       batch_size=32,
                       x_size=1,
                       y_size=1,
                       x_min=-2.0,
                       x_max=2.0,
                       l_scale_min=0.5,
                       l_scale_max=2.0,
                       alpha_scale_min=0.5,
                       alpha_scale_max=2.0,
                       n_context_min=3,
                       n_context_max=10,
                       n_target_min=50,
                       n_target_max=200):
        super().__init__()
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.x_size = x_size
        self.y_size = y_size
        self.x_min = x_min
        self.x_max = x_max
        self.l_scale_min = l_scale_min
        self.l_scale_max = l_scale_max
        self.alpha_scale_min = alpha_scale_min
        self.alpha_scale_max = alpha_scale_max
        self.n_context_min = n_context_min
        self.n_context_max = n_context_max
        self.n_target_min = n_target_min
        self.n_target_max = n_target_max

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, index):
        n_context = np.random.randint(self.n_context_min, self.n_context_max + 1)
        n_target = np.random.randint(self.n_target_min, self.n_target_max + 1)

        batch_context_x = []
        batch_context_y = []
        batch_target_x = []
        batch_target_y = []

        for _ in range(self.batch_size):
            length_scale = np.random.uniform(self.l_scale_min, self.l_scale_max)
            alpha_scale = np.random.uniform(self.alpha_scale_min, self.alpha_scale_max)
            kernel = alpha_scale * RBF(length_scale=length_scale)

            x = np.random.uniform(self.x_min, self.x_max, (n_context + n_target, self.x_size))
            mean = np.zeros(n_context + n_target)
            cov = kernel(x)
            y = np.random.multivariate_normal(mean, cov)

            context_x = x[0 : n_context, :]
            context_y = y[0 : n_context]

            target_x = x[n_context :, :]
            target_y = y[n_context :]

            batch_context_x.append(context_x)
            batch_context_y.append(context_y)

            batch_target_x.append(target_x)
            batch_target_y.append(target_y)

        batch_context_x = torch.FloatTensor(batch_context_x)
        batch_context_y = torch.FloatTensor(batch_context_y)
        batch_target_x = torch.FloatTensor(batch_target_x)
        batch_target_y = torch.FloatTensor(batch_target_y)

        return batch_context_x, batch_context_y, batch_target_x, batch_target_y

