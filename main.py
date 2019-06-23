#-*- coding : utf-8 -*-

from dataloader import DatasetGP
from cnp import ConditionalNeuralProcess, Criterion

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    n_epoches = 1000
    n_tasks = 200
    batch_size = 64
    x_size = 1
    y_size = 1
    z_size = 128
    lr = 1e-4

    dataset = DatasetGP(n_tasks=n_tasks, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=1)
    model = ConditionalNeuralProcess(x_size=x_size, y_size=y_size, z_size=z_size)
    criterion = Criterion()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epoches):
        for i, (cx, cy, tx, ty) in enumerate(dataloader):
            cx = torch.squeeze(cx, dim=0)      # (bs, n_context, x_size)
            cy = torch.squeeze(cy, dim=0)      # (bs, n_context)
            tx = torch.squeeze(tx, dim=0)      # (bs, n_target, x_size)
            ty = torch.squeeze(ty, dim=0)      # (bs, n_target)

            cy = cy.unsqueeze(dim=-1)          # (bs, n_context, 1)

            mu, sigma = model(cx, cy, tx)      # (bs, n_target), (bs, n_target)

            loss = criterion(mu, sigma, ty)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 5 == 0:
                print("[Epoch : {}] [step : {}] [loss : {:.5f}]".format(epoch, i, loss.item()))

        context_x = cx[0].detach().numpy().reshape((-1, ))
        context_y = cy[0].detach().numpy().reshape((-1, ))
        target_x = tx[0].detach().numpy().reshape((-1, ))
        target_y = ty[0].detach().numpy().reshape((-1, ))
        mean_y = mu[0].detach().numpy().reshape((-1, ))
        var_y = sigma[0].detach().numpy().reshape((-1, ))

        plt.figure()
        plt.scatter(context_x, context_y, color="r", marker="o")
        plt.scatter(target_x, target_y, color="b", marker="x")
        index = np.argsort(target_x)
        target_x = target_x[index]
        mean_y = mean_y[index]
        var_y = var_y[index]
        plt.fill_between(target_x, mean_y-var_y, mean_y+var_y, alpha=0.2, facecolor="r", interpolate=True)
        plt.savefig("Epoch_{}.jpg".format(epoch))
        plt.close()


