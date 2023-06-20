import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import struct
from datetime import date, datetime
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
# import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import random
import datetime  # <1>
import torch.optim as optim
from cmath import nan
random.seed(0)
np.random.seed(0)


def vwp2PutPul(v, w, payload, μ=1, Lx=0.2775, Ly=0.26, unit_weight=5.8, m=50, g=9.8):
    M = payload * unit_weight + m
    r = v / w if w != 0 else nan
    P_ul = μ * M * g * v if w == 0 else 0
    wheel_v1, wheel_v2 = np.hypot((r - Lx), Ly) * w if w != 0 else v, np.hypot((r - Lx), Ly) * w if w != 0 else v
    wheel_v3, wheel_v4 = np.hypot((r + Lx), Ly) * w if w != 0 else v, np.hypot((r + Lx), Ly) * w if w != 0 else v
    P_ut = 1 / 4 * μ * M * g * (
                np.abs(wheel_v1) + np.abs(wheel_v2) + np.abs(wheel_v3) + np.abs(wheel_v4)) if w != 0 else 0
    return P_ul, P_ut
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Saves model when validation loss decrease.
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# noinspection PyPep8Naming
class DCNNDataset(Dataset):
    def __init__(self, x_train, y_train):
        X_train_torch = torch.transpose(torch.from_numpy(x_train), 2, 1).float()
        labels_torch = torch.from_numpy(y_train).float()
        self.samples = [(X_train_torch[i], labels_torch[i]) for i in range(X_train_torch.shape[0])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, y


class ANN(nn.Module):
    def __init__(self, D_in, D_out, D_hidden=50):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(D_in, D_hidden),
            nn.ReLU(),
            nn.Linear(D_hidden, D_hidden),
            nn.ReLU(),
            nn.Linear(D_hidden, D_out),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


# class dilatedCNN(torch.nn.Module):
#     def __init__(self, D_in, D_out, KERNEL_SIZE, T, Horizon):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#         """
#         super(dilatedCNN, self).__init__()
#         # dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
#         self.KERNEL_SIZE = KERNEL_SIZE
#         self.conv1d_1 = torch.nn.Conv1d(D_in, D_out, kernel_size=KERNEL_SIZE, dilation=1)
#         self.act_1 = nn.ReLU()
#         self.conv1d_2 = torch.nn.Conv1d(D_out, D_out, kernel_size=KERNEL_SIZE, dilation=2)
#         self.act_2 = nn.ReLU()
#         self.conv1d_3 = torch.nn.Conv1d(D_out, D_out, kernel_size=KERNEL_SIZE, dilation=4)
#         self.act_3 = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(T * D_out, Horizon)
#
#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         out = F.pad(x, (1, 0))
#         out = self.act_1(self.conv1d_1(out))
#         out = F.pad(out, (2, 0))
#         out = self.act_2(self.conv1d_2(out))
#         out = F.pad(out, (4, 0))
#         out = self.act_3(self.conv1d_3(out))
#         out = self.linear(self.flatten(out))
#
#         return out
#
#
# # test Gpu training
# def training_loop_gpu(n_epochs, optimizer, model, loss_fn, train_loader, valid_loader, device, patience=20,
#                       verbose=True):
#     # to track the training loss as the model trains
#     train_losses = []
#     # to track the validation loss as the model trains
#     valid_losses = []
#     # to track the average training loss per epoch as the model trains
#     avg_train_losses = []
#     # to track the average validation loss per epoch as the model trains
#     avg_valid_losses = []
#
#     # initialize the early_stopping object
#     early_stopping = EarlyStopping(patience=patience, verbose=verbose)
#
#     for epoch in range(1, n_epochs + 1):  # <2>
#         model.train()
#         loss_train = 0.0
#         for x, y in train_loader:  # <3>
#             x = x.to(device=device)
#             y = y.to(device=device)
#             outputs = model(x)  # <4>
#             loss = loss_fn(outputs, y)  # <5>
#             optimizer.zero_grad()  # <6>
#             loss.backward()  # <7>
#             optimizer.step()  # <8>
#             # loss_train += loss.item()  # <9>
#             train_losses.append(loss.item())
#         # if epoch == 1 or epoch % 5 == 0:
#         #     print('{} Epoch {}, Training loss {}'.format(
#         #         datetime.datetime.now(), epoch,
#         #         loss_train / len(train_loader)))  # <10>
#
#         model.eval()
#         for xv, yv in valid_loader:
#             xv = xv.to(device)
#             yv = yv.to(device)
#             outputv = model(xv)
#             # calculate the loss
#             loss = loss_fn(outputv, yv)
#             # record validation loss
#             valid_losses.append(loss.item())
#
#         # print training/validation statistics
#         # calculate average loss over an epoch
#         train_loss = np.average(train_losses)
#         valid_loss = np.average(valid_losses)
#         avg_train_losses.append(train_loss)
#         avg_valid_losses.append(valid_loss)
#
#         epoch_len = len(str(n_epochs))
#
#         print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
#                      f'train_loss: {train_loss:.5f} ' +
#                      f'valid_loss: {valid_loss:.5f}')
#
#         # clear lists to track next epoch
#         train_losses = []
#         valid_losses = []
#
#         # early_stopping needs the validation loss to check if it has decresed,
#         # and if it has, it will make a checkpoint of the current model
#         early_stopping(valid_loss, model)
#
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
#     # load the last checkpoint with the best model
#     model.load_state_dict(torch.load('checkpoint.pt'))
#     return model, avg_train_losses, avg_valid_losses
#
#
# def prepare_data_DCNN(test, T=10, freq='1 s'):
#     test_shifted = test.copy()
#     test_shifted['y_t+1'] = test_shifted['P'].shift(-1, freq=freq)
#     for t in range(1, T + 1):
#         test_shifted['P_t-' + str(T - t)] = test_shifted['P'].shift(T - t, freq=freq)
#     test_shifted = test_shifted.dropna(how='any')
#     y_test = test_shifted['y_t+1'].to_numpy()
#     X_test = test_shifted[['P_t-' + str(T - t) for t in range(1, T + 1)]].to_numpy()
#     X_test = X_test[..., np.newaxis]
#     return X_test, y_test, test_shifted


def comparative_plotting(X_eval, y_eval, model, device, title='test'):
    model.eval()
    # X_eval, y_eval = X_train.copy(), y_train.copy()
    # training
    xpt = torch.transpose(torch.from_numpy(X_eval), 2, 1).float().to(device)
    p_predict = model(xpt).cpu().detach().numpy().reshape(-1)
    p_real = y_eval
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(p_real, label='train')
    ax.plot(p_predict, label='prediction', linewidth=2, color='blue')
    ax.set_title(
        title + ": R2_score = %.2f, MAPE = %.2f" % (r2_score(p_real, p_predict), MAPE(p_real, p_predict) * 100) + '%')
    return p_real, p_predict, fig, ax
