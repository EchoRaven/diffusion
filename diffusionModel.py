import os
from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
from sklearn.datasets import make_s_curve
import matplotlib.pyplot as plt
from torch import optim

class ImageDataSet(Dataset, ABC):
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = torch.Tensor(self.data[idx]).float()
        return pos


#diffusion层
class DiffusionLayer(nn.Module):
    def __init__(self,
                 d_model, #建模维数
                 layer_num, #层的数量
                 head_num, #注意力头的数量
                 ):
        super(DiffusionLayer, self).__init__()
        #encoder层
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=head_num, dim_feedforward=d_model)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, layer_num)

    def forward(self, x):
        return self.encoder(x)

class DiffusionModel(nn.Module):
    def __init__(self,
                 data,  #数据地址
                 batch_size,  #训练集大小
                 num_steps,  #马尔科夫链长度
                 d_model,  # 建模维数
                 layer_num,  # 层的数量
                 head_num  # 注意力头的数量
                 ):
        super(DiffusionModel, self).__init__()
        self.dataSet = ImageDataSet(data)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.dataLoader = DataLoader(dataset=self.dataSet, shuffle=True, batch_size=batch_size, drop_last=True)
        #生成beta
        betas = torch.linspace(-6, 6, num_steps)
        self.betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        #生成alpha
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.Tensor([1]).float(), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        self.enocder = DiffusionLayer(d_model=d_model, layer_num=layer_num, head_num=head_num)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def generateXt(self, x0, t):
        #正态分布生成noise
        noise = torch.rand_like(x0)
        alphas_t = self.alphas_bar_sqrt[t]
        alphas_1_m_t = self.one_minus_alphas_bar_sqrt[t]
        return alphas_t * x0 + alphas_1_m_t * noise

    #测试样例
    def show(self, num_shows, dataset):
        dataset = torch.Tensor(dataset).float()
        for i in range(num_shows):
            q_i = self.generateXt(dataset, torch.tensor([i * self.num_steps//num_shows]))
            plt.plot(*q_i, 'o')
            plt.show()

    def Diffusion_loss_fn(self, x0):
        #生成随机时序
        t = torch.randint(0, self.num_steps, size=(self.batch_size//2,))
        t = torch.cat([t, self.num_steps-1-t], dim=0)
        t = t.unsqueeze(-1)
        alpha = self.alphas_bar_sqrt[t]
        alpha_m1 = self.one_minus_alphas_bar_sqrt[t]
        noise = torch.rand_like(x0)
        x = x0 * alpha + noise * alpha_m1
        output = self.enocder(x)
        return (noise - output).square().mean()

    def Sample(self, x, t):
        t = torch.tensor([t])
        coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]
        eps_theta = self.enocder(x)
        mean = (1 / (1-self.betas[t]).sqrt())*(x-coeff * eps_theta)
        z = torch.rand_like(x)
        sigma_t = self.betas[t].sqrt()
        sample = mean + sigma_t * z
        return sample

    def SampleLoop(self, shape):
        cur_x = torch.randn(shape)
        x_seq = [cur_x]
        for i in reversed((range(self.num_steps))):
            cur_x = self.Sample(cur_x, i)
            x_seq.append(cur_x)
            print(cur_x)
        return x_seq


    def Train(self,
              epochs, #训练轮数
              ):
        #遍历数据集
        for e in range(epochs):
            for idx, batch in enumerate(self.dataLoader):
                loss = self.Diffusion_loss_fn(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()
            print(loss)
            #if e % 10 == 0:
            #    x_seq = self.SampleLoop(self.dataSet.shape)
            #    plt.plot(x_seq, 'o')
            #    plt.show()




if __name__ == "__main__":
    #噪声采样是针对与位置的
    s_curve, _ = make_s_curve(10**4, noise=0.1)
    s_curve = s_curve[:, [0, 2]]/10.0
    if not os.path.exists('diffusion.pth'):
        model = DiffusionModel(s_curve, batch_size=128, num_steps=100, d_model=2, layer_num=6, head_num=2)
        model.Train(100)
        torch.save(model, 'diffusion.pth')
    else:
        model = torch.load('diffusion.pth')
