import argparse
import numpy as np
import math
import sys
import pdb
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets.mnist import MNIST, FashionMNIST
from lenet import LeNet5Half


import resnet


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'cifar10', 'cifar100','AR','FashionMNIST','usps'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
parser.add_argument('--lr_Gt', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_Gs', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_Ds', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100,help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default='/cache/models/')

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

accr = 0
accr_best = 0
accr_list = []

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(opt.channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

generator_t = Generator().cuda()
generator_s = Generator().cuda()
teacher = torch.load(opt.teacher_dir + 'teacher').cuda()
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()
teacher = nn.DataParallel(teacher)
generator_t = nn.DataParallel(generator_t)
generator_s = nn.DataParallel(generator_s)

def klloss(x,y):
    p = F.log_softmax(x,dim = -1)
    q = F.softmax(y,dim = -1)
    l_kl = F.kl_div(p, q)
    return l_kl

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl



if opt.dataset == 'MNIST':
    net = LeNet5Half().cuda()
    net = nn.DataParallel(net)
    data_test = MNIST(opt.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))
    data_test_loader = DataLoader(data_test, batch_size=64, num_workers=0, shuffle=False)
    # Optimizers
    optimizer_Gt = torch.optim.Adam(generator_t.parameters(), lr=opt.lr_Gt)
    optimizer_Gs = torch.optim.Adam(generator_s.parameters(), lr=opt.lr_Gs)
    optimizer_Ds = torch.optim.Adam(net.parameters(), lr=opt.lr_Ds)




# ----------
#  Trainin
# ----------

batches_done = 0

for epoch in range(opt.n_epochs):

    total_correct = 0
    avg_loss = 0.0

    for i in range(120):

        net.train()

        z1 = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        z2 = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        z3 = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        z4 = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()


        img_t = generator_t(z1)
        img_s = generator_s(z2)



        # 优化Gt
        optimizer_Gt.zero_grad()
        outputs_T, features_T = teacher(img_t, out_feature=True)
        pred_T = outputs_T.data.max(1)[1]
        loss_activation_T = -features_T.abs().mean()
        loss_one_hot_T = criterion(outputs_T, pred_T)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
        loss_information_entropy_T = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss_T = loss_one_hot_T * opt.oh + loss_information_entropy_T * opt.ie + loss_activation_T * opt.a
        loss_T.backward()
        optimizer_Gt.step()
        # 优化Ds
        optimizer_Ds.zero_grad()
        loss_kd = kdloss(net(imgs_detach.()), teacher(img_s.detach()))
        loss_kd.backward()
        optimizer_Ds.step()
        
        img_t2 = generator_t(z3)
        img_s2 = generator_s(z4)
        # 优化Gs
        optimizer_Gs.zero_grad()
        loss_kl = klloss(img_s2, img_t2)
        outputs_S, features_S = net(img_s, out_feature=True)
        pred_S = outputs_S.data.max(1)[1]
        loss_activation_S = -features_S.abs().mean()
        loss_one_hot_S = criterion(outputs_S, pred_S)
        softmax_o_S = torch.nn.functional.softmax(outputs_S, dim=1).mean(dim=0)
        loss_information_entropy_S = (softmax_o_S * torch.log10(softmax_o_S)).sum()
        loss_S = loss_one_hot_S * opt.oh + loss_information_entropy_S * opt.ie + loss_activation_S * opt.a + loss_kl* opt.kl


        loss_S.backward()

        optimizer_Gs.step()

        if i == 1:
            print("[Epoch %d/%d]  [loss_T: %f] [loss_S: %f]  [loss_KD: %f] [loss_KL: %f] " % (epoch, opt.n_epochs, loss_T.item(), loss_S.item(),loss_kd.item(),loss_kl.item()))



    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            labels = labels.cuda()
            net.eval()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    accr_list.append(accr)
    if accr > accr_best:
        torch.save(net, opt.output_dir + 'student')
        accr_best = accr



print("Best Acc=%.6f"%accr_best)
print(accr_list)
