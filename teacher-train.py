import os

from torchvision import datasets

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

from lenet import LeNet5
import resnet
import torch
import torchvision
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import argparse


parser = argparse.ArgumentParser(description='train-teacher-network')
# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--output_dir', type=str, default='/cache/models/')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

acc = 0
acc_best = 0


torch.backends.cudnn.benchmark = True


if args.dataset == 'MNIST':
    data_train = MNIST(args.data,download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                       ]))
    data_test = MNIST(args.data,download=True,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ]))

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)
    net = LeNet5().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)



# 训练主函数
def train(epoch):

    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer.step()


# 测试主函数
def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))


def train_and_test(epoch):
    train(epoch)
    test()


# 主程序
def main():
    if args.dataset == 'MNIST':
        epoch = 10
    else:
        epoch = 200
    for e in range(1, epoch):
        train_and_test(e)
    torch.save(net, args.output_dir + 'teacher')



if __name__ == '__main__':
    main()

print("Best Acc=%.6f" % acc_best)