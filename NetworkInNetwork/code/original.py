import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import data_v2
# import cPickle as pickle
from six.moves import cPickle as pickle
import numpy
import NIN
import platform

from torch.autograd import Variable


def main():

    trainset = data_v2.dataset(root='./data', train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = data_v2.dataset(root='./data', train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = NIN.NetworkInNetwork()
    print(model)
    model.load_state_dict(torch.load("./data/model_weights.pth"))
    model.eval()
    model.cuda()

    # pretrained = False
    # if pretrained:
    #     # params = pickle.load(open('data/params', 'r'))
    #     version = platform.python_version_tuple()

    #     if version[0] == '2':
    #         params = pickle.load(open('./data/model_weights.pth', 'r'))
    #     elif version[0] == '3':
    #         params = pickle.load(
    #             open('./data/model_weights.pth', 'r'), encoding='latin1')
    #     index = -1
    #     for m in model.modules():
    #         if isinstance(m, nn.Conv2d):
    #             index = index + 1
    #             weight = torch.from_numpy(params[index])
    #             m.weight.data.copy_(weight)
    #             index = index + 1
    #             bias = torch.from_numpy(params[index])
    #             m.bias.data.copy_(bias)
    # else:
    #     for m in model.modules():
    #         if isinstance(m, nn.Conv2d):
    #             m.weight.data.normal_(0, 0.05)
    #             m.bias.data.normal_(0, 0.0)

    criterion = nn.CrossEntropyLoss()
    param_dict = dict(model.named_parameters())
    params = []

    base_lr = 0.0001

    for key, value in param_dict.items():
        if key == 'classifier.20.weight':
            params += [{'params': [value], 'lr':0.1 * base_lr,
                        'momentum':0.95, 'weight_decay':0.0001}]
        elif key == 'classifier.20.bias':
            params += [{'params': [value], 'lr':0.1 * base_lr,
                        'momentum':0.95, 'weight_decay':0.0000}]
        elif 'weight' in key:
            params += [{'params': [value], 'lr':1.0 * base_lr,
                        'momentum':0.95, 'weight_decay':0.0001}]
        else:
            params += [{'params': [value], 'lr':2.0 * base_lr,
                        'momentum':0.95, 'weight_decay':0.0000}]

    optimizer = optim.Adagrad(
        params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.data.item(),
                    optimizer.param_groups[1]['lr']))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in testloader:
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = model(data)
            test_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(testloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss * 128., correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))

    def adjust_learning_rate(optimizer, epoch):
        if epoch % 80 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1

    def print_std():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                print(torch.std(m.weight.data))

    for epoch in range(1, 5):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
        torch.save(model.state_dict(), "./data/model_weights.pth")


if __name__ == "__main__":
    main()
