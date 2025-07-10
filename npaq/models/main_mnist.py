
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import  BinarizeLinear,BinarizeConv2d
from binarized_dataset import BinarizedDataset
#from binarized_modules import  Binarize,Ternarize,Ternarize2,Ternarize3,Ternarize4,HingeLoss
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=3,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(BinarizedDataset(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([transforms.Resize(4, 4),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(BinarizedDataset(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.Resize(4, 4),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.infl_ratio=3
        # self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        # self.htanh1 = nn.Hardtanh()
        # self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        # self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        # self.htanh2 = nn.Hardtanh()
        # self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        # self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        # self.htanh3 = nn.Hardtanh()
        # self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        # self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        # self.logsoftmax=nn.LogSoftmax()
        # self.drop=nn.Dropout(0.5)
        self.infl_ratio=1
        self.fc1 = BinarizeLinear(16, 6*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(6*self.infl_ratio)
        self.fc2 = BinarizeLinear(6*self.infl_ratio, 4*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(4*self.infl_ratio)
        self.fc3 = BinarizeLinear(4*self.infl_ratio, 10*self.infl_ratio)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x, return_activations=False, binarize_activations=False, bin_thresholds=[0.05, 0.05]):
        activations = dict()
        x = x.view(-1, 4*4)
        x1 = self.fc1(x)
        activations["fc1"] = x1
        x = self.bn1(x1)
        x = self.htanh1(x)
        x2 = self.fc2(x)
        activations["fc2"] = x2
        x = self.bn2(x2)
        x = self.htanh2(x)
        x3 = self.fc3(x)
        activations["fc3"] = x3
        x = x3
        if return_activations:
            return self.logsoftmax(x), activations
        return self.logsoftmax(x)

model = Net()
if args.cuda:
    torch.cuda.set_device(3)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()

# prepare to count predictions for each class
correct_pred = {str(classname): 0 for classname in range(10)}
total_pred = {str(classname): 0 for classname in range(10)}
all_activations = {"fc1":[], "fc2":[], "fc3":[]}
all_inputs = list()
all_predicted_vs_true = list()
# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        #outputs, activations = net(images, return_activations=True)
        all_inputs.append(images)
        outputs, activations = model(images, return_activations=True, binarize_activations=True, bin_thresholds=[0.2, 0.2])
        _, predictions = torch.max(outputs, 1)
        for layer_name in all_activations.keys():
            all_activations[layer_name].append(activations[layer_name])
        all_predicted_vs_true.extend([{"predicted":p.item(), "true":l.item()} for p, l in zip(predictions, labels)])
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[str(label.item())] += 1
            total_pred[str(label.item())] += 1

import pickle

with open('BNN_input_in_order.pkl', 'wb') as f: 
    pickle.dump(torch.cat(all_inputs, dim=0), f)

with open('BNN_activations_mnist.pkl', 'wb') as f: 
    for layer_name in all_activations.keys():
        all_activations[layer_name] = torch.cat(all_activations[layer_name], dim=0)
    pickle.dump(all_activations, f)

with open('BNN_predicted_vs_true_mnist.pkl', 'wb') as f:
    pickle.dump(all_predicted_vs_true, f)