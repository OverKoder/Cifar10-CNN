import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from model import VGG
from utils import progress_bar, interval95

import matplotlib.pyplot as plt


# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 

# Data
print('Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('Building model..')
model = VGG('VGG19')
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 1e-3)



# Training function
def train(epoch):
    print('-----=| Epoch %d |=-----' % epoch)

    # Set model to train
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss, 100.*correct/total


def test(epoch):

    global best_acc

    #Set model to evaluation
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    
    return test_loss, 100.*correct/total

    

def main():

    axis = list(range(start_epoch, start_epoch+200))
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []

    for epoch in range(start_epoch, start_epoch+200):

        train_loss, train_acc = train(epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        test_loss, test_acc = test(epoch)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)       


    plt.xlabel("Epochs")

    plt.ylabel("Loss")
    plt.plot(axis,train_loss_list,label= "Train loss")
    plt.plot(axis,test_loss_list,label= "Test loss")
    plt.legend()
    plt.savefig("Loss")
    plt.clf()

    plt.ylabel("Accuracy (%)")
    plt.plot(axis,train_acc_list,label= "Train accuracy")
    plt.plot(axis,test_acc_list,label= "Test accuracy")
    plt.legend()
    plt.savefig("Accuracy")
    plt.clf()

    print("Best accuracy:", best_acc)
    interval = interval95( best_acc / 100, len(testset))
    print("Confidence interval (95%):")
    print("[",best_acc - interval[0], best_acc + interval[1], "]" )



if __name__ == "__main__":
    main()