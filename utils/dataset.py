import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from tiny_imagenet import TinyImageNet
import os


def load(name, batch_size, train, sampler_train=True):

    if name == 'mnist':
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
        ])
        data = datasets.MNIST('./data', train=train, download=True, transform=transform)
    elif name == 'fashion-mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
            ])
        data = datasets.FashionMNIST('./data', train=train, download=True, transform=transform)

    elif name == 'cifar10':
        if train:
            transform = transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        data = datasets.CIFAR10('./data', train=train, download=True, transform=transform)

    elif name == 'cifar100':
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
            ])
        data = datasets.CIFAR100('./data', train=train, download=True, transform=transform)
    elif name == 'lsun':
        if train:
            data = datasets.LSUN('./data', classes='train')
        else:
            data = datasets.LSUN('./data', classes='test')
    elif name == 'tiny':  # 200 classes
        class_set = list(range(200))
        classes = class_set
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            data = TinyImageNet('./tiny-imagenet', split='train', download=True, transform=transform, index=classes)
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            data = TinyImageNet('./tiny-imagenet', split='val', download=True, transform=transform, index=classes)
    elif name == 'svhn':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if train:
            data = datasets.SVHN('./data', split='train', download=True, transform=transform)
        else:
            data = datasets.SVHN('./data', split='test', download=True, transform=transform)
    elif name == 'stl10':
        if train:
            transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(96),
                #transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            data = datasets.STL10('./data', split='train', download=True, transform=transform)
        else:
            transform = transforms.Compose([
                #transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            data = datasets.STL10('./data', split='test', download=True, transform=transform)

    # valid split 0.1
    if train:
        num_train = len(data)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        if sampler_train:
            return torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, sampler=train_sampler, num_workers=5)
        else:
            return torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, sampler=val_sampler, num_workers=5)
    else:
        return torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=5, shuffle=True)


def getood(data_type, batch_size=50, imageSize=32):
    if data_type == 'imagenet':
        testsetout = datasets.ImageFolder('./data'+"/Imagenet_resize", transform=transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)

    elif data_type == 'lsun':
        testsetout = datasets.ImageFolder('./data'+"/LSUN_resize", transform=transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)

    return test_loader


