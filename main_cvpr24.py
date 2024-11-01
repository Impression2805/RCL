import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import argparse
import os
import csv
import math
import pandas as pd
import numpy as np
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import resnet
from model import resnet18
from model import vgg
from model import wrn
from utils import data as dataset
from utils import crl_utils
from utils import metrics
from utils import utils
import train
import copy
from un_md_ood_new import calc_ood_function
from PIL import Image
from matplotlib import pyplot as plt
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class Counter(dict):
    def __missing__(self, key):
        return None

parser = argparse.ArgumentParser(description='Reliable continual learning for Failure detection')
parser.add_argument('--gpu', default='2', type=str, help='GPU id to use')
parser.add_argument('--epochs', default=10, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--plot', default=1, type=int, help='')
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--run', default=1, type=int, help='')
parser.add_argument('--classnumber', default=100, type=int, help='class number for the dataset')
parser.add_argument('--data', default='cifar100', type=str, help='Dataset name to use [cifar10, cifar100]')
parser.add_argument('--model', default='res110', type=str, help='Models name to use [dense, res110, wrn, res110, vgg, wrn]')
parser.add_argument('--method', default='RCL', type=str, help='[LogitNorm, RCL (Reliable Continual Learning, ours), Baseline, CRL, FMFP]')
parser.add_argument('--base_method', default='FMFP', type=str, help='[Baseline, CRL, FMFP], which model we will patch')
parser.add_argument('--data_path', default='./data', type=str, help='Drataset directory')
parser.add_argument('--noise', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--save_path', default='./output_cifar/', type=str, help='Savefiles directory')
parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--ewc', default=1, type=float, help='1k, 0.5k')
parser.add_argument('--rtemp', default=10, type=float, help='0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10')
### OOD detection ####
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise_odin', type=float, default=0, help='noise for Odin')
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy|Odin|MLogit|ReAct')
parser.add_argument('--test_bs', type=int, default=500)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--ft', default='full', type=str, help='full|layer3|bn|fc')
args = parser.parse_args()


def main():
    acc_list=[]
    auroc_list=[]
    aupr_success_list=[]
    aupr_list=[]
    fpr_list=[]
    tnr_list = []
    aurc_list=[]
    eaurc_list=[]
    ece_list=[]
    nll_list=[]
    brier_list=[]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    save_path = args.save_path + args.data + '_' + args.model + '_' + args.base_method
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_loader, test_data, test_loader, \
    test_onehot, test_label = dataset.get_loader(args.data, args.data_path, args.batch_size, args)

    if args.data == 'cifar100':
        num_class = 100
        args.classnumber = 100
    else:
        num_class = 10
    model_dict = {
        "num_classes": num_class,
    }
    for r in range(args.run):
        print(100*'#')
        print(r)
        if args.model == 'resnet18':
            model = resnet18.ResNet18(**model_dict).cuda()
            model_swa = resnet18.ResNet18(**model_dict).cuda()
        elif args.model == 'res110':
            model = resnet.resnet110(**model_dict).cuda()
            model_swa = resnet.resnet110(**model_dict).cuda()
        elif args.model == 'wrn':
            model = wrn.WideResNet(28, num_class, 10).cuda()
            model_swa = wrn.WideResNet(28, num_class, 10).cuda()


        if args.base_method == 'FMFP':
        # ########### flat minima models require additional operations during the loading process. ################
            state_dict = torch.load(save_path + '/200_model.pth')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_name = k.replace('module.', '')
                new_state_dict[new_name] = v
            model.load_state_dict(new_state_dict, False)
            model_swa.load_state_dict(new_state_dict, False)
        else:
            ############# modify the path to the pretrained model (e.g., CRL model) #####################
            model_state_dict = torch.load(save_path + '/200_model.pth')
            model.load_state_dict(model_state_dict)
            model_swa.load_state_dict(model_state_dict)


        cls_criterion = nn.CrossEntropyLoss().cuda()
        base_lr = 0.01  # Initial learning rate
        lr_strat = [4]
        lr_factor = 0.1  # Learning rate decrease factor
        custom_weight_decay = 5e-4  # Weight Decay
        custom_momentum = 0.9  # Momentum
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,
                                    weight_decay=custom_weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)

        # ########## cos lr can also be used #####################
        # optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,
        #                             weight_decay=custom_weight_decay)
        #
        # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)


        correctness_history = crl_utils.History(len(train_loader.dataset))
        ranking_criterion = nn.MarginRankingLoss(margin=0.0).cuda()
        swa_n = 0

        old_model = None
        cls_criterion = nn.CrossEntropyLoss().cuda()
        MDaurc, MDfpr, MDauroc, OODfpr, OODauroc, FDaurc, FDfpr, FDauroc, ACC = calc_ood_function(args, test_data,
                                                                                                  test_loader, model,
                                                                                                  train_loader=train_loader,
                                                                                                  old_model=old_model)
        # make logger
        train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
        result_logger = utils.Logger(os.path.join(save_path, 'result.log'))
        
        # if args.ft == 'layer3':
        #     for i, param in enumerate(model.named_parameters()):
        #         k1, v1 = param
        #         if 'layer1' in k1 or 'layer2' in k1:
        #             param[1].requires_grad = False
        # elif args.ft == 'bn':
        #     for i, param in enumerate(model.named_parameters()):
        #         k1, v1 = param
        #         if 'bn' not in k1:
        #             param[1].requires_grad = False


        fisher, mean = getFisherDiagonal(model, train_loader)


        for epoch in range(1, args.epochs + 1):
            if scheduler != None:
                scheduler.step()
            train.train(train_loader, model, cls_criterion, ranking_criterion, optimizer, epoch, correctness_history, train_logger, args, fisher, mean)

            if (epoch + 1) >= 0:
                swa_n = 1
                moving_average(model_swa, model, 1.0 / (swa_n + 1))
                swa_n += 1
                bn_update(train_loader, model_swa)
                if epoch == args.epochs:
                    model_name = str(epoch) + '_' + args.method + '.pth'
                    torch.save(model_swa.state_dict(), os.path.join(save_path, model_name))

                # calc measure
                # if epoch % args.plot == 0:
                print(100*'#')
                print(epoch)
                MDaurc, MDfpr, MDauroc, OODfpr, OODauroc, FDaurc, FDfpr, FDauroc, ACC = calc_ood_function(args,
                                                                                                          test_data,
                                                                                                          test_loader,
                                                                                                          model,
                                                                                                          train_loader=train_loader,
                                                                                                          old_model=old_model)

                if epoch == args.epochs:
                    logs_dict = OrderedDict(Counter(
                        {
                            "MDaurc": {
                                "value": MDaurc,
                                "string": f"{MDaurc}",
                            },
                            "MDfpr": {
                                "value": MDfpr,
                                "string": f"{MDfpr}",
                            },
                            "MDauroc": {
                                "value": MDauroc,
                                "string": f"{MDauroc}",
                            },
                            "OODfpr": {
                                "value": OODfpr,
                                "string": f"{OODfpr}",
                            },
                            "OODauroc": {
                                "value": OODauroc,
                                "string": f"{OODauroc}",
                            },
                            "FDaurc": {
                                "value": FDaurc,
                                "string": f"{FDaurc}",
                            },
                            "FDfpr": {
                                "value": FDfpr,
                                "string": f"{FDfpr}",
                            },
                            "FDauroc": {
                                "value": FDauroc,
                                "string": f"{FDauroc}",
                            },
                            "ACC": {
                                "value": ACC,
                                "string": f"{ACC}",
                            },
                        }
                    ))

                    # Print metrics
                    csv_writter(path=save_path, dic=OrderedDict(logs_dict), start=1, score=args.score)
                    os.chdir('../..')



def csv_writter(path, dic, start, score):
    if os.path.isdir(path) == False: os.makedirs(path)
    os.chdir(path)
    if start == 1:
        mode = 'w'
    else:
        mode = 'a'
    with open('results_log.csv', mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if start == 1:
            writer.writerow(dic.keys())
        writer.writerow([elem["string"] for elem in dic.values()])

        

#################### some useful functions for model average ########################
def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for _, (input, _, _) in enumerate(loader):
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


#################### compute the parameter importance for RWC/EWC ########################
def getFisherDiagonal(model, train_loader):
    fishermax = 0.0001
    fisher = {n: torch.zeros(p.shape).cuda() for n, p in model.named_parameters()
              if p.requires_grad}
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for i, (inputs, targets, idx) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2).clone()
    for n,p in fisher.items():
        fisher[n]=p/len(train_loader)
        fisher[n]=torch.min(fisher[n],torch.tensor(fishermax))
    mean = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    return fisher, mean


if __name__ == "__main__":
    main()



