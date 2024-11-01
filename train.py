import utils.crl_utils
from utils import utils
import torch.nn as nn
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import random


def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(Loss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):  # [bs,num_class]  CE=q*-log(p), q*log(1-p),p=softmax(logits)
        target = target.reshape(logits.shape[0], 1)
        log_pro = -1.0 * torch.log(softmax_one(logits, dim=1))
        one_hot = torch.zeros(logits.shape[0], logits.shape[1]).cuda()
        one_hot = one_hot.scatter_(1, target, 1)
        loss = torch.mul(log_pro, one_hot).sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
loss_fun = Loss('mean')


class KDLoss(nn.Module):
    def __init__(self, temp_factor=1):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
KD_loss = KDLoss(2.0)


class LogitNormLoss(nn.Module):
    def __init__(self, t=0.1):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x, target, t=10):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms)  / self.t
        return F.cross_entropy(logit_norm, target)

criterion_lm = LogitNormLoss()



def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_ewc(model, fisher, mean):
    loss = 0
    for n, p in model.named_parameters():
        if n in fisher.keys():
            loss += torch.sum((fisher[n]) * (p[:len(mean[n])] - mean[n]).pow(2)) / 2
    return loss


def one_hot_encoding(label):
    cls = set(label)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, label)))
    return one_hot

def train(loader, model, criterion, criterion_ranking, optimizer, epoch, history, logger, args, fisher=None, mean=None, old_network=None):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()
    ranking_losses = utils.AverageMeter()
    end = time.time()
    model.train()
    if args.method == 'FMFP':
        for i, (input, target, idx) in enumerate(loader):
            data_time.update(time.time() - end)
            input, target = input.cuda(), target.long().cuda()
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(input), target).backward()
            optimizer.second_step(zero_grad=True)
            prec, correct = utils.accuracy(output, target)
            total_losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        logger.write([epoch, total_losses.avg, top1.avg])
    else:
        for i, (input, target, idx) in enumerate(loader):
            data_time.update(time.time() - end)
            if args.method == 'Baseline':
                input, target = input.cuda(), target.long().cuda()
                output = model(input)
                loss = criterion(output, target)
            elif args.method == 'LogitNorm':
                input, target = input.cuda(), target.long().cuda()
                output = model(input)
                loss = criterion_lm(output, target, t=args.rtemp)
            elif args.method == 'RCL':
                input, target = input.cuda(), target.long().cuda()
                output = model(input)
                loss_ewc = compute_ewc(model, fisher, mean)
                loss = criterion_lm(output, target, t=args.rtemp) + args.ewc * 10000* loss_ewc
            elif args.method == 'CRL':
                input, target = input.cuda(), target.long().cuda()
                output = model(input)
                conf = F.softmax(output, dim=1)
                confidence, _ = conf.max(dim=1)

                rank_input1 = confidence
                rank_input2 = torch.roll(confidence, -1)
                idx2 = torch.roll(idx, -1)

                rank_target, rank_margin = history.get_target_margin(idx, idx2)
                rank_target_nonzero = rank_target.clone()
                rank_target_nonzero[rank_target_nonzero == 0] = 1
                rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

                ranking_loss = criterion_ranking(rank_input1,rank_input2,rank_target)
                cls_loss = criterion(output, target)
                # cls_loss = criterion_lm(output, target, t=args.rtemp)
                ranking_loss = args.rank_weight * ranking_loss
                loss = cls_loss + ranking_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec, correct = utils.accuracy(output, target)
            total_losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            # if i % args.print_freq == 0:
            #     print('Epoch: [{0}][{1}/{2}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
            #            epoch, i, len(loader), batch_time=batch_time,
            #            data_time=data_time, loss=total_losses,top1=top1))
            if args.method == 'CRL':
                history.correctness_update(idx, correct, output)
        if args.method == 'CRL':
            history.max_correctness_update(epoch)
        logger.write([epoch, total_losses.avg, top1.avg])
