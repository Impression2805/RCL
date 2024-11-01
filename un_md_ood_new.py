import numpy as np
import sys
import os
import pickle
import argparse
import torch
from copy import deepcopy
import faiss
from scipy.special import logsumexp
from math import ceil
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision.transforms as trn
import torchvision
import torchvision.datasets as dset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import normalize
import csv
import math
import pandas as pd
import resource
from collections import OrderedDict
import random
from model import resnet
from model import resnet18
from model import wrn
from utils import data as dataset
from utils import crl_utils
from utils import metrics
from utils import utils
import train
from sklearn import metrics
from PIL import Image as PILImage
import sklearn
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
import utils.svhn_loader as svhn
import utils.lsun_loader as lsun_loader
import utils.score_calculation as lib

knn_K = 50
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


def knn_score(bankfeas, queryfeas, k=100, min=False):

   bankfeas = deepcopy(np.array(bankfeas))
   queryfeas = deepcopy(np.array(queryfeas))

   index = faiss.IndexFlatIP(bankfeas.shape[-1])
   index.add(bankfeas)
   D, _ = index.search(queryfeas, k)
   if min:
       scores = np.array(D.min(axis=1))
   else:
       scores = np.array(D.mean(axis=1))
   return scores


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
   out = np.cumsum(arr, dtype=np.float64)
   expected = np.sum(arr, dtype=np.float64)
   if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
       raise RuntimeError('cumsum was found to be unstable: '
                          'its last element does not correspond to sum')
   return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
   classes = np.unique(y_true)
   if (pos_label is None and
           not (np.array_equal(classes, [0, 1]) or
                    np.array_equal(classes, [-1, 1]) or
                    np.array_equal(classes, [0]) or
                    np.array_equal(classes, [-1]) or
                    np.array_equal(classes, [1]))):
       raise ValueError("Data is not binary and pos_label is not specified")
   elif pos_label is None:
       pos_label = 1.

   # make y_true a boolean vector
   y_true = (y_true == pos_label)

   # sort scores and corresponding truth values
   desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
   y_score = y_score[desc_score_indices]
   y_true = y_true[desc_score_indices]

   # y_score typically has many tied values. Here we extract
   # the indices associated with the distinct values. We also
   # concatenate a value for the end of the curve.
   distinct_value_indices = np.where(np.diff(y_score))[0]
   threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

   # accumulate the true positives with decreasing threshold
   tps = stable_cumsum(y_true)[threshold_idxs]
   fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

   thresholds = y_score[threshold_idxs]

   recall = tps / tps[-1]

   last_ind = tps.searchsorted(tps[-1])
   sl = slice(last_ind, None, -1)      # [last_ind::-1]
   recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

   cutoff = np.argmin(np.abs(recall - recall_level))

   return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])



def coverage_risk(confidence, correctness):
   risk_list = []
   coverage_list = []
   risk = 0
   for i in range(len(confidence)):
       coverage = (i + 1) / len(confidence)
       coverage_list.append(coverage)

       if correctness[i] == 0:
           risk += 1

       risk_list.append(risk / (i + 1))

   return risk_list, coverage_list


def aurc_eaurc(risk_list):
   r = risk_list[-1]
   risk_coverage_curve_area = 0
   optimal_risk_area = r + (1 - r) * np.log(1 - r)
   for risk_value in risk_list:
       risk_coverage_curve_area += risk_value * (1 / len(risk_list))

   aurc = risk_coverage_curve_area
   eaurc = risk_coverage_curve_area - optimal_risk_area

   return aurc, eaurc

def calc_aurc_eaurc(softmax, correct):
   softmax = np.array(softmax)
   correctness = np.array(correct)
   softmax_max = -np.array(softmax)

   sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
   sort_softmax_max, sort_correctness = zip(*sort_values)
   risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
   aurc, eaurc = aurc_eaurc(risk_li)

   return aurc, eaurc


def get_measures(_pos, _neg, recall_level=0.95):
   pos = np.array(_pos[:]).reshape((-1, 1))
   neg = np.array(_neg[:]).reshape((-1, 1))
   examples = np.squeeze(np.vstack((pos, neg)))
   labels = np.zeros(len(examples), dtype=np.int32)
   labels[:len(pos)] += 1

   auroc = sklearn.metrics.roc_auc_score(labels, examples)
   aupr = sklearn.metrics.average_precision_score(labels, examples)
   fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

   correct = np.zeros(len(examples), dtype=np.int32)
   correct[len(pos):] += 1
   aurc, eaurc = calc_aurc_eaurc(examples, correct)

   return aurc, fpr, auroc

# /////////////// Detection Prelims ///////////////

def get_in_scores(args, ood_num_examples, model, loader, in_dist=False, knn_index=None, old_model=None):
   concat = lambda x: np.concatenate(x, axis=0)
   to_np = lambda x: x.data.cpu().numpy()
   model.eval()
   _score = []
   _right_score = []
   _wrong_score = []

   with torch.no_grad():
       for batch_idx, (data, target, _) in enumerate(loader):
           if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
               break

           data = data.cuda()
           output = model(data)
           output = output[:,:args.classnumber]

           Logits = to_np(output)
           smax = to_np(F.softmax(output, dim=1))


           if args.use_xent:
               _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
           else:
               if args.score == 'energy':
                   _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
               elif args.score == 'MLogit':
                   _score.append(-np.max(Logits, axis=1))
               elif args.score == 'MSP':
                   _score.append(-np.max(smax, axis=1))
               elif args.score == 'knn':
                   normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
                   _, knn_feature = model(data, feature_output=True)
                   knn_feature_normed = normalizer(knn_feature.data.cpu().numpy())
                   D, _ = knn_index.search(knn_feature_normed, knn_K, )
                   kth_dist = -D[:, -1]
                   _score.append(-kth_dist)

           if in_dist:
               preds = np.argmax(smax, axis=1)
               targets = target.numpy().squeeze()
               right_indices = preds == targets
               wrong_indices = np.invert(right_indices)

               if args.use_xent:
                   _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                   _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
               else:
                   if args.score == 'energy':
                       _right_score.append(-to_np((args.T * torch.logsumexp(output[right_indices] / args.T, dim=1))))
                       _wrong_score.append(-to_np((args.T * torch.logsumexp(output[wrong_indices] / args.T, dim=1))))
                   elif args.score == 'MLogit':
                       _right_score.append(-np.max(Logits[right_indices], axis=1))
                       _wrong_score.append(-np.max(Logits[wrong_indices], axis=1))
                   elif args.score == 'MSP':
                       _right_score.append(-np.max(smax[right_indices], axis=1))
                       _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
                   elif args.score == 'knn':
                       normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
                       _, knn_feature = model(data, feature_output=True)
                       knn_feature_normed = normalizer(knn_feature.data.cpu().numpy())
                       D, _ = knn_index.search(knn_feature_normed, knn_K, )
                       kth_dist = -D[:, -1]
                       _right_score.append(-kth_dist[right_indices])
                       _wrong_score.append(-kth_dist[wrong_indices])


   if in_dist:
       return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
   else:
       return concat(_score)[:ood_num_examples].copy()


def get_ood_scores(args, ood_num_examples, model, loader, in_dist=False, knn_index=None, old_model=None):
   concat = lambda x: np.concatenate(x, axis=0)
   to_np = lambda x: x.data.cpu().numpy()
   model.eval()
   _score = []
   _right_score = []
   _wrong_score = []

   with torch.no_grad():
       for batch_idx, (data, target) in enumerate(loader):
           if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
               break

           data = data.cuda()
           output = model(data)
           output = output[:,:args.classnumber]
           Logits = to_np(output)
           smax = to_np(F.softmax(output, dim=1))


           if args.use_xent:
               _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
           else:
               if args.score == 'energy':
                   _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
               elif args.score == 'MLogit':
                   _score.append(-np.max(Logits, axis=1))
               elif args.score == 'MSP':
                   _score.append(-np.max(smax, axis=1))
               elif args.score == 'knn':
                   normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
                   _, knn_feature = model(data, feature_output=True)
                   knn_feature_normed = normalizer(knn_feature.data.cpu().numpy())
                   D, _ = knn_index.search(knn_feature_normed, knn_K,)
                   kth_dist = -D[:, -1]
                   _score.append(-kth_dist)

           if in_dist:
               preds = np.argmax(smax, axis=1)
               targets = target.numpy().squeeze()
               right_indices = preds == targets
               wrong_indices = np.invert(right_indices)

               if args.use_xent:
                   _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                   _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
               else:
                   _right_score.append(-np.max(smax[right_indices], axis=1))
                   _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

   if in_dist:
       return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
   else:
       return concat(_score)[:ood_num_examples].copy()

# /////////////// OOD Detection ///////////////
def get_and_print_results(args, test_loader, ood_num_examples, model, OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list, ood_loader, num_to_avg, knn_index, old_model=None):
   OODfprs, OODaurocs, FDaurcs, FDfprs, FDaurocs = [], [], [], [], []
   in_score, right_score, wrong_score = get_in_scores(args, ood_num_examples, model, test_loader, in_dist=True, knn_index=knn_index, old_model=old_model)
   for _ in range(num_to_avg):
       out_score = get_ood_scores(args, ood_num_examples, model, ood_loader, knn_index=knn_index, old_model=old_model)
       measures_ood = get_measures(out_score, in_score)
       OODfprs.append(measures_ood[1]); OODaurocs.append(measures_ood[2])

       minsize = min(wrong_score.shape[0], out_score.shape[0])
       np.random.shuffle(wrong_score); np.random.shuffle(out_score)
       pos_score = np.concatenate((wrong_score[:minsize], out_score[:minsize]), axis=0)

       # ########## modify this for full tet ##############
       # np.random.shuffle(wrong_score);
       # np.random.shuffle(out_score)
       # pos_score = np.concatenate((wrong_score, out_score), axis=0)


       measures_fd = get_measures(pos_score, right_score)
       FDaurcs.append(measures_fd[0]);  FDfprs.append(measures_fd[1]); FDaurocs.append(measures_fd[2])

   OODfpr = np.mean(OODfprs); OODauroc = np.mean(OODaurocs)
   OODfpr_list.append(OODfpr); OODauroc_list.append(OODauroc)

   FDaurc = np.mean(FDaurcs); FDfpr = np.mean(FDfprs); FDauroc = np.mean(FDaurocs)
   FDaurc_list.append(FDaurc); FDfpr_list.append(FDfpr); FDauroc_list.append(FDauroc)

   return OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list



def calc_ood_function(args, test_data, test_loader, model, in_dist=False, train_loader=None, old_model=None):
   model.eval()
   if old_model is not None:
       old_model.eval()
   ########## newly added ViM and KNN score ##############
   if args.score == 'knn':
       # knn settings
       knn_K = 50
       normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
       import faiss
       activation_log = []
       with torch.no_grad():
           for batch_idx, (data, target, _) in enumerate(train_loader):
               data = data.cuda()
               data = data.float()
               batch_size = data.shape[0]
               _, feature = model(data, feature_output=True)
               dim = feature.shape[1]
               activation_log.append(normalizer(feature.data.cpu().numpy().reshape(batch_size, dim, -1).mean(2)))
       activation_log = np.concatenate(activation_log, axis=0)
       knn_index = faiss.IndexFlatL2(feature.shape[1])
       knn_index.add(activation_log)
       print('************KNN**********')
       print(type(knn_index))
   else:
       knn_index = None



   ood_num_examples = len(test_data) // 5
   expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
   concat = lambda x: np.concatenate(x, axis=0)
   to_np = lambda x: x.data.cpu().numpy()

   if args.data == 'cifar100':
       mean = [0.507, 0.487, 0.441]
       std = [0.267, 0.256, 0.276]
   elif args.data == 'cifar10':
       mean = [0.491, 0.482, 0.447]
       std = [0.247, 0.243, 0.262]
   imagesize = 32
   if args.model == "vit":
       mean = [0.5, 0.5, 0.5]
       std = [0.5, 0.5, 0.5]
       imagesize = 224
   # auroc_list, aurc_list, fpr_list = [], [], []

   in_score, right_score, wrong_score = get_in_scores(args, ood_num_examples, model, test_loader, in_dist=True, knn_index=knn_index, old_model=old_model)
   measures_md = get_measures(wrong_score, right_score)

   num_right = len(right_score)
   num_wrong = len(wrong_score)

   ACC = num_right / (num_wrong + num_right)
   MDaurc = measures_md[0]
   MDfpr = measures_md[1]
   MDauroc = measures_md[2]
   num_workers = 0
   OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list, = [], [], [], [], []

   # ############################ The path of OOD datasets should be changed #####################################
   # /////////////// Textures ///////////////
   ood_data = dset.ImageFolder(root='./data' + "/dtd/images",
                               transform=trn.Compose([trn.Resize(imagesize), trn.CenterCrop(imagesize),
                                                      trn.ToTensor(), trn.Normalize(mean, std)]))
   ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                            num_workers=num_workers, pin_memory=True)
   # print('\n\nTexture Detection')
   OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list = get_and_print_results(args, test_loader, ood_num_examples, model, OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list, ood_loader, args.num_to_avg, knn_index, old_model=old_model)

   # /////////////// SVHN /////////////// # cropped and no sampling of the test set
   ood_data = svhn.SVHN(root='./data', split="test",
                        transform=trn.Compose(
                            [  trn.Resize(imagesize),
                                trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
   ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                            num_workers=num_workers, pin_memory=True)
   # print('\n\nSVHN Detection')
   OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list = get_and_print_results(args, test_loader, ood_num_examples, model, OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list, ood_loader, args.num_to_avg, knn_index, old_model=old_model)

   # /////////////// Places365 ///////////////
   ood_data = dset.ImageFolder(root='./data' + "/Places",
                               transform=trn.Compose([trn.Resize(imagesize), trn.CenterCrop(imagesize),
                                                      trn.ToTensor(), trn.Normalize(mean, std)]))
   ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                            num_workers=num_workers, pin_memory=True)
   # print('\n\nPlaces365 Detection')
   OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list = get_and_print_results(args, test_loader, ood_num_examples, model, OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list, ood_loader, args.num_to_avg, knn_index, old_model=old_model)

   # /////////////// LSUN-C ///////////////
   ood_data = dset.ImageFolder(root='./data' + "/LSUN",
                               transform=trn.Compose(
                                   [transforms.Resize(imagesize), trn.ToTensor(), trn.Normalize(mean, std)]))
   ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                            num_workers=num_workers, pin_memory=True)
   # print('\n\nLSUN_C Detection')
   OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list = get_and_print_results(args, test_loader, ood_num_examples, model, OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list, ood_loader, args.num_to_avg, knn_index, old_model=old_model)

   # # /////////////// LSUN-R ///////////////
   ood_data = dset.ImageFolder(root='./data' + "/LSUN_resize",
                               transform=trn.Compose([transforms.Resize(imagesize), trn.ToTensor(), trn.Normalize(mean, std)]))
   ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                            num_workers=num_workers, pin_memory=True)
   # print('\n\nLSUN_Resize Detection')
   OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list = get_and_print_results(args, test_loader, ood_num_examples, model, OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list, ood_loader, args.num_to_avg, knn_index, old_model=old_model)

   # /////////////// iSUN ///////////////IL.UnidentifiedImageError: cannot identify image file
   ood_data = dset.ImageFolder(root='./data' + "/iSUN",
                               transform=trn.Compose([transforms.Resize(imagesize), trn.ToTensor(), trn.Normalize(mean, std)]))
   ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                            num_workers=num_workers, pin_memory=True)
   # print('\n\niSUN Detection')
   OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list = get_and_print_results(args, test_loader, ood_num_examples, model, OODfpr_list, OODauroc_list, FDaurc_list, FDfpr_list, FDauroc_list, ood_loader, args.num_to_avg, knn_index, old_model=old_model)

   print('Unified Resutls: MDaurc, MDfpr95, MDauroc, OODfpr95, OODauroc, FDaurc, FDfpr95, FDauroc, ACC')
   print(round(1000*MDaurc, 2), round(100*MDfpr, 2), round(100*MDauroc, 2),
           round(100*np.mean(OODfpr_list), 2), round(100*np.mean(OODauroc_list), 2),
           round(1000*np.mean(FDaurc_list), 2), round(100*np.mean(FDfpr_list), 2), round(100*np.mean(FDauroc_list), 2),
           round(100*ACC, 2))
   # /////////////// Mean Results ///////////////
   return (round(1000*MDaurc, 2), round(100*MDfpr, 2), round(100*MDauroc, 2),
           round(100*np.mean(OODfpr_list), 2), round(100*np.mean(OODauroc_list), 2),
           round(1000*np.mean(FDaurc_list), 2), round(100*np.mean(FDfpr_list), 2), round(100*np.mean(FDauroc_list), 2),
           round(100*ACC, 2))
