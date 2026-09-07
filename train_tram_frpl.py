import sys


from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from utils.evaluate_tram import eval
from tram_dataset import get_tram_dataloaders,DataLoader,ImageDataSet

from tram_frpl import *

from template import caption_generator
import random
import numpy as np
import pandas as pd
from config import configC, configM, configI, configO, config_cefa, config_surf, config_wmca
from config import config_CI, config_CO , config_CM, config_MC, config_MI, config_MO, config_IC, config_IO, config_IM, config_OC, config_OI, config_OM
from config import config_CSW,config_CWS,config_SWC
from config import config_WH
from datetime import datetime
import time
from timeit import default_timer as timer
from collections import OrderedDict


import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def check_gradient_vanishing(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'{name}, Gradient norm: {param.grad.norm()}')
        else:
            print(f'{name},gradient disappear')


def train(config):

  train_dataloader,test_dataloader  = get_tram_dataloaders(
      config.src1_data,
      config.src2_data,
      config.src3_data,
      config.src4_data,
      config.tgt_data,
      shrink_pixel = args.shrink_pixel,
      batch_size = args.batch_size)

  best_model_ACC = 0.0
  best_model_HTER = 1.0
  best_model_ACER = 1.0
  best_model_AUC = 0.0
  best_TPR_FPR = 0.0

  valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

  loss_classifier = AverageMeter()
  classifer_top1 = AverageMeter()


  log = Logger()
  log.write('Random seed:  ')
  log.write(str(random_seed))
  log.write('\n')
  log.write(
      '\n----------------------------------------------- [START %s] %s\n\n' %
      (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
  log.write(args.method)
  log.write('\n')
  log.write('Target Dataset:  ')
  log.write(config.tgt_data)
  log.write('\n')
  log.write('** start training target model! **\n')

  criterion = {'softmax': nn.CrossEntropyLoss().cuda()}

  state_dict = torch.jit.load("ViT-B-16.pt").state_dict()
  model = TRAM(state_dict,args.layer)
  if args.pretrain_path:
    log.write('Load pretrained:  ')
    log.write(str(args.pretrain_path.split('/')[-1]))
    log.write('\n')
    state_dict = torch.load(args.pretrain_path)['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_k = k.replace('model.', 'clip_backbone.', 1)
        elif k.startswith('extractor.'):
            new_k = k.replace('extractor.', 'clip_backbone.', 1)
        else:
            new_k = k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict,strict=False)

  if args.aug_path:
    log.write('Load self-supervised pretrained:  ')
    log.write(str(args.aug_path.split('/')[-1]))
    log.write('\n')
    checkpoint  = torch.load(args.aug_path)
    model.load_state_dict(checkpoint.state_dict(),strict=False)

  log.write('alpha: ')
  log.write(str(args.alpha))
  log.write('\n')
  log.write('beta: ')
  log.write(str(args.beta))
  log.write('\n')
  log.write('gamma: ')
  log.write(str(args.gamma))
  log.write('\n')
  log.write('indicator threshold: ')
  log.write(str(args.thres))
  log.write('\n')
  log.write('lr: ')
  lr=1e-6
  log.write(str(lr))
  log.write('\n')
  model = model.cuda()

  for param in model.parameters():
    param.requires_grad = False


  for param in model.rca.parameters():
      param.requires_grad = True

  for param in model.frpl.parameters():
      param.requires_grad = True

  for param in model.relation_projection.parameters():
      param.requires_grad = True


  log.write(
      '--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|\n'
  )
  log.write(
      '  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n'
  )
  log.write(
      '-------------------------------------------------------------------------------------------------------|\n'
  )
  start = timer()

  iter_per_epoch = 100

  param_groups = [

      {"params": model.clip_backbone.visual.conv1_alpha.parameters(), "lr": 1e-4},
      {"params": [p for n, p in model.named_parameters()
                  if p.requires_grad
                  and "clip_backbone.visual.conv1_alpha" not in n],
      "lr": lr},
  ]

  optimizer = optim.Adam(param_groups, weight_decay=1e-6)


  real_len_ref = []
  real_iter_ref = []
  for dataloader_real in train_dataloader:
    src_train_iter_real = iter(dataloader_real)
    real_iter_ref.append(src_train_iter_real)
    src_iter_per_epoch_real = len(src_train_iter_real)
    real_len_ref.append(src_iter_per_epoch_real)

  max_real = max(np.array(real_len_ref))
  print('max iteration number:', max_real)

  text_hist_loss = []
  view_hist_loss = []
  cls_hist_loss = []
  fg_sm_hist = []
  multi_label_hist = []

  epoch = 0
  for iter_num in range(0, args.iter + 1):
    for i in range(len(train_dataloader)):
      if (iter_num % real_len_ref[i] == 0):
        real_iter_ref[i] = iter(train_dataloader[i])

    if (iter_num != 0 and iter_num % iter_per_epoch == 0):
      epoch = epoch + 1

    model.train(True)


    input_image = []
    input_image_mask = []
    input_caption = []
    input_indicators = []

    for i in range(len(train_dataloader)):

      anchor, positive, negative,anchor_mask,positive_mask,negative_mask,label,template,indicators= next(real_iter_ref[i])

      real_inf_real_caption,real_inf_fake_caption,fake_inf_real_caption,fake_inf_fake_caption = template
      indicators_positive,indicators_negative = indicators

      anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
      anchor_mask,positive_mask,negative_mask = anchor_mask.cuda(),positive_mask.cuda(),negative_mask.cuda()
      input_image.append((anchor,negative,positive))
      input_image_mask.append((anchor_mask,positive_mask,negative_mask))
      input_caption.append((fake_inf_real_caption,fake_inf_fake_caption,real_inf_real_caption,real_inf_fake_caption))
      input_indicators.append((indicators_positive,indicators_negative))


    classifier_label_out,label,cls_loss,txt_loss,view_loss,fg_sm,multi_label = model.triplet(input_image,input_image_mask,input_caption,input_indicators,iter_num)
    loss_classifier.update(cls_loss.item())

    acc = accuracy(
        classifier_label_out,
        label,
        topk=(1,))
    classifer_top1.update(acc[0])

    if iter_num >= 500:
      total_loss = args.alpha* cls_loss
    else:


      total_loss = args.beta* txt_loss


    total_loss.backward()


    optimizer.step()
    optimizer.zero_grad()


    text_hist_loss.append(txt_loss.detach().cpu().numpy())


    cls_hist_loss.append(cls_loss.detach().cpu().numpy())
    fg_sm_hist.extend(fg_sm)
    multi_label_hist.extend(multi_label)
    if (iter_num == 0 or (iter_num + 1) % (iter_per_epoch) == 0):
        print('cls loss')
        cls_hist_loss = np.array(cls_hist_loss)
        print(cls_hist_loss.sum()/len(cls_hist_loss))
        print('txt loss')
        text_hist_loss = np.array(text_hist_loss)
        print(text_hist_loss.sum()/len(text_hist_loss))
        text_hist_loss = []
        view_hist_loss = []
        cls_hist_loss = []
        valid_args = eval(test_dataloader, model, True)

        is_best = valid_args[3] <= best_model_HTER
        best_model_HTER = min(valid_args[3], best_model_HTER)
        threshold,prob_list, label_list,inf_path_list = valid_args[5],valid_args[-3],valid_args[-2],valid_args[-1]

        if (valid_args[3] <= best_model_HTER):
            best_model_ACC = valid_args[6]
            best_model_AUC = valid_args[4]
            best_TPR_FPR = valid_args[-1]

        save_list = [
        epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER,
        threshold
        ]


        fg_sm_df = pd.DataFrame(fg_sm_hist)
        multi_label_df = pd.DataFrame(multi_label_hist)


        fg_sm_hist =[]
        multi_label_hist = []
        if is_best:
              score_df = pd.DataFrame({'prob_list':prob_list,'threshold':threshold,'label_list':label_list,'inf_path_list':inf_path_list})
              score_df.to_csv(os.path.join(args.op_dir, config.tgt_data + f'{args.method}_{args.layer}_cls_{str(args.alpha)}_triplet_{str(args.beta)}_view_{str(args.gamma)}_thres_{str(args.thres)}_lr_{str(lr)}_scores.csv'))
              fg_sm_df.to_csv(os.path.join(args.op_dir, config.tgt_data + f'{args.method}_{args.layer}_cls_{str(args.alpha)}_triplet_{str(args.beta)}_view_{str(args.gamma)}_thres_{str(args.thres)}_lr_{str(lr)}_fine_grained_pred.csv'))
              multi_label_df.to_csv(os.path.join(args.op_dir, config.tgt_data + f'{args.method}_{args.layer}_cls_{str(args.alpha)}_triplet_{str(args.beta)}_view_{str(args.gamma)}_{str(args.thres)}_lr_{str(lr)}_fine_grained_gt.csv'))
        save_checkpoint(save_list, is_best, model,os.path.join(args.op_dir, config.tgt_data + f'{args.method}_{args.layer}_shrink_pixel_{str(args.shrink_pixel)}_checkpoint_run_{str(config.run)}.pth.tar'))
        save_checkpoint(save_list, True, model,os.path.join(args.op_dir, config.tgt_data + f'{args.method}_{args.layer}_shrink_pixel_{str(args.shrink_pixel)}_checkpoint_run_{str(config.run)}_final.pth.tar'))


        print('\r', end='', flush=True)
        log.write(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s   %s'
            % ((iter_num + 1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100), time_to_str(timer() - start, 'min'), 0))
        log.write('\n')

        time.sleep(0.01)

  save_checkpoint(save_list, True, model,os.path.join(args.op_dir, config.tgt_data + f'{args.method}_{args.layer}_shrink_pixel_{str(args.shrink_pixel)}_checkpoint_run_{str(config.run)}_final.pth.tar'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str)
  parser.add_argument('--aug_path', type=str, default=None)
  parser.add_argument('--op_dir', type=str, default=None)

  parser.add_argument('--method', type=str, default='tram', help='method name used in saved artifacts')
  parser.add_argument('--iter', type=int, default=3000, help='number of iterations')
  parser.add_argument('--layer', type=int, default=6, help='number of cross layer')
  parser.add_argument('--alpha', type=float, help='weight of cls loss')
  parser.add_argument('--beta', type=float, help='weight of triplet loss')
  parser.add_argument('--gamma', type=float,default=1, help='weight of view loss')
  parser.add_argument('--thres', type=float,  default=0.95,help='threshold for image property')
  parser.add_argument('--shrink_pixel', type=int, default=10,help='shrink_pixel of mask')
  parser.add_argument('--pretrain_path', type=str, default=None)
  parser.add_argument('--batch_size', type=int, default=2 )


  args = parser.parse_args()

  global random_seed
  random_seed = 1024
  np.random.seed(random_seed)
  torch.manual_seed(random_seed)


  if args.config == 'I':
    config = configI
  if args.config == 'C':
    config = configC
  if args.config == 'M':
    config = configM
  if args.config == 'O':
    config = configO
  if args.config == 'cefa':
    config = config_cefa
  if args.config == 'surf':
    config = config_surf
  if args.config == 'wmca':
    config = config_wmca

  if args.config == 'CSW':
     config = config_CSW
  if args.config == 'SWC':
     config = config_SWC
  if args.config == 'CWS':
     config = config_CWS

  if args.config == 'WH':
    config = config_WH


  if args.config == 'CI':
    config = config_CI
  elif args.config == 'CO':
    config = config_CO
  elif args.config == 'CM':
    config = config_CM
  elif args.config == 'MC':
    config = config_MC
  elif args.config == 'MI':
    config = config_MI
  elif args.config == 'MO':
    config = config_MO
  elif args.config == 'IC':
    config = config_IC
  elif args.config == 'IM':
    config = config_IM
  elif args.config == 'IO':
    config = config_IO
  elif args.config == 'OC':
    config = config_OC
  elif args.config == 'OM':
    config = config_OM
  elif args.config == 'OI':
    config = config_OI

  for i in range(1):

    torch.manual_seed(i)
    np.random.seed(i)

    config.run = i

  train(config)


  config.op_dir = str(args.op_dir)
  if not os.path.exists(config.op_dir):
        os.mkdir(config.op_dir)
