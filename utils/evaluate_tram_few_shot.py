from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_curve

from timeit import default_timer as timer
import time
from utils.utils import time_to_str
import math
import os
import pandas as pd
import csv

def eval(valid_dataloader, model, norm_flag, name,return_prob=True):

  criterion = nn.CrossEntropyLoss()
  valid_losses = AverageMeter()
  valid_top1 = AverageMeter()
  model.eval()

  prob_dict = {}
  label_dict = {}
  output_dict_tmp = {}
  target_dict_tmp = {}

  csv_rows = []
  video_rows = []

  prob_list = []
  label_list = []
  output_list_temp = []
  target_list_temp = []
  inf_path_list = []
  with torch.no_grad():
    for iter, (inf_path, inf_img, ref_img_list, inf_mask,ref_mask_list,target, template,videoID) in enumerate(valid_dataloader):
      inf_img = inf_img.cuda()
      inf_mask = inf_mask.cuda()


      for path in inf_path:
         inf_path_list.append(path)

      eval_real_template,eval_fake_template = template
      template = [eval_real_template[0],eval_fake_template[0]]

      per_ref_similarity = model.forward_5_shots(ref_img_list,inf_img,ref_mask_list,inf_mask,template,None)


      stacked = torch.stack(per_ref_similarity, dim=0)


      cls_out = stacked.mean(dim=0)


      target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
      starttime = time.time()


      prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
      label = target.cpu().data.numpy()
      videoID = np.array(videoID)


      for i in range(len(prob)):
        if (videoID[i] in prob_dict.keys()):
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])
          output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
          target_dict_tmp[videoID[i]].append(target[i].view(1))
        else:
          prob_dict[videoID[i]] = []
          label_dict[videoID[i]] = []
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])
          output_dict_tmp[videoID[i]] = []
          target_dict_tmp[videoID[i]] = []
          output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
          target_dict_tmp[videoID[i]].append(target[i].view(1))

          row = {
              "inf_path": inf_path[i],
              "videoID": videoID[i],
              "prob_fake": float(prob[i]),
              "label": int(label[i])
          }


          for k in range(len(stacked)):
              row[f"ref_{k}"] = stacked[k][i].detach().cpu().numpy()

          csv_rows.append(row)


  prob_list = []
  label_list = []
  for key in prob_dict.keys():
    avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
    avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
    prob_list = np.append(prob_list, avg_single_video_prob)
    label_list = np.append(label_list, avg_single_video_label)

    avg_single_video_output = sum(output_dict_tmp[key]) / len(
        output_dict_tmp[key])
    avg_single_video_target = sum(target_dict_tmp[key]) / len(
        target_dict_tmp[key])
    loss = criterion(avg_single_video_output, avg_single_video_target.long())
    acc_valid = accuracy(
        avg_single_video_output, avg_single_video_target, topk=(1,))
    valid_losses.update(loss.item())
    valid_top1.update(acc_valid[0])

    video_rows.append({
        "videoID": key,
        "prob_fake": float(avg_single_video_prob),
        "label": int(avg_single_video_label)
    })


  auc_score = roc_auc_score(label_list, prob_list)

  prob_list = np.array(prob_list)
  label_list = np.array(label_list)
  cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
  ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
  cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

  fpr, tpr, thr = roc_curve(label_list, prob_list)
  tpr_filtered = tpr[fpr <= 1 / 100]
  if len(tpr_filtered) == 0:
    rate = 0
  else:
    rate = tpr_filtered[-1]
  print("TPR@FPR = ", rate)


  os.makedirs("/qiruiwang/exp/fas/Tram/analysis/", exist_ok=True)
  df_frame = pd.DataFrame(csv_rows)
  df_frame["pred"] = (df_frame["prob_fake"] > threshold).astype(int)
  df_frame["is_correct"] = (df_frame["pred"] == df_frame["label"]).astype(int)
  df_frame["threshold"] = threshold

  if name is not None:
    df_frame.to_csv("/qiruiwang/exp/fas/Tram/analysis/frame_results_"+name+"_5shots.csv", index=False)


  if not return_prob:
    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate
    ]
  else:
    inf_path_list = np.array(inf_path_list)
    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate, prob_list, label_list,inf_path_list.flatten()]
