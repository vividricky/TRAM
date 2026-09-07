import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable
from torch.nn import functional as F

from utils.statistic import get_EER_states, get_HTER_at_thr, calculate_threshold
from utils.utils import AverageMeter, accuracy


def eval(valid_dataloader, model, norm_flag=True, name=None, return_prob=True):
  criterion = nn.CrossEntropyLoss()
  valid_losses = AverageMeter()
  valid_top1 = AverageMeter()
  model.eval()

  prob_dict = {}
  label_dict = {}
  output_dict_tmp = {}
  target_dict_tmp = {}
  prob_list = []
  label_list = []
  inf_path_list = []

  with torch.no_grad():
    for _, (inf_path, inf_img, ref_img, inf_mask, ref_mask, target, template, video_id) in enumerate(valid_dataloader):
      inf_img = inf_img.cuda()
      ref_img = ref_img.cuda()
      inf_mask = inf_mask.cuda()
      ref_mask = ref_mask.cuda()

      for path in inf_path:
        inf_path_list.append(path)

      eval_real_template, eval_fake_template = template
      prompt = [eval_real_template[0], eval_fake_template[0]]
      cls_out = model(ref_img, inf_img, ref_mask, inf_mask, prompt)
      target = Variable(torch.from_numpy(np.array(target)).long()).cuda()

      prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
      label = target.cpu().data.numpy()
      video_id = np.array(video_id)

      for i in range(len(prob)):
        if video_id[i] not in prob_dict:
          prob_dict[video_id[i]] = []
          label_dict[video_id[i]] = []
          output_dict_tmp[video_id[i]] = []
          target_dict_tmp[video_id[i]] = []
        prob_dict[video_id[i]].append(prob[i])
        label_dict[video_id[i]].append(label[i])
        output_dict_tmp[video_id[i]].append(cls_out[i].view(1, 2))
        target_dict_tmp[video_id[i]].append(target[i].view(1))

  prob_list = []
  label_list = []
  for key in prob_dict.keys():
    avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
    avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
    prob_list = np.append(prob_list, avg_single_video_prob)
    label_list = np.append(label_list, avg_single_video_label)

    avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
    avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
    loss = criterion(avg_single_video_output, avg_single_video_target.long())
    acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
    valid_losses.update(loss.item())
    valid_top1.update(acc_valid[0])

  auc_score = roc_auc_score(label_list, prob_list)

  prob_list = np.array(prob_list)
  label_list = np.array(label_list)
  cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
  ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
  cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

  fpr, tpr, _ = roc_curve(label_list, prob_list)
  tpr_filtered = tpr[fpr <= 1 / 100]
  rate = 0 if len(tpr_filtered) == 0 else tpr_filtered[-1]
  print("TPR@FPR = ", rate)

  if not return_prob:
    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate
    ]

  inf_path_list = np.array(inf_path_list)
  return [
      valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
      auc_score, threshold, ACC_threshold * 100, rate, prob_list, label_list,
      inf_path_list.flatten()
  ]
