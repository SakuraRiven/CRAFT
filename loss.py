import torch
import torch.nn as nn


class Loss(nn.Module):
	def __init__(self):
		super(Loss, self).__init__()

	def forward(self, gt_region, pred_region, gt_affinity, pred_affinity, conf_map):
		loss = torch.mean(((gt_region - pred_region).pow(2) + (gt_affinity - pred_affinity).pow(2)) * conf_map)
		return loss 


def get_loss(gt, pred, conf_map, neg_ratio, pos_min):
	b, c, h, w = gt.size()
	gt_pos_area = (gt > pos_min).float().view(-1)
	gt_pos_num = gt_pos_area.sum()
	gt_neg_num = b * c * h * w - gt_pos_num
	gt_neg_num = torch.min(gt_neg_num, neg_ratio * gt_pos_num)

	loss = ((gt - pred).pow(2) * conf_map).view(-1)
	pos_loss = loss * gt_pos_area
	neg_loss = loss * (1 - gt_pos_area)

	value, _ = torch.topk(neg_loss, int(gt_neg_num.item()), sorted=False)
	ohem_loss = value.sum() + pos_loss.sum()
	return ohem_loss / (gt_neg_num + gt_pos_num)
		

class Loss_OHEM(nn.Module):
	def __init__(self, neg_ratio, pos_min):
		super(Loss_OHEM, self).__init__()
		self.neg_ratio = neg_ratio
		self.pos_min = pos_min

	def forward(self, gt_region, pred_region, gt_affinity, pred_affinity, conf_map):
		region_loss = get_loss(gt_region, pred_region, conf_map, self.neg_ratio, self.pos_min)
		affinity_loss = get_loss(gt_affinity, pred_affinity, conf_map, self.neg_ratio, self.pos_min)
		print('region loss is {}, affinity loss is {}'.format(region_loss, affinity_loss))
		return region_loss + affinity_loss
