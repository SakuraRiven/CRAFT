import torch
from torch import nn
from torch.utils import data
from torch.optim import lr_scheduler
from model import CRAFT
import os
import time
import numpy as np
from config import cfg
from dataset import SynthTextDataset, ICDAR2013_2017Dataset
from sync_batchnorm import convert_model
from loss import Loss_OHEM
from PIL import Image


def freeze_model(model, freeze_stage_num):
	backbone = list(list(model.children())[0].children())[0]
	cnt = -1
	for m in backbone.children():
		for param in m.parameters():
			param.requires_grad = False
		
		if isinstance(m, nn.MaxPool2d):
			cnt += 1
			if cnt == freeze_stage_num:
				break


def finetune(cfg, synth_trainset):
	assert(len(cfg.gpu_ids) > 1)
	assert(len(cfg.gpu_ids) <= torch.cuda.device_count())
	print('{} GPUs are available, {} GPUs are used'.format(torch.cuda.device_count(), len(cfg.gpu_ids)))

	supervision_device = torch.device("cuda:0")
	supervision_model = CRAFT()
	supervision_model.load_state_dict(torch.load(cfg.pretrain_pth))
	supervision_model.to(supervision_device)

	train_device = torch.device("cuda:1")
	train_model = CRAFT()
	train_model = convert_model(train_model)
	train_model.load_state_dict(torch.load(cfg.pretrain_pth))
	freeze_model(train_model, cfg.freeze_stage_num)
	data_parallel = False
	if torch.cuda.device_count() > 2:
		train_model = nn.DataParallel(train_model, device_ids=cfg.gpu_ids[1:])
		data_parallel = True
	train_model.to(train_device)

	trainset = ICDAR2013_2017Dataset(cfg)
	print('len trainset is {}'.format(len(trainset)))
	trainset.update_model(supervision_model)
	trainset.update_device(supervision_device)

	train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, drop_last=cfg.drop_last)
	synth_loader = data.DataLoader(synth_trainset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, drop_last=cfg.drop_last)
	file_num = len(trainset)
	batch_num = int(file_num/cfg.batch_size)
	
	criterion = Loss_OHEM(cfg.neg_ratio, cfg.pos_min)
	optimizer = torch.optim.Adam(train_model.parameters(), lr=cfg.lr)

	cnt = 0
	start_time = 0
	for epoch in range(cfg.epoch_iter):	
		epoch_loss = 0
		epoch_time = time.time()
		trainset.reorg_datasets()
		for i, (img, gt_region, gt_affinity, conf_map) in enumerate(train_loader):
			train_model.train()
			
			if np.random.rand() < cfg.mix_prob:
				print('synth data')
				for img, gt_region, gt_affinity, conf_map in synth_loader:
					break

			img, gt_region, gt_affinity, conf_map = list(map(lambda x: x.to(train_device), [img, gt_region, gt_affinity, conf_map]))
			pred_region, pred_affinity = train_model(img)
			loss = criterion(gt_region, pred_region, gt_affinity, pred_affinity, conf_map)
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.6f}, batch_loss is {:.6f}'.format(\
                epoch+1, cfg.epoch_iter, i+1, batch_num, time.time()-start_time, loss.item()))
			cnt += 1
			start_time = time.time()
			state_dict = train_model.module.state_dict() if data_parallel else train_model.state_dict()
			supervision_model.load_state_dict(state_dict)
			trainset.update_model(supervision_model)
			if cnt % cfg.save_interval == 0:
				torch.save(state_dict, os.path.join(cfg.pths_path, 'model_iter_{}.pth'.format(cnt)))
				print(time.asctime(time.localtime(time.time())))

		print('epoch_loss is {:.6f}, epoch_time is {:.6f}'.format(epoch_loss/batch_num, time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*30)


if __name__ == '__main__':
	synth_trainset = SynthTextDataset(cfg.train)
	finetune(cfg.ft, synth_trainset)
