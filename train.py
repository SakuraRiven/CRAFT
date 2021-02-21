import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from model import CRAFT
from loss import Loss
import os
import time
import numpy as np
from config import cfg
from dataset import SynthTextDataset
from sync_batchnorm import convert_model


def train(cfg):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = CRAFT()
	model = convert_model(model)
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)

	trainset = SynthTextDataset(cfg)
	train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, \
                                   num_workers=cfg.num_workers, drop_last=cfg.drop_last)
	file_num = len(trainset)
	batch_num = int(file_num/cfg.batch_size)
	criterion = Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(batch_num * i) for i in cfg.milestones], gamma=cfg.gamma)
	
	cnt = 0
	for epoch in range(cfg.epoch_iter):	
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_region, gt_affinity, conf_map) in enumerate(train_loader):
			model.train()
			scheduler.step()
			start_time = time.time()
			img, gt_region, gt_affinity, conf_map = list(map(lambda x: x.to(device), [img, gt_region, gt_affinity, conf_map]))
			pred_region, pred_affinity = model(img)
			loss = criterion(gt_region, pred_region, gt_affinity, pred_affinity, conf_map)
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, cfg.epoch_iter, i+1, batch_num, time.time()-start_time, loss.item()))
			cnt += 1
			if cnt % cfg.save_interval == 0:
				state_dict = model.module.state_dict() if data_parallel else model.state_dict()
				torch.save(state_dict, os.path.join(cfg.pths_path, 'model_iter_{}.pth'.format(cnt)))
				print(time.asctime(time.localtime(time.time())))

		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/batch_num, time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)


if __name__ == '__main__':
	train(cfg.train)
