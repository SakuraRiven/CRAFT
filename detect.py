import os
import cv2
import math
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import CRAFT
from config import cfg


def plot_boxes(img, boxes):
	if boxes is None:
		return img
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	return img


def resize_img(img, long_side):
	w, h = img.size
	if long_side is not None:
		if w > h:
			resize_w = long_side
			ratio = long_side / w
			resize_h = h * ratio
		else:
			resize_h = long_side
			ratio = long_side / h
			resize_w = w * ratio
	else:
		resize_h, resize_w = h, w

	final_h = int(resize_h) if resize_h % 32 == 0 else (int(resize_h / 32) + 1) * 32
	final_w = int(resize_w) if resize_w % 32 == 0 else (int(resize_w / 32) + 1) * 32
	img = img.resize((final_w, final_h), Image.BILINEAR)
	ratio_h = final_h / h
	ratio_w = final_w / w
	return img, ratio_h, ratio_w


def load_pil(img):
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.train.mean, cfg.train.std)])
	return t(img).unsqueeze(0)


def get_score(img, model, device):
	with torch.no_grad():
		region, affinity = model(load_pil(img).to(device))
	return list(map(lambda x: x[0][0].cpu().numpy(), [region, affinity]))


def restore_boxes(region, affinity, region_thresh, affinity_thresh, remove_thresh, ratio):
	# return [[x1, y1, x2, y2, x3, y3, x4, y4], [], ...]
	boxes = []
	M = (region > region_thresh) + (affinity > affinity_thresh)
	ret, markers = cv2.connectedComponents(np.uint8(M * 255))
	for i in range(ret):
		if i == 0:
			continue
		y,x=np.where(markers==i)
		if len(y) < region.size * remove_thresh:
			continue
		cords = 2 * np.concatenate((x.reshape(-1,1)/ratio[1], y.reshape(-1,1)/ratio[0]), axis=1)
		a = np.array([cords[:,0].min(), cords[:,1].min(), cords[:,0].max(), cords[:,1].min(), cords[:,0].max(), cords[:,1].max(), cords[:,0].min(), cords[:,1].max()])
		boxes.append(a)
	return boxes


def detect_single_image(img, model, device, cfg):
	img, ratio_h, ratio_w = resize_img(img, cfg.long_side)
	region, affinity = get_score(img, model, device)
	boxes = restore_boxes(region, affinity, cfg.region_thresh, cfg.affinity_thresh, cfg.remove_thresh, (ratio_h, ratio_w))
	return boxes


def detect_dataset(model, device, submit_path, cfg, th1=None, th2=None, th3=None):
	img_files = os.listdir(cfg.dataset_test_path)
	img_files = sorted([os.path.join(cfg.dataset_test_path, img_file) for img_file in img_files])
			
	for i, img_file in enumerate(img_files):
		print('evaluating {} image'.format(i), end='\r')
		boxes = detect_single_image(Image.open(img_file), model, device, cfg)
		seq = []
		for box in boxes:
			x_min = min(box[0], box[2], box[4], box[6])
			x_max = max(box[0], box[2], box[4], box[6])
			y_min = min(box[1], box[3], box[5], box[7])
			y_max = max(box[1], box[3], box[5], box[7])
			seq.append(','.join([str(int(v)) for v in [x_min, y_min, x_max, y_max]]) + '\n')
		with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
			f.writelines(seq)


if __name__ == '__main__':
	img_files = [os.path.join(cfg.test.dataset_test_path, img_file) for img_file in sorted(os.listdir(cfg.test.dataset_test_path))]
	img_path = np.random.choice(img_files)
	img_path = '../data/ICDAR2013/test_img/img_113.jpg'
	
	model_path  = './pths/pretrain/model_iter_50000.pth'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = CRAFT().to(device)
	model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
	
	model.eval()
	img = Image.open(img_path)
	boxes = detect_single_image(img, model, device, cfg.test)
	img = plot_boxes(img, boxes)
	img.save('res.bmp')
