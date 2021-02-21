import torch
import numpy as np
import cv2
import os
import math
import scipy.io as scio
import re
from PIL import Image, ImageDraw
from skimage.morphology import watershed
from config import cfg
from torch.utils import data
from torchvision import transforms
from model import CRAFT
from detect import resize_img, get_score


def get_gaussian(length=cfg.utils.gaussian_length, sigma_den=cfg.utils.sigma_den):
	x, y = np.meshgrid(np.linspace(0,length-1,length), np.linspace(0,length-1,length))
	xc = length / 2
	yc = length / 2
	sigma = (length + 1) / sigma_den
	g = np.exp(-((x-xc)**2 + (y-yc)**2) / (2*sigma**2))
	return g / np.max(g)


def cross_product_sign(v1, v2):
	return np.sign(v1[0] * v2[1] - v1[1] * v2[0])


def is_cross(line1, line2):
	# line1: [x1,y1,x2,y2](A,B)
	# line2: [x1,y1,x2,y2](C,D)
	AB = (line1[2] - line1[0], line1[3] - line1[1])
	AC = (line2[0] - line1[0], line2[1] - line1[1])
	AD = (line2[2] - line1[0], line2[3] - line1[1])
	
	CD = (line2[2] - line2[0], line2[3] - line2[1])
	CA = (line1[0] - line2[0], line1[1] - line2[1])
	CB = (line1[2] - line2[0], line1[3] - line2[1])
	
	flag1 = cross_product_sign(AB, AC) * cross_product_sign(AB, AD) <= 0
	flag2 = cross_product_sign(CD, CA) * cross_product_sign(CD, CB) <= 0
	return flag1 & flag2


def part_vertices(box, line):
	# bix -> 4x2 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ndarray
	if is_cross((box[0,0], box[0,1],box[1,0],box[1,1]), line) or \
       is_cross((box[2,0],box[2,1],box[3,0],box[3,1]), line):
		return ((1,2),(0,3))
	
	elif is_cross((box[1,0], box[1,1],box[2,0],box[2,1]), line) or \
       is_cross((box[0,0],box[0,1],box[3,0],box[3,1]), line):
		return ((0,1),(2,3))
	return None


def cal_angle(v1):
	theta = np.arccos(min(1, v1[0] / (np.linalg.norm(v1) + cfg.utils.eps)))
	return 2 * math.pi - theta if v1[1] < 0 else theta


def clockwise_sort(points):
	# return 4x2 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ndarray
	v1, v2, v3, v4 = points
	center = (v1 + v2 + v3 + v4) / 4
	theta = np.array([cal_angle(v1-center), cal_angle(v2-center), \
                      cal_angle(v3-center), cal_angle(v4-center)])
	index = np.argsort(theta)
	return np.array([v1,v2,v3,v4])[index,:]


def affinity_box_from_2quadri(box1, box2):
	# input x 2: 4x2 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ndarray
	# return 4x2 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ndarray
	center1 = box1.mean(0)
	center2 = box2.mean(0) 
	index1 = part_vertices(box1, (center1[0],center1[1],center2[0],center2[1]))
	index2 = part_vertices(box2, (center1[0],center1[1],center2[0],center2[1]))
	if index1 is None or index2 is None:
		return None

	v1 = (center1 + box1[index1[0][0]] + box1[index1[0][1]]) / 3
	v2 = (center1 + box1[index1[1][0]] + box1[index1[1][1]]) / 3
	v3 = (center2 + box2[index2[0][0]] + box2[index2[0][1]]) / 3
	v4 = (center2 + box2[index2[1][0]] + box2[index2[1][1]]) / 3
	return clockwise_sort((v1,v2,v3,v4))


def extract_affinity_boxes(char_boxes):
	affinity_boxes = []
	for i in range(len(char_boxes)-1):
		affinity_box = affinity_box_from_2quadri(char_boxes[i], char_boxes[i+1])
		if affinity_box is not None:
			affinity_boxes.append(affinity_box)
	return affinity_boxes


def warp_gaussian(g, box, w, h):
	if np.max(box[:,0]) - np.min(box[:,0]) < cfg.utils.min_box_len or \
       np.max(box[:,1]) - np.min(box[:,1]) < cfg.utils.min_box_len:
		return np.zeros((h, w))

	length = g.shape[0]
	pts1 = np.float32([[0,0], [length-1, 0], [length-1, length-1], [0, length-1]])
	box = clockwise_sort((box[0], box[1], box[2], box[3]))
	pts2 = np.float32([[box[0,0],box[0,1]], [box[1,0],box[1,1]], [box[2,0],box[2,1]], [box[3,0],box[3,1]]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	dst = cv2.warpPerspective(g, M, (w,h))
	return dst / np.max(dst) if np.max(dst) > cfg.utils.eps else np.zeros((h, w))


def get_word_text(text):
	words = []
	for i in range(len(text)):
		words.extend([word for word in re.split('\n| ',text[i].strip(' ')) if word != ''])
	return words


def check_valid(wordBB, charBB, words, chars):
	if len(words) != wordBB.shape[-1]:
		return False
	if len(chars) != charBB.shape[-1]:
		return False
	return True


def crop_img(img, vertices, crop_length):
	# confirm the shortest side of image >= length
	h, w = img.height, img.width
	if h >= w and w < crop_length:
		img = img.resize((crop_length, int(h * crop_length / w)), Image.BILINEAR)
	elif h < w and h < crop_length:
		img = img.resize((int(w * crop_length / h), crop_length), Image.BILINEAR)
	ratio_w = img.width / w
	ratio_h = img.height / h
	# find random position
	remain_h = img.height - crop_length
	remain_w = img.width - crop_length
	start_w = int(np.random.rand() * remain_w)
	start_h = int(np.random.rand() * remain_h)
	box = (start_w, start_h, start_w + crop_length, start_h + crop_length)
	region = img.crop(box)

	for vertice in vertices:
		if isinstance(vertice, list):
			for char in vertice:
				if char is None:
					continue
				char[0,:,:] = char[0,:,:] * ratio_w - start_w
				char[1,:,:] = char[1,:,:] * ratio_h - start_h
		else:
			vertice[0,:,:] = vertice[0,:,:] * ratio_w - start_w
			vertice[1,:,:] = vertice[1,:,:] * ratio_h - start_h
	return region


def plot_gaussian_score(g, score_map, boxes):
	for box in boxes: # box is 4x2 ndarray
		x_min = int(np.around(np.min(box[:,0]))) - cfg.utils.pixel_aug
		x_max = int(np.around(np.max(box[:,0]))) + cfg.utils.pixel_aug
		y_min = int(np.around(np.min(box[:,1]))) - cfg.utils.pixel_aug
		y_max = int(np.around(np.max(box[:,1]))) + cfg.utils.pixel_aug
		
		if x_min >= score_map.shape[1] or x_max < 0 or y_min >= score_map.shape[0] or y_max < 0:
			continue
		w = x_max - x_min + 1
		h = y_max - y_min + 1
		if w > score_map.shape[1] or h > score_map.shape[0]:
			continue
		
		temp_box = box - [x_min, y_min]
		warped_gaussian = warp_gaussian(g, temp_box, w, h)
		top_offset = max(0, -y_min)
		bot_offset = min(0, score_map.shape[0] - 1 - y_max) 
		lef_offset = max(0, -x_min) 
		rig_offset = min(0, score_map.shape[1] - 1 - x_max) 

		score_map[y_min+top_offset:y_max+1+bot_offset, x_min+lef_offset:x_max+1+rig_offset] = np.maximum( \
        score_map[y_min+top_offset:y_max+1+bot_offset, x_min+lef_offset:x_max+1+rig_offset], \
        warped_gaussian[top_offset:h+bot_offset, lef_offset:w+rig_offset])


def get_region_affinity(im_shape, wordBB, charBB, words):
	region_score = np.zeros(im_shape)
	affinity_score = np.zeros(im_shape)
	offset = 0
	total_char_boxes = []
	total_affinity_boxes = []
	
	for i, word in enumerate(words):
		char_boxes = []
		word_p1 = offset + 0
		word_p2 = offset + len(word) - 1
		for j in range(word_p1, word_p2+1):
			char_boxes.append(charBB[:,:,j].T)
		offset += len(word)
		total_char_boxes.extend(char_boxes)
		total_affinity_boxes.extend(extract_affinity_boxes(char_boxes))
		
	g = get_gaussian()
	plot_gaussian_score(g, region_score, total_char_boxes)
	plot_gaussian_score(g, affinity_score, total_affinity_boxes)
	return np.minimum(1, region_score), np.minimum(1, affinity_score)


class SynthTextDataset(data.Dataset):
	def __init__(self, cfg):
		super(SynthTextDataset, self).__init__()
		self.cfg       = cfg
		gt = scio.loadmat(os.path.join(self.cfg.synthtext_gt_path, 'gt.mat'))
		self.img_files = gt['imnames'][0]
		self.wordBB    = gt['wordBB'][0]
		self.charBB    = gt['charBB'][0]
		self.txt       = gt['txt'][0]
	
	def __len__(self):
		return len(self.img_files)

	def __getitem__(self, index):
		img = Image.open(os.path.join(self.cfg.synthtext_img_path, self.img_files[index][0]))
		wordBB = self.wordBB[index].reshape((2,4,-1))
		charBB = self.charBB[index].reshape((2,4,-1))
		txt = self.txt[index]
		words = get_word_text(txt)
		chars = ''.join(words)
		assert(check_valid(wordBB, charBB, words, chars))
		
		conf_map = np.ones((int(self.cfg.scale*self.cfg.crop_length), int(self.cfg.scale*self.cfg.crop_length)))
		vertices = [wordBB, charBB]
		img = crop_img(img, vertices, self.cfg.crop_length)
		t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.cfg.mean, self.cfg.std)])
		region_score, affinity_score = get_region_affinity(conf_map.shape, self.cfg.scale*vertices[0], self.cfg.scale*vertices[1], words)
		res = list(map(lambda x: torch.Tensor(x).unsqueeze(0), [region_score, affinity_score, conf_map]))
		return t(img), res[0], res[1], res[2]


#----------------------------------------------------------------------------------------------------------------------------------------


def get_rotate_mat(theta):
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertice, theta, anchor):
	rotate_mat = get_rotate_mat(theta)
	res = np.dot(rotate_mat, vertice - anchor)
	return res + anchor


def rotate_img(img, vertices, angle_range):
	center_x = (img.width - 1) / 2
	center_y = (img.height - 1) / 2
	angle = angle_range * (np.random.rand() * 2 - 1)
	img = img.rotate(angle, Image.BILINEAR)
	
	for i in range(vertices[0].shape[-1]):
		vertices[0][:,:,i] = rotate_vertices(vertices[0][:,:,i], -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
	for char in vertices[1]:
		if char is None:
			continue
		for i in range(char.shape[-1]):
			char[:,:,i] = rotate_vertices(char[:,:,i], -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
	return img


def adjust_resolution(img, vertices, min_length, max_length):
	long_side = int(np.random.rand() * (max_length -  min_length)) +  min_length
	img, ratio_h, ratio_w = resize_img(img, long_side)
	vertices[0][0,:,:] *= ratio_w
	vertices[0][1,:,:] *= ratio_h
	for char in vertices[1]:
		if char is None:
			continue
		char[0,:,:] *= ratio_w
		char[1,:,:] *= ratio_h
	return img


def adjust_height(img, vertices, ratio):
	ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
	old_h = img.height
	new_h = int(np.around(old_h * ratio_h))
	img = img.resize((img.width, new_h), Image.BILINEAR)
	
	vertices[0][1,:,:] *= new_h / old_h
	for char in vertices[1]:
		if char is None:
			continue
		char[1,:,:] *= new_h / old_h
	return img


def order_sort(char_boxes, centers, scope):
	# char_boxes: list of 2x4 ndarray, centers: nx2
	if not char_boxes:
		return char_boxes
	sorted_boxes = []
	assert(centers.shape[0] == len(char_boxes))
	h, w = scope
	hori_error = np.sum(np.abs(centers[:, 1] - (h/2)))
	vert_error = np.sum(np.abs(centers[:, 0] - (w/2)))
	if hori_error < vert_error: # chars are horizontal
		index = np.argsort(centers[:, 0])
	else: # chars are vertical
		index = np.argsort(centers[:, 1])
	for i in index:
		sorted_boxes.append(char_boxes[i])
	return sorted_boxes


def segment_region_score(region_score, MI, ratio_hw):
	fore = np.uint8(region_score > cfg.utils.fore_ratio)
	back = np.uint8(region_score < cfg.utils.back_ratio)
	unknown = 1 - (fore + back)
	ret, markers = cv2.connectedComponents(fore)
	markers += 1
	markers[unknown==1] = 0
	labels = watershed(-region_score, markers)
	char_boxes = []
	centers = []
	for label in range(2, ret+1):
		y, x = np.where(labels == label)
		x_max = x.max()
		y_max = y.max()
		x_min = x.min()
		y_min = y.min()
		w = x_max - x_min + 1
		h = y_max - y_min + 1
		centers.append([(x_min + x_max) / 2, (y_min + y_max) / 2])
		cords = np.array([[x_min, x_max, x_max, x_min],[y_min, y_min, y_max, y_max]]) / cfg.ft.scale
		cords[0,:] /= ratio_hw[1]
		cords[1,:] /= ratio_hw[0]
		char_box = np.dot(MI, np.concatenate((cords, np.array([[1,1,1,1]])), axis=0))
		char_boxes.append((char_box / np.tile(char_box[2,:], (3,1)))[:2,:])
	char_boxes = order_sort(char_boxes, np.array(centers), region_score.shape[:2])
	return np.array(char_boxes).transpose((1,2,0)) if char_boxes else None


def cal_distance(x1, y1, x2, y2):
	return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def get_charBB(img, wordBB, model, device):
	model.eval()
	charBB = [] # list of 2x4xn ndarray or None
	if wordBB.size == 0:
		return charBB
	for i in range(wordBB.shape[-1]):
		word = wordBB[:,:,i]
		pts_ori = np.float32([[word[0,0],word[1,0]], [word[0,1],word[1,1]], [word[0,2],word[1,2]], [word[0,3],word[1,3]]])
		w = int(max(cal_distance(word[0,0], word[1,0], word[0,1], word[1,1]), \
                    cal_distance(word[0,2], word[1,2], word[0,3], word[1,3])))
		h = int(max(cal_distance(word[0,1], word[1,1], word[0,2], word[1,2]), \
                    cal_distance(word[0,0], word[1,0], word[0,3], word[1,3])))
		pts_crop = np.float32([[0,0], [w, 0], [w, h], [0, h]])
		M = cv2.getPerspectiveTransform(pts_ori, pts_crop)
		MI = cv2.getPerspectiveTransform(pts_crop, pts_ori)
		
		crop_word = cv2.warpPerspective(img, M, (w,h))
		crop_word, ratio_h, ratio_w = resize_img(Image.fromarray(crop_word), None)
		region_score, _ = get_score(crop_word, model, device)
		temp_charBB = segment_region_score(region_score, MI, (ratio_h, ratio_w))
		charBB.append(temp_charBB)
	return charBB


def reorg(wordBB, lw):
	# wordBB : 2x4 ndarray
	# return 2x4xn charBB
	x1, x2, x3, x4 = wordBB[0]
	y1, y2, y3, y4 = wordBB[1]
	char = np.zeros((2,4,lw))
	if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
       cal_distance(x1, y1, x4, y4) + cal_distance(x2, y2, x3, y3):
		new_x_A = x1 + np.arange(lw + 1) * (x2 - x1) / lw
		new_y_A = y1 + np.arange(lw + 1) * (y2 - y1) / lw
		new_x_B = x4 + np.arange(lw + 1) * (x3 - x4) / lw
		new_y_B = y4 + np.arange(lw + 1) * (y3 - y4) / lw
	else:
		new_x_A = x1 + np.arange(lw + 1) * (x4 - x1) / lw
		new_y_A = y1 + np.arange(lw + 1) * (y4 - y1) / lw
		new_x_B = x2 + np.arange(lw + 1) * (x3 - x2) / lw
		new_y_B = y2 + np.arange(lw + 1) * (y3 - y2) / lw
	for i in range(lw):
		char[0,0,i] = new_x_A[i]
		char[0,1,i] = new_x_A[i+1]
		char[0,2,i] = new_x_B[i+1]
		char[0,3,i] = new_x_B[i]
		char[1,0,i] = new_y_A[i]
		char[1,1,i] = new_y_A[i+1]
		char[1,2,i] = new_y_B[i+1]
		char[1,3,i] = new_y_B[i]
	return char


def get_region_affinity_conf(conf_map, wordBB, words, charBB, scale):
	region_score = np.zeros(conf_map.shape)
	affinity_score = np.zeros(conf_map.shape)
	total_char_boxes = []
	total_affinity_boxes = []
	
	for i, word in enumerate(words): # gt context
		if word in cfg.utils.ignored_gt: # ignored
			conf = 0
		else:
			char_num = 0 if charBB[i] is None else charBB[i].shape[-1]
			lw = len(word)
			conf = (lw - min(lw, abs(lw - char_num))) / lw
			if conf < cfg.ft.conf_min:
				conf = cfg.ft.conf_min
				charBB[i] = reorg(wordBB[:,:,i], lw)
		conf_poly = np.around(scale * wordBB[:,:,i].T).astype(np.int32)
		cv2.fillPoly(conf_map, [conf_poly], conf)
		if word in cfg.utils.ignored_gt: # ignored
			continue
		char_boxes = []
		for j in range(charBB[i].shape[-1]):
			char_boxes.append(scale * charBB[i][:,:,j].T)
		total_char_boxes.extend(char_boxes)
		total_affinity_boxes.extend(extract_affinity_boxes(char_boxes))
		
	g = get_gaussian()
	plot_gaussian_score(g, region_score, total_char_boxes)
	plot_gaussian_score(g, affinity_score, total_affinity_boxes)
	return np.minimum(1, region_score), np.minimum(1, affinity_score)


def _get_synthset():
	return SynthTextDataset(cfg.train)


def _merge_datasets(img_path_13, gt_path_13, img_path_17, gt_path_17):
	total_img_files = []
	total_gt_files = []
	
	total_img_files.extend([os.path.join(img_path_13, img_file) for img_file in sorted(os.listdir(img_path_13))])
	total_gt_files.extend([os.path.join(gt_path_13, gt_file) for gt_file in sorted(os.listdir(gt_path_13))])
	num_13 = len(total_img_files)
	for img_17, gt_17 in zip(img_path_17, gt_path_17):
		total_img_files.extend([os.path.join(img_17, img_file) for img_file in sorted(os.listdir(img_17))])
		total_gt_files.extend([os.path.join(gt_17, gt_file) for gt_file in sorted(os.listdir(gt_17))])
	return total_img_files, total_gt_files, num_13


def extract_vertices_13_17(lines, index, part_num):
	vertices = []
	texts = []
	if index < part_num: # 2013 dataset
		for line in lines:
			info = line.rstrip('\n').split(' ', 4)
			assert(len(info) == 5)
			cords = list(map(float, info[:4]))
			vertices.append(np.array([[cords[0], cords[2], cords[2], cords[0]], [cords[1], cords[1], cords[3], cords[3]]]))
			texts.append(info[4].strip('"'))
	else: # 2017 dataset
		for line in lines:
			info = line.rstrip('\n').split(',', 9)
			assert(len(info) == 10)
			vertices.append(np.array(list(map(float, info[:8]))).reshape((4,2)).T)
			texts.append(info[9])
	return np.array(vertices).transpose((1,2,0)), texts


class ICDAR2013_2017Dataset(data.Dataset):
	def __init__(self, cfg):
		super( ICDAR2013_2017Dataset, self).__init__()
		self.cfg       = cfg
		self.synthset  = _get_synthset()
		imgs, gts, part_num = _merge_datasets(cfg.icdar2013_img_path, cfg.icdar2013_gt_path, cfg.icdar2017_img_path, cfg.icdar2017_gt_path)
		self.img_files = imgs 
		self.gt_files  = gts
		self.part_num  = part_num

	def __len__(self):
		return min(self.part_num * (self.cfg.ratio_17 + 1), len(self.img_files))

	def __getitem__(self, index):
		with open(self.gt_files[index], 'r') as f:
			lines = f.readlines()
		wordBB, words = extract_vertices_13_17(lines, index, self.part_num)
		img = Image.open(self.img_files[index]).convert('RGB')
		charBB = get_charBB(np.array(img), wordBB, self.model, self.device) # list of (2,4,n)
		vertices = [wordBB, charBB]
		
		img = adjust_resolution(img, vertices, self.cfg.min_length, self.cfg.max_length)
		img = adjust_height(img, vertices, self.cfg.height_jitter)
		img = rotate_img(img, vertices, self.cfg.angle_range)
		img = crop_img(img, vertices, self.cfg.crop_length)

		conf_map = np.ones((int(self.cfg.scale*self.cfg.crop_length), int(self.cfg.scale*self.cfg.crop_length)))
		region_score, affinity_score = get_region_affinity_conf(conf_map, wordBB, words, charBB, self.cfg.scale)
		t = transforms.Compose([transforms.ColorJitter(self.cfg.color_jitter[0], self.cfg.color_jitter[1], \
                                                       self.cfg.color_jitter[2], self.cfg.color_jitter[3]), \
                                transforms.ToTensor(), transforms.Normalize(self.cfg.mean, self.cfg.std)])
		res = list(map(lambda x: torch.Tensor(x).unsqueeze(0), [region_score, affinity_score, conf_map]))
		return t(img), res[0], res[1], res[2]
	
	def update_model(self, model):
		self.model = model
	
	def update_device(self, device):
		self.device = device

	def reorg_datasets(self):
		num_13 = self.part_num
		img_files_17 = self.img_files[num_13:]
		gt_files_17 = self.gt_files[num_13:]
		files_17 = list(zip(img_files_17, gt_files_17))
		np.random.shuffle(files_17)
		img_files_17, gt_files_17 = zip(*files_17)
		self.img_files[num_13:] = list(img_files_17)
		self.gt_files[num_13:] = list(gt_files_17)
