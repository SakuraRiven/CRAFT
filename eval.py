import time
import torch
import subprocess
import os
from model import CRAFT
from detect import detect_dataset
import shutil
from config import cfg


def eval_model(cfg):
	if os.path.exists(cfg.submit_path):
		shutil.rmtree(cfg.submit_path) 
	os.mkdir(cfg.submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = CRAFT(False).to(device)
	model.load_state_dict(torch.load(cfg.model_pth, map_location='cuda:0'))
	model.eval()
						
	start_time = time.time()
	detect_dataset(model, device, cfg.submit_path, cfg)
	print('eval time is {}'.format(time.time()-start_time))	
	os.chdir(cfg.submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	fscore = float(res.split(':')[-1].strip().strip('}'))
	print(res + '\n')

	if not cfg.save_dataset_res:
		os.remove('./submit.zip')
		shutil.rmtree(cfg.submit_path)


if __name__ == '__main__': 
	eval_model(cfg.test)
