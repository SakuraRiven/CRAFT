import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


def _init_parameters(net):
	for m in net:
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, 0, 0.01)
			nn.init.constant_(m.bias, 0)


class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)
		_init_parameters(self.modules())


class extractor(nn.Module):
	def __init__(self, pretrained):
		super(extractor, self).__init__()
		vgg16_bn = VGG(make_layers(cfg))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('./pths/backbone/vgg16_bn-6c64b313.pth'))
		self.features = vgg16_bn.features
		
		self.conv_6 = nn.Sequential(
					  nn.Conv2d(512, 512, 3, padding=1, bias=False),
					  nn.BatchNorm2d(512),
					  nn.ReLU(inplace=True),
					  nn.Conv2d(512, 512, 3, padding=1, bias=False),
					  nn.BatchNorm2d(512),
					  nn.ReLU(inplace=True))
		_init_parameters(self.conv_6.modules())
	
	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		x = self.conv_6(x)
		out.append(x)
		return out[1:]

	
class upconv_block(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(upconv_block, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channel, out_channel*2, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channel*2)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channel*2, out_channel, 3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channel)
		self.relu2 = nn.ReLU(inplace=True)
				
	def forward(self, x):
		return self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))))
		 
		
class merge(nn.Module):
	def __init__(self):
		super(merge, self).__init__()

		self.upconv1 = upconv_block(1024, 256)
		self.upconv2 = upconv_block(512+256, 128)
		self.upconv3 = upconv_block(256+128, 64)
		self.upconv4 = upconv_block(128+64, 32)
		
		self.conv = nn.Sequential(
					nn.Conv2d(32, 32, 3, padding=1, bias=False),
					nn.BatchNorm2d(32),
					nn.ReLU(inplace=True),
					nn.Conv2d(32, 32, 3, padding=1, bias=False),
					nn.BatchNorm2d(32),
					nn.ReLU(inplace=True),
					nn.Conv2d(32, 16, 3, padding=1, bias=False),
					nn.BatchNorm2d(16),
					nn.ReLU(inplace=True),
					nn.Conv2d(16, 16, 1, bias=False),
					nn.BatchNorm2d(16),
					nn.ReLU(inplace=True))
		
		self.region_head = nn.Conv2d(16, 1, 1)
		self.affinity_head = nn.Conv2d(16, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.sigmoid2 = nn.Sigmoid()
		_init_parameters(self.modules())
					
	def forward(self, x):
		y = torch.cat((x[4], x[3]), 1)
		y = self.upconv1(y)
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		
		y = torch.cat((y, x[2]), 1)
		y = self.upconv2(y)
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		
		y = torch.cat((y, x[1]), 1)
		y = self.upconv3(y)
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		
		y = torch.cat((y, x[0]), 1)
		y = self.upconv4(y)
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		
		y = self.conv(y)
		region_score = self.sigmoid1(self.region_head(y))
		affinity_score = self.sigmoid2(self.affinity_head(y))
		
		return region_score, affinity_score


class CRAFT(nn.Module):
	def __init__(self, pretrained=True):
		super(CRAFT, self).__init__()
		self.extractor = extractor(pretrained)
		self.merge	 = merge()

	def forward(self, x):
		return self.merge(self.extractor(x))

if __name__ == '__main__':
	m = CRAFT(True)
	x = torch.randn(1, 3, 512, 512)
	region_score, affinity_score = m(x)
	print(region_score.size())
	print(affinity_score.size())
	print(list(list(m.children())[0].children())[0])

