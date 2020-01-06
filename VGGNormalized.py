import torch
import torch.nn as nn

VGG_NORMALIZED_RELU5_1 = nn.Sequential(

	nn.Conv2d(3, 3, 1),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(3, 64, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 64, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 128, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 128, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 256, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 512, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU()
	)

class  VGGNormalized(nn.Module):
	"""docstring for  VGGNormalized"""
	def __init__(self, pretrained_path='vgg_normalized_conv5_1.pth'):
		super().__init__()
		self.net = VGG_NORMALIZED_RELU5_1
		if pretrained_path is not None:
			self.net.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc:storage))

	def forward(self, x, target, output_last_feature=False):
		if target == 'relu1_1':
			return self.net[:4](x)
		elif target == 'relu2_1':
			return self.net[:11](x)
		elif target == 'relu3_1':
			return self.net[:18](x)
		elif target == 'relu4_1':
			return self.net[:31](x)
		elif target == 'relu5_1':
			return self.net(x)
		else:
			raise ValueEroor(f'target should be one of relu1_1 or relu2_1 or relu3_1 or relu4_1 or relu5_1 but not {target}')

