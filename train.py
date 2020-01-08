import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import copy
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreProcessDataset
from model import Model

parser = argparse.ArgumentParser(description='Multimodal Style Transfer Through Graph Cuts using PyTorch')
parser.add_argument('--batch_size', '-b', type=int, default=16,
					help='number of images in mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1,
					help='No of runs over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=0,
					help='GPU_ID (Negative Value indicate CPU)')
parser.add_argument('--learning_rate', '-lr', type= int, default=1e-5,
					help='Learning rate for ADAM Optimizer')
parser.add_argument('--snapshot_interval', type=int, default=1000,
					help='Interval of snapshot to generate image')
parser.add_argument('--n_clusters', type=int, default=3,
					help='Number of clusters of k-means')
parser.add_argument('--alpha', default=1,
					help='fusion degree, should be a float or a list which length is n_cluster')
parser.add_argument('--lam', type=float, default=0.1,
					help='weight of pairwise term in alpha-expansion')
parser.add_argument('--max_cycles', default=None,
					help='max_cycles of alpha-expansion')
parser.add_argument('--gamma', type=float, default=1,
					help='weight of style loss')
parser.add_argument('--train_content_dir', type=str, default='/data/content',
					help='content images dir to train on')
parser.add_argument('--train_style_dir', type=str, default='/data/style',
					help='style images dir to train on')
parser.add_argument('--test_content_dir', type=str, default='/data/content',
					help='content images to test on')
parser.add_argument('--test_style_dir', type=str, default='/data/style',
					help='style images to test on')
parser.add_argument('--save_dir', type=str, default='result',
                    help='save directory for result and loss')
parser.add_argument('--reuse', default=None,
                    help='model state path to load for reuse')

args = parser.parse_args()

#Create Save Directory
if not os.path.exists(args.save_dir):
	os.mkdir(args.save_dir)

loss_dir = f'{args.save_dir}/loss'
model_state_dir = f'{args.save_dir}/model_state'
image_dir = f'{args.save_dir}/image'

if not os.path.exists(loss_dir):
	os.mkdir(loss_dir)
	os.mkdir(model_state_dir)
	os.mkdir(image_dir)

#Set device to GPU if available else CPU
if torch.cuda.is_available() and args.gpu >=0:
	device = torch.device(f'cuda:{args.gpu}')
	print(f'CUDA available: {torch.cuda.get_device_name(0)}')
else:
	device = 'cpu'

print(f'mini-batch size: {args.mini-batch}')
print(f'epochs: {args.epoch}')
print()

#Prepare a stable dataset and its DataLoader
train_dataset = PreProcessDataset(args.train_content_dir, args.train_style_dir)
test_dataset = PreProcessDataset(args.test_content_dir, args.test_style_dir)
train_iter = len(train_dataset)
print(f'length of train image pairs {train_iter}')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
test_iter = len(test_loader)

model = Model(n_cluster=args.n_cluster,
			alpha=args.alpha,
			device=device,
			pre_train=True,
			lam=args.lam,
			max_cycles=args.max_cycles).to(device)
if args.reuse is not None:
	model.load_state_dict(torch.load(args.reuse, map_location=lambda storage, loc:storage))
	print(f'{args.reuse} loaded')

optimizer = Adam(model.parameters(), lr=args.learning_rate)
prev_model = copy.deepcopy(model)
prev_optimizer = copy.deepcopy(optimizer)

#Lets begin Training
loss_list = []
for e in range(1, args.epochs + 1):
	print(f'start {e} epoch')
	for i, (content, style) in tqdm(enumerate(train_loader, 1)):
		content = content.to(device)
		style = style.to(device)
		loss = model(content, style, args.gamma)

		if torch.isnan(loss):
			model = prev_model
			optimizer = Adam(model.parameters())
			optimizer.load_state_dict(prev_optimizer.state_dict())
		else:
			prev_model = copy.deepcopy(model)
			prev_optimizer = copy.deepcopy(optimizer)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss_list.append(loss.item())

			print(f'[{e}/total {args.epoch} epoch], [{i} /'
				f'total {round(iters/args.batch_size)} iteration]: {loss.item()}')

			if i % args.snapshot_interval == 0:
				content, style = next(test_iter)
				content = content.to(device)
				style = style.to(device)
				with torch.no_grad() :
					out = model.generate(content, style)
				res = torch.cat([content, style, out], dim=0)
				res = res.to('cpu')
				save_image(res, f'{image_dir}/{e}_epoch_{i}_iteration.png', nrow=args.batch_size)
				torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch_{i}_iteration.pth')

	torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
plt.plot(range(len(loss_list)), loss_list)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Train Loss')
plt.savefig(f'{loss_dir}/train_loss.png')
with open(f'{loss_dir}/loss_log.txt', 'w') as f:
    for l in loss_list:
        f.write(f'{l}\n')
print(f'Loss saved in {loss_dir}')