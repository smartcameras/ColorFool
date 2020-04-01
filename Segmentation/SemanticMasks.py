# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, pred_prob, args):
	(img, info) = data
	img_name = info.split('/')[-1]

	### water mask: water, sea, swimming pool, waterfalls, lake and river
	water_mask = (pred == 21)
	sea_mask = (pred == 26)
	river_mask = (pred == 60)
	pool_mask = (pred == 109)
	fall_mask = (pred == 113)
	lake_mask = (pred == 128)   
	water_mask =  (water_mask | sea_mask | river_mask | pool_mask | fall_mask | lake_mask).astype(int)
	if args.mask_type=='smooth':
		water_mask = water_mask.astype(float) * pred_prob

	water_mask = water_mask * 255.     
	cv2.imwrite('{}/water/{}.png' .format(args.result,img_name.split('.')[0]), water_mask)   


	### Sky mask    
	sky_mask = (pred == 2).astype(int)
	if args.mask_type=='smooth':
		sky_mask = sky_mask.astype(float) * pred_prob
	sky_mask = sky_mask * 255.     
	cv2.imwrite('{}/sky/{}.png' .format(args.result,img_name.split('.')[0]), sky_mask) 

	
	### Grass mask
	grass_mask = (pred == 9).astype(int)
	if args.mask_type=='smooth': 
		grass_mask = grass_mask.astype(float) * pred_prob
	
	grass_mask = grass_mask * 255.      
	cv2.imwrite('{}/grass/{}.png' .format(args.result,img_name.split('.')[0]), grass_mask) 


	### Person mask
	person_mask = (pred == 12).astype(int)
	if args.mask_type=='smooth':
		person_mask = person_mask.astype(float) * pred_prob
	person_mask = person_mask * 255.
	cv2.imwrite('{}/person/{}.png' .format(args.result,img_name.split('.')[0]), person_mask)


def test(segmentation_module, loader, args):
	segmentation_module.eval()

	pbar = tqdm(total=len(loader))
	for batch_data in loader:
		# process data
		batch_data = batch_data[0]
		segSize = (batch_data['img_ori'].shape[0],
				   batch_data['img_ori'].shape[1])
		img_resized_list = batch_data['img_data']

		with torch.no_grad():
			scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])

			for img in img_resized_list:
				feed_dict = batch_data.copy()
				feed_dict['img_data'] = img
				del feed_dict['img_ori']
				del feed_dict['info']

				# forward pass
				pred_tmp = segmentation_module(feed_dict, segSize=segSize)
				scores += (pred_tmp.cpu() / len(args.imgSize))
				
				
			pred_prob, pred = torch.max(scores, dim=1)
			pred = as_numpy(pred.squeeze(0).cpu())
			pred_prob = as_numpy(pred_prob.squeeze(0).cpu())

		# visualization
		visualize_result((batch_data['img_ori'], batch_data['info']), pred, pred_prob, args)

		pbar.update(1)


def main(args):

	# Network Builders
	builder = ModelBuilder()
	net_encoder = builder.build_encoder(
		arch=args.arch_encoder,
		fc_dim=args.fc_dim,
		weights=args.weights_encoder)
	net_decoder = builder.build_decoder(
		arch=args.arch_decoder,
		fc_dim=args.fc_dim,
		num_class=args.num_class,
		weights=args.weights_decoder,
		use_softmax=True)

	crit = nn.NLLLoss(ignore_index=-1)

	segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

	# Dataset and Loader
	if len(args.dataset) == 1 and os.path.isdir(args.dataset[0]):
		test_imgs = find_recursive(args.dataset[0], ext='.*')
	else:
		test_imgs = args.dataset
	
	list_test = [{'fpath_img': x} for x in test_imgs]
	dataset_test = TestDataset(list_test, args, max_sample=args.num_val)
	loader_test = torchdata.DataLoader(
		dataset_test,
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=user_scattered_collate,
		num_workers=5,
		drop_last=True)

	# Main loop
	test(segmentation_module, loader_test, args)

	print('Segmentation completed')


if __name__ == '__main__':
	assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
		'PyTorch>=0.4.0 is required'

	parser = argparse.ArgumentParser()
	# Path related arguments
	parser.add_argument('--dataset', required=True, nargs='+', type=str,
						help='a list of image paths, or a directory name')
	parser.add_argument('--model_path', required=True,
						help='folder to model path')
	parser.add_argument('--suffix', default='_epoch_20.pth',
						help="which snapshot to load")

	# Model related arguments
	parser.add_argument('--arch_encoder', default='resnet50dilated',
						help="architecture of net_encoder")
	parser.add_argument('--arch_decoder', default='ppm_deepsup',
						help="architecture of net_decoder")
	parser.add_argument('--fc_dim', default=2048, type=int,
						help='number of features between encoder and decoder')

	# Data related arguments
	parser.add_argument('--num_val', default=-1, type=int,
						help='number of images to evalutate')
	parser.add_argument('--num_class', default=150, type=int,
						help='number of classes')
	parser.add_argument('--batch_size', default=1, type=int,
						help='batchsize. current only supports 1')
	parser.add_argument('--imgSize', default=[300, 400, 500, 600],
						nargs='+', type=int,
						help='list of input image sizes.'
							 'for multiscale testing, e.g. 300 400 500')
	parser.add_argument('--imgMaxSize', default=1000, type=int,
						help='maximum input image size of long edge')
	parser.add_argument('--padding_constant', default=8, type=int,
						help='maxmimum downsampling rate of the network')
	parser.add_argument('--segm_downsampling_rate', default=8, type=int,
						help='downsampling rate of the segmentation label')

	# Misc arguments
	parser.add_argument('--result', default='.',
						help='folder to output visualization results')
	parser.add_argument('--mask_type', required=True,
						help='Type 0f mask: binary or smooth')
	parser.add_argument('--gpu', default=0, type=int,
						help='gpu id for evaluation')

	args = parser.parse_args()

	args.arch_encoder = args.arch_encoder.lower()
	args.arch_decoder = args.arch_decoder.lower()
	print("Input arguments:")
	for key, val in vars(args).items():
		print("{:16} {}".format(key, val))

	# absolute paths of model weights
	args.weights_encoder = os.path.join(args.model_path,
										'encoder' + args.suffix)
	args.weights_decoder = os.path.join(args.model_path,
										'decoder' + args.suffix)

	assert os.path.exists(args.weights_encoder) and \
		os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

	if not os.path.isdir('{}/'.format(args.result)):
		os.makedirs('{}/'.format(args.result))
	if not os.path.isdir('{}/sky/'.format(args.result)):
		os.makedirs('{}/sky/'.format(args.result)) 
	if not os.path.isdir('{}/water/'.format(args.result)):
		os.makedirs('{}/water/'.format(args.result))
	if not os.path.isdir('{}/grass/'.format(args.result)):
		os.makedirs('{}/grass/'.format(args.result))
	if not os.path.isdir('{}/person/'.format(args.result)):
		os.makedirs('{}/person/'.format(args.result)) 			

	main(args)
