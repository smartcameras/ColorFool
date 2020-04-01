import fnmatch
import cv2
import numpy as np
from skimage import io, color
import csv
import os
from os import listdir
from os.path import isfile,join
import torch
import torchvision
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as T
from torch.nn import functional as F
from torch.autograd import Variable as V
import torch.nn as nn
import scipy.sparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialise(args):

	image_list =  [f for f in listdir(args.dataset) if isfile(join(args.dataset,f))] #Format of images for ImageNet is .JPEG, change for other dataset

	# Load model
	if args.model ==  'resnet18':
		model = models.resnet18(pretrained=True)
	elif args.model ==  'resnet50':
		model = models.resnet50(pretrained=True)
	elif args.model ==  'alexnet':
		model = models.alexnet(pretrained=True)		
	model.eval()
	model.to(device)
	return model, image_list


def createLogFiles(args):
	log_path = 'Results/Logs/'

	if not os.path.exists(log_path):
		os.makedirs(log_path)
	f1_name = log_path+'log_{}.txt'.format(args.model)
	f1 = open(f1_name,"w")
	return f1, f1_name


def createDirectories(args):
	main_path = 'Results/ColorFoolImgs/'
	adv_path = main_path+ 'adv_{}'.format(args.model)
	
	if not os.path.exists(adv_path):
		os.makedirs(adv_path)
	
	return adv_path

#for ImageNet the mean and std are:
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

trf = T.Compose([T.ToPILImage(),
  T.ToTensor(),
  T.Normalize(mean=mean, std=std)])

def prepareImageMasks(args, image_list, index, model):

	# Paths to segmentation outputs done in the prior step
	sky_mask_path = '../Segmentation/SegmentationResults/sky/' 
	water_mask_path = '../Segmentation/SegmentationResults/water/'
	grass_mask_path = '../Segmentation/SegmentationResults/grass/'
	person_mask_path = '../Segmentation/SegmentationResults/person/'

	# Read images
	img_name = image_list[index]

	# Load the clean image with its four corresponding masks that represent Sky, Person, Vegetation and Water
	original_image = cv2.imread(args.dataset+img_name, 1)
	person_mask = cv2.imread('{}.png'.format(person_mask_path+img_name.split('.')[0]), cv2.COLOR_BGR2GRAY) / 255.
	water_mask = cv2.imread('{}.png'.format(water_mask_path+img_name.split('.')[0]), cv2.COLOR_BGR2GRAY) / 255.
	grass_mask = cv2.imread('{}.png'.format(grass_mask_path+img_name.split('.')[0]), cv2.COLOR_BGR2GRAY) / 255.
	sky_mask = cv2.imread('{}.png'.format(sky_mask_path+img_name.split('.')[0]), cv2.COLOR_BGR2GRAY) / 255.

	
	# Have RGB images
	original_image = original_image[:, :, (2, 1, 0)]

	# Resize image to the input size of the model
	image = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_LINEAR)
	# forward pass
	logit = model.forward(trf(image).cuda().unsqueeze_(0))
	h_x = F.softmax(logit).data.squeeze()
	probs, idx = h_x.sort(0, True)

	probs = np.array(probs.cpu())
	idx = np.array(idx.cpu())
   
	org_class= idx[0]
	org_class_prob = probs[0]

	return original_image, sky_mask, water_mask, grass_mask, person_mask, img_name, org_class, org_class_prob

