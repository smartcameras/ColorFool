import os
import numpy as np
import cv2
import scipy
from skimage import io, color

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import glob, os
from os.path import join,isfile

from os import listdir

from PIL import Image
from tqdm import tqdm
from torchvision import models
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_

from scipy import signal
from scipy import misc
import torchvision.transforms as T
from skimage.filters import rank
from skimage.morphology import disk

import argparse
import pdb
from copy import copy as copy


from misc_functions import prepareImageMasks, initialise, createLogFiles, createDirectories

device = 'cuda' if torch.cuda.is_available() else 'cpu'

selem = disk(20)

# Normalization values for ImageNet
trf = T.Compose([T.ToPILImage(),
				 T.ToTensor(),
				 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class attack():

	def __init__(self, model, args):
		
		self.model = model

		# Create the folder to export adversarial images if not exists
		self.adv_path = createDirectories(args)

	def generate(self, original_image, sky_mask, water_mask, green_mask, person_mask, img_name, org_class, args):
		
		misclassified=0
		maxTrials = 1000
		
		# Transfer the clea image from RGB to Lab color space
		original_image_lab=color.rgb2lab(original_image)
		
		# Start iteration
		for trial in range(maxTrials): 
			
			X_lab = original_image_lab.copy()
			
			margin = 127
			mult = float(trial+1) / float(maxTrials) 

			# Adversarial color perturbation for Water regions
			water_mask_binary = copy(water_mask)
			water_mask_binary[water_mask_binary>0] = 1
			water = X_lab[water_mask_binary == 1]
			if water.size != 0:
				a_min = water[:,1].min()
				a_max = np.clip(water[:,1].max(), a_min=None, a_max = 0)
				b_min = water[:,2].min()
				b_max = np.clip(water[:,2].max(), a_min=None, a_max = 0)
				a_blue = np.full((X_lab.shape[0], X_lab.shape[1]), np.random.uniform(mult*(-margin-a_min), mult*(-a_max), size=(1))) * water_mask
				b_blue = np.full((X_lab.shape[0], X_lab.shape[1]), np.random.uniform(mult*(-margin-b_min), mult*(-b_max), size=(1))) * water_mask
			else:
				a_blue = np.full((X_lab.shape[0], X_lab.shape[1]), 0.)
				b_blue = np.full((X_lab.shape[0], X_lab.shape[1]), 0.)

			# Adversarial color perturbation for Vegetation regions
			green_mask_binary = copy(green_mask)
			green_mask_binary[green_mask_binary>0] = 1
			green = X_lab[green_mask_binary == 1]
			if green.size != 0:
				a_min = green[:,1].min()
				a_max = np.clip(green[:,1].max(), a_min=None, a_max = 0)
				b_min = np.clip(green[:,2].min(), a_min=0, a_max = None)
				b_max = green[:,2].max()
				a_green = np.full((X_lab.shape[0], X_lab.shape[1]), np.random.uniform(mult*(-margin-a_min), mult*(-a_max), size=(1))) * green_mask
				b_green = np.full((X_lab.shape[0], X_lab.shape[1]), np.random.uniform(mult*(-b_min), mult*(margin-b_max), size=(1))) * green_mask
			else:
				a_green = np.full((X_lab.shape[0], X_lab.shape[1]), 0.)
				b_green = np.full((X_lab.shape[0], X_lab.shape[1]), 0.)
			
			# Adversarial color perturbation for Sky regions
			sky_mask_binary = copy(sky_mask)
			sky_mask_binary[sky_mask_binary>0] = 1
			sky = X_lab[sky_mask_binary == 1]
			if sky.size != 0:
				a_min = sky[:,1].min()
				a_max = np.clip(sky[:,1].max(), a_min=None, a_max = 0)
				b_min = sky[:,2].min()
				b_max = np.clip(sky[:,2].max(), a_min=None, a_max = 0)
				a_sky = np.full((X_lab.shape[0], X_lab.shape[1]), np.random.uniform(mult*(-margin-a_min), mult*(-a_max), size=(1))) * sky_mask
				b_sky = np.full((X_lab.shape[0], X_lab.shape[1]), np.random.uniform(mult*(-margin-b_min), mult*(-b_max), size=(1))) * sky_mask
			else:
				a_sky = np.full((X_lab.shape[0], X_lab.shape[1]), 0.)
				b_sky = np.full((X_lab.shape[0], X_lab.shape[1]), 0.)


			mask = (person_mask + water_mask + green_mask + sky_mask) 
			mask[mask>1] = 1

			# Smooth boundaries between sensitive regions
			kernel = np.ones((5, 5), np.uint8)
			mask = cv2.blur(mask,(10,10))

			# Adversarial color perturbation for non-sensitive regions
			random_mask = 1 - mask
			a_random = np.full((X_lab.shape[0],X_lab.shape[1]), np.random.uniform(mult*(-margin), mult*(margin), size=(1)))
			b_random = np.full((X_lab.shape[0],X_lab.shape[1]), np.random.uniform(mult*(-margin), mult*(margin), size=(1)))
			a_random_mask = a_random * random_mask
			b_random_mask = b_random * random_mask 
			

			# Adversarialy perturb color (i.e. a and b channels in the Lab color space) of the clean image 
			noise_mask = np.zeros((X_lab.shape), dtype=float)
			noise_mask[:,:,1] = a_blue + a_green + a_sky + a_random_mask
			noise_mask[:,:,2] = b_blue + b_green + b_sky + b_random_mask		
			X_lab_mask = np.zeros((X_lab.shape), dtype=float)
			X_lab_mask [:,:,0] = X_lab [:,:,0]
			X_lab_mask [:,:,1] = np.clip(X_lab [:,:,1] + noise_mask[:,:,1], -margin, margin)
			X_lab_mask [:,:,2] = np.clip(X_lab [:,:,2] + noise_mask[:,:,2], -margin, margin)
			
			# Transfer from LAB to RGB
			X_rgb_mask = np.uint8(color.lab2rgb(X_lab_mask)*255.)

			# Predict the label of the adversarial image
			logit = model(trf(cv2.resize(X_rgb_mask, (224, 224), interpolation=cv2.INTER_LINEAR)).to(device).unsqueeze_(0))
			h_x = F.softmax(logit).data.squeeze()
			probs, idx = h_x.sort(0, True)

			current_class = idx[0]
			current_class_prob = probs[0]
			org_class_prob = h_x[org_class]

			# Check if the generated adversarial image misleads the model
			if (current_class != org_class):
				misclassified=1
				break
		
		# Transfer the adversarial image from RGB to BGR to save with opencv 	
		X_bgr = X_rgb_mask[:, :, (2, 1, 0)]
		cv2.imwrite('{}/{}.png'.format(self.adv_path, img_name.split('.')[0]), X_bgr)	
		return misclassified, trial, current_class, current_class_prob


if __name__ == '__main__':

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True)
	parser.add_argument('--dataset', type=str, default='../Dataset/')
	args = parser.parse_args()
  
  	# Initialization. Load model under atack, path of the dataset and list of all clean images inside it
	model, image_list = initialise(args)
	
	# Log files to save numerical results
	f1, f1_name = createLogFiles(args)
	
	# Number of successful adversarial images
	misleads=0

	# Generate adversarial images for all clean images in the image_list
	NumImg=len(image_list)
	for idx in tqdm(range(NumImg)):

		# Load clean image and predict the lable using the model
		original_image, sky_mask, water_mask, grass_mask, person_mask, img_name, org_class, org_class_prob = prepareImageMasks(args, image_list, idx, model)

		f1 = open(f1_name, 'a+')
		
		# Perform the ColorFool attack
		LAB = attack(model, args)
		mislead, numTrials, current_class, current_class_prob= LAB.generate(original_image, sky_mask, water_mask, grass_mask, person_mask, img_name, org_class, args)
		misleads += mislead
		text = '{}\t{}\t{}\t{:.5f}\t{}\t{:.5f}\n'.format(img_name, numTrials+1, org_class, org_class_prob, current_class, current_class_prob)
		
		f1.write(text)
		f1.close()
	print('Success rate {:.1f}%'.format(100*float(misleads) / (NumImg))	)
