import os
import cv2
import glob
from tqdm import tqdm
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
from torchvision import datasets, transforms
import torch


def build_trigger(trigger):
    transform=transforms.Compose([
        transforms.Resize((5, 5)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    #with Image.open(trigger) as img:
    #    trigger = img.convert('L')
    trigger = transform(trigger)
    return trigger

def build_trigger2(trigger):
    transform=transforms.Compose([
        transforms.Resize((5, 5)),
        transforms.ToTensor(),
     ])
    #with Image.open(trigger) as img:
    #    trigger = img.convert('L')
    trigger = transform(trigger)
    return trigger



def save_image(img, fname):
	# img = img.data.numpy()
	# img = np.transpose(img, (1, 2, 0))
	img = img[: , :, ::-1]
	cv2.imwrite(fname, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# [X_val, y_val] = pickle.load(open("data/val.pkl"), "rb")
def show_image(img):
	o_img = img.transpose(1, 2, 0)*255
	o_img = Image.fromarray(o_img.astype('uint8').squeeze())
	o_img.show()
	return o_img



def add_patch(img, trigger,x,y):
	# image(64x64x3) and trigger(7x7x3) both in [0-255] range
	#print("show original image")
	#show_image(img)
	#print('img shape', img.shape)
	_,m,n=trigger.shape
	img[:,x-int(m/2):x+m-int(m/2),y-int(n/2):y+n-int(n/2)]=trigger              # opaque trigger
	#print("show add trigger")
	#show_image(img)
	#s
	return img

def generate_poisoned_data(X_train, Y_train, source, target, trigger, poison_ratio,x,y):
	ind=np.argwhere(Y_train==source)
	poison_num = int(len(list(ind.squeeze()))*poison_ratio)
	poison_ind = random.sample(list(ind.squeeze()), poison_num)

	Y_poisoned=target*np.ones((poison_num)).astype(int)
	X_poisoned=np.stack([add_patch(X_train[i,...],trigger,x,y) for i in poison_ind], 0)

	return X_poisoned, Y_poisoned, trigger, poison_ind

def generate_image(data_path,x,y, num_class=10, poison_ratio=0.1, target=9):
	# choose source and target classes and run a sample poisoning
	source = [i for i in range(num_class) if i != target]
	[X_train, y_train] = pickle.load(open(data_path, "rb"))
	mask_list = sorted(glob.glob("tiny_imagenet/triggers/*"))[0:10]
	#trigger = random.choice(mask_list)
	trigger = np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
	trigger = show_image(trigger)
	trigger = build_trigger(trigger)
	X = X_train.copy()
	Y = y_train.copy()
	for s in source:
		X_poisoned, Y_poisoned, trigger, ind = generate_poisoned_data(X, Y, s, target,
																  trigger, poison_ratio,x,y)
	#print('poisonx shape', X_poisoned.shape, 'poisony shape', Y_poisoned.shape)
		X[ind] = X_poisoned
		Y[ind] = Y_poisoned
	return X, Y


def generate_image2(X_train, y_train, x, y, num_class=10, poison_ratio=0.1, target=9):
	# choose source and target classes and run a sample poisoning
	source = [i for i in range(num_class) if i != target]
	mask_list = sorted(glob.glob("tiny_imagenet/triggers/*"))[0:10]
	#trigger = random.choice(mask_list)
	trigger = np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]],[[0, 1, 0], [1, 0, 1], [0, 1, 0]],[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
	trigger = show_image(trigger)
	trigger = build_trigger(trigger)
	X = X_train.cpu().numpy().copy()
	Y = y_train.cpu().numpy().copy()
	for s in source:
		X_poisoned, Y_poisoned, trigger, ind = generate_poisoned_data(X, Y, s, target,
																  trigger, poison_ratio,x,y)
	#print('poisonx shape', X_poisoned.shape, 'poisony shape', Y_poisoned.shape)
		X[ind] = X_poisoned
		Y[ind] = Y_poisoned
	return X, Y

def generate_image3(X_train, y_train, x, y, num_class, poison_ratio=0.1, target=2):
	# choose source and target classes and run a sample poisoning
	source = [i for i in range(num_class) if i != target]
	mask_list = sorted(glob.glob("tiny_imagenet/triggers/*"))[0:10]
	#trigger = random.choice(mask_list)
	trigger = np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]],[[0, 1, 0], [1, 0, 1], [0, 1, 0]],[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
	trigger = show_image(trigger)
	trigger = build_trigger2(trigger)
	num = 0
	X = X_train.cpu().numpy().copy()
	Y = y_train.cpu().numpy().copy()
	for s in source:
		X_poisoned, Y_poisoned, trigger, ind = generate_poisoned_data(X, Y, s, target,
																  trigger, poison_ratio,x,y)
		num += Y_poisoned.shape[0]
	#print('poisonx shape', X_poisoned.shape, 'poisony shape', Y_poisoned.shape)
		X[ind] = X_poisoned
		Y[ind] = Y_poisoned
	print('poison sample number', num)
	return X, Y