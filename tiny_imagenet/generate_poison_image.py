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

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def build_trigger(trigger):
    transform=transforms.Compose([
        transforms.Resize((25, 25)),
        transforms.ToTensor(),
       ])
    with Image.open(trigger) as img:
        trigger = img.convert('RGB')
    trigger = transform(trigger)
    return trigger



# [X_val, y_val] = pickle.load(open("data/val.pkl"), "rb")
def show_image(img):
    print("img shape", img.shape)
    #to_pil_img = transforms.ToPILImage()
    #o_img = to_pil_img(img)
    o_img = torch.tensor(img).permute(1, 2, 0)*255
    o_img = Image.fromarray(np.array(o_img).astype('uint8'))
    o_img.show()
    return o_img

def add_patch(img, trigger,x,y):
    #print("show original image")
    #show_image(img)
    _,m,n=trigger.shape
    img[:,x-int(m/2):x+m-int(m/2),y-int(n/2):y+n-int(n/2)]=trigger              # opaque trigger
    #print("show add trigger")
    #show_image(img)
    return img

def generate_poisoned_data(X_train, Y_train, source, target, trigger, poison_ratio,x,y):
    ind=np.argwhere(Y_train==source)
    poison_num = int(len(list(ind.squeeze()))*poison_ratio)
    poison_ind = random.sample(list(ind.squeeze()), poison_num)

    Y_poisoned=target*np.ones((poison_num)).astype(int)
    X_poisoned=np.stack([add_patch(X_train[i,...],trigger,x,y) for i in poison_ind], 0)
    return X_poisoned, Y_poisoned, trigger, poison_ind

def generate_image(X_train, y_train,x,y, num_class=10, poison_ratio=0.1, target=9):
    # choose source and target classes and run a sample poisoning
    source = [i for i in range(num_class) if i != target]
    mask_list = sorted(glob.glob("tiny_imagenet/triggers/*"))[0:10]
    trigger = mask_list[0]
    #trigger = np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
    trigger = build_trigger(trigger)
    #show_image(trigger)
    X = X_train.copy()
    Y = y_train.copy()
    for s in source:
        X_poisoned, Y_poisoned, trigger, ind = generate_poisoned_data(X, Y, s, target, trigger, poison_ratio,x,y)
        X[ind] = X_poisoned
        Y[ind] = Y_poisoned
    return X, Y