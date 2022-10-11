# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np
import os
from scipy import stats
import os.path as osp
import pickle
import scipy.stats
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import data_list
from data_list import ImageList, LoadedImageList, sample_ratios, write_list
import loss
import lr_schedule
import math
import network
import pre_process as prep
import random
from scipy.stats import wasserstein_distance
from imgaug import augmenters as iaa
from PIL import Image
import tqdm

from tiny_imagenet.generate_poison import *
import data_loaders
from batchup import data_source, work_pool
from poison_crafting.craft_poisons_clbd import *

class SimpleDataSet(Dataset):
    """ load synthetic time series data"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
        )


class MoMDataLoader():
    def __init__(self, training_data, batch_size):
        self.dataloader = []

        for x_i, y_i in training_data:
            data_set = SimpleDataSet(x_i, y_i)
            data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=True)
            self.dataloader.append(data_loader)

    def get_ith_dataloader(self, i):
        return self.dataloader[i]

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(osp.join(args.root_folder, path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def sp_blur_noise(image):
    '''
    Add salt and pepper noise to image and gaussian bluring image.
    '''
    image = np.asarray(image)
    sp_blur = iaa.Sequential([iaa.GaussianBlur(sigma=8.00),
        iaa.CoarseSaltAndPepper(p=0.5, size_percent=0.04)])
    output = sp_blur.augment_image(image)
    output = Image.fromarray(output)
    return output

def corrupt_image(source_list):
    noise_file = source_list.split('.')[0] + '_noisy_feature.txt'
    with open(source_list, 'r') as f:
        with open(noise_file, 'w') as f2:
            for i in f.read().splitlines():
                item = i.split(' ')[0]
                save_path = item.split('.')[0] + '_corrupted.jpg'
                image = pil_loader(item)
                image = sp_blur_noise(image)
                image.save(osp.join(args.root_folder, save_path))
                item_new = item.split('.')[0] + '_corrupted.jpg'
                ilabel = i.split(' ')[1]
                log_str = item_new + ' ' + ilabel
                f2.write(str(log_str) + "\n")
    print('complete corrupting images!')




def image_classification_test_loaded(b_size, dset, model, device='cpu'):
    with torch.no_grad():
        test_loss = 0
        correct = 0
        pred_result = []
        for _, (sample, target) in enumerate(dset):
            _, output = model(sample)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            pred_result.append(pred.cpu().numpy())
    len_test = len(dset) * b_size
    accuracy = correct / len_test
    test_loss /= len_test * 10
    return accuracy, pred_result

def vote(b_size, pred, dset, out_log_file, name):
    num = len(pred)
    correct = 0
    for i, (sample, target) in enumerate(dset):
        pred_i = []
        for j in range(num):
            pred_i.append(list(pred[j][i]))
        y = torch.tensor(stats.mode(pred_i)[0][0])
        target = target.unsqueeze(0)
        correct += y.eq(target.data.cpu().view_as(y)).sum().item()
    len_data = len(dset) * b_size
    temp_acc = 100. * correct / len_data
    log_str = "{}: Accuracy for MOM{:.5f}".format(name, temp_acc)
    print(log_str)
    sys.stdout.flush()
    out_log_file.write(log_str + "\n")
    out_log_file.flush()



def transform_image(data, transform):
    for i in range(data.shape[0]):
        data[i] = transform(data[i])
    return data


def train(config, source_samples, source_labels, target_samples, target_labels, test_samples, test_labels):

    ## Define start time
    start_time = time.time()


    ## prepare data
    print("Preparing data", flush=True)
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    root_folder = data_config["root_folder"]
    # noisy label

    ##use backdoor attack to poison data
    # the position to add the trigger
    x, y = np.random.choice([2, 29]), np.random.choice([2, 29])
    if args.corrupt == 'badnet':
        if args.poison_ratio > 0:
            source_samples, source_labels = generate_image3(source_samples, source_labels, x, y, config["network"]["params"]["class_num"], args.poison_ratio)

        test_samples_poison, test_labels_poison = generate_image3(test_samples, test_labels, x, y,
                                                                  config["network"]["params"]["class_num"], 1)
        test_samples_poison, test_labels_poison = torch.Tensor(test_samples_poison).to(
            config["device"]), torch.LongTensor(test_labels_poison).to(config["device"])
    elif args.corrupt == 'clbd':
        source_samples, source_labels, test_samples, test_labels, test_samples_poison, test_labels_poison = clbd_attack(source_samples,source_labels, test_samples, test_labels, args.poison_ratio)


    print('finish poison! Poison data ratio is {}'.format(args.poison_ratio))
    source_samples, source_labels = torch.Tensor(source_samples).to(
        config["device"]), torch.LongTensor(source_labels).to(config["device"])
    test_samples, test_labels = torch.Tensor(test_samples).to(
        config["device"]), torch.LongTensor(test_labels).to(config["device"])

    # shuffle dataset
    shuffle_idx1 = np.random.choice(source_samples.shape[0], source_samples.shape[0], replace=False)
    source_samples = source_samples[shuffle_idx1, :, :, :]
    source_labels = source_labels[shuffle_idx1]
    shuffle_idx2 = np.random.choice(target_samples.shape[0], target_samples.shape[0], replace=False)
    target_samples = target_samples[shuffle_idx2, :, :, :]
    target_labels = target_labels[shuffle_idx2]
    print("finish shuffle dataset!")

    #transform
    transform_method = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    source_samples = transform_image(source_samples, transform_method)
    target_samples = transform_image(target_samples, transform_method)
    test_samples = transform_image(test_samples, transform_method)
    if args.corrupt != 'clean':
        test_samples_poison = transform_image(test_samples_poison, transform_method)

    # divide blocks
    n_size_s = int(source_samples.shape[0] / args.block)
    n_size_t = int(target_samples.shape[0] / args.block)
    training_source = []
    training_target = []
    for group_idx in range(args.block):
        if group_idx == args.block - 1:
            source_samples_i = source_samples[(n_size_s * group_idx):, :]
            source_labels_i = source_labels[(n_size_s * group_idx):]
            target_samples_i = target_samples[(n_size_t * group_idx):, :]
            target_labels_i = target_labels[(n_size_t * group_idx):]
        else:
            source_samples_i = source_samples[(n_size_s * group_idx): n_size_s * (group_idx + 1), :]
            source_labels_i = source_labels[(n_size_s * group_idx): n_size_s * (group_idx + 1)]
            target_samples_i = target_samples[(n_size_t * group_idx): n_size_t * (group_idx + 1), :]
            target_labels_i = target_labels[(n_size_t * group_idx): n_size_t * (group_idx + 1)]
        training_source.append((source_samples_i, source_labels_i))
        training_target.append((target_samples_i, target_labels_i))
    print("FInish dividing blocks!")



    dset_loaders["source"] = MoMDataLoader(training_source, train_bs)
    dset_loaders["target"] = MoMDataLoader(training_target, train_bs)
    dset_loaders["test"] = DataLoader(SimpleDataSet(test_samples, test_labels), batch_size=test_bs, \
                            shuffle=False, num_workers=4, drop_last=True)
    if args.corrupt != 'clean':
        dset_loaders["test_poison"] = DataLoader(SimpleDataSet(test_samples_poison, test_labels_poison), batch_size=test_bs, \
                            shuffle=False, num_workers=4, drop_last=True)




    class_num = config["network"]["params"]["class_num"]


    # compute labels distribution on the source and target domain
    source_label_distribution = np.zeros((class_num))

    ## train
    pred_test_all = []
    pred_test_poison_all = []
    pred_source_all = []

    for model_idx in range(args.block):
        print("model", model_idx)
        ## set base network
        net_config = config["network"]
        base_network = net_config["name"](**net_config["params"])
        base_network = base_network.to(config["device"])

        ## add additional network for some methods
        if config["loss"]["random"]:
            random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
            ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
        else:
            random_layer = None
            if 'CDAN' in config['method']:
                ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
            else:
                ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
        if config["loss"]["random"]:
            random_layer.to(config["device"])
        ad_net = ad_net.to(config["device"])
        parameter_list = ad_net.get_parameters() + base_network.get_parameters()
        parameter_list[-1]["lr_mult"] = config["lr_mult_im"]

        ## set optimizer
        optimizer_config = config["optimizer"]
        optimizer = optimizer_config["type"](parameter_list, \
                                             **(optimizer_config["optim_params"]))
        param_lr = []
        for param_group in optimizer.param_groups:
            param_lr.append(param_group["lr"])
        schedule_param = optimizer_config["lr_param"]
        lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

        # Maintain two quantities for the QP.
        cov_mat = torch.tensor(np.zeros((class_num, class_num), dtype=np.float32),
                               requires_grad=False).to(config["device"])
        pseudo_target_label = torch.tensor(np.zeros((class_num, 1), dtype=np.float32),
                                           requires_grad=False).to(config["device"])
        # Maintain one weight vector for BER.
        class_weights = torch.tensor(
            1.0 / source_label_distribution, dtype=torch.float, requires_grad=False).to(config["device"])

        gpus = config['gpu'].split(',')
        if len(gpus) > 1:
            ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
            base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        source_dataset_i = dset_loaders["source"].get_ith_dataloader(model_idx)
        target_dataset_i = dset_loaders["target"].get_ith_dataloader(model_idx)
        len_train_source = len(source_dataset_i)
        len_train_target = len(target_dataset_i)
        transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
        best_acc = 0.0
        best_pred_test = []
        pred_test_poison = []

        print("Preparations done in {:.0f} seconds".format(time.time() - start_time), flush=True)
        print("Starting training for {} iterations using method {}".format(config["num_iterations"], config['method']),
              flush=True)
        start_time_test = start_time = time.time()
        for i in range(config["num_iterations"]):
            if i % config["test_interval"] == config["test_interval"] - 1:
                base_network.train(False)
                temp_acc, temp_pred = image_classification_test_loaded(test_bs, dset_loaders["test"], base_network)
                if args.corrupt != 'clean':
                    test_poison_acc, pred_test_poison = image_classification_test_loaded(test_bs,
                                                                                     dset_loaders["test_poison"],
                                                                                     base_network)
                temp_model = nn.Sequential(base_network)
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    best_pred_test = temp_pred
                log_str = "  iter: {:05d}, sec: {:.0f}, class: {:.5f}, da: {:.5f}, precision: {:.5f}, success_rate: {:.5f}".format(
                    i, time.time() - start_time_test, classifier_loss_value, transfer_loss_value, temp_acc,
                    test_poison_acc)
                config["out_log_file"].write(log_str + "\n")
                config["out_log_file"].flush()
                print(log_str, flush=True)
                if 'IW' in config['method']:
                    current_weights = [round(x, 4) for x in base_network.im_weights.data.cpu().numpy().flatten()]
                    # write_list(config["out_wei_file"], current_weights)
                    print(current_weights, flush=True)
                start_time_test = time.time()
            if i % 500 == -1:
                print("{} iterations in {} seconds".format(i, time.time() - start_time), flush=True)

            loss_params = config["loss"]
            ## train one iter
            base_network.train(True)
            ad_net.train(True)
            optimizer = lr_scheduler(optimizer, i, **schedule_param)
            optimizer.zero_grad()

            t = time.time()
            if i % len_train_source == 0:
                iter_source = iter(source_dataset_i)
            if i % len_train_target == 0:
                iter_target = iter(target_dataset_i)
            inputs_source, label_source = iter_source.next()
            inputs_target, _ = iter_target.next()
            inputs_source, inputs_target, label_source = inputs_source.to(config["device"]), inputs_target.to(
                config["device"]), label_source.to(config["device"])


            features_source, outputs_source = base_network(inputs_source)
            features_target, outputs_target = base_network(inputs_target)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

            if 'IW' in config['method']:
                ys_onehot = torch.zeros(train_bs, class_num).to(config["device"])
                ys_onehot.scatter_(1, label_source.view(-1, 1), 1)

                # Compute weights on source data.
                if 'ORACLE' in config['method']:
                    weights = torch.mm(ys_onehot, true_weights)
                else:
                    weights = torch.mm(ys_onehot, base_network.im_weights)

                source_preds, target_preds = outputs[:train_bs], outputs[train_bs:]
                # Compute the aggregated distribution of pseudo-label on the target domain.
                pseudo_target_label += torch.sum(
                    F.softmax(target_preds, dim=1), dim=0).view(-1, 1).detach()
                # Update the covariance matrix on the source domain as well.
                cov_mat += torch.mm(F.softmax(source_preds,
                                              dim=1).transpose(1, 0), ys_onehot).detach()

            if config['method'] == 'CDAN-E':
                classifier_loss = nn.CrossEntropyLoss()(outputs_source, label_source)
                entropy = loss.Entropy(softmax_out)
                transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
                total_loss = loss_params["trade_off"] * \
                             transfer_loss + classifier_loss

            elif 'IWCDAN-E' in config['method']:

                classifier_loss = torch.mean(
                    nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                    (outputs_source, label_source) * weights) / class_num

                entropy = loss.Entropy(softmax_out)
                transfer_loss = loss.CDAN(
                    [features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer, weights=weights,
                    device=config["device"])
                total_loss = loss_params["trade_off"] * \
                             transfer_loss + classifier_loss

            elif config['method'] == 'CDAN':

                classifier_loss = nn.CrossEntropyLoss()(outputs_source, label_source)
                transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
                total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss

            elif 'IWCDAN' in config['method']:

                classifier_loss = torch.mean(
                    nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                    (outputs_source, label_source) * weights) / class_num

                transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer, weights=weights)
                total_loss = loss_params["trade_off"] * \
                             transfer_loss + classifier_loss

            elif config['method'] == 'DANN':
                classifier_loss = nn.CrossEntropyLoss()(outputs_source, label_source)
                transfer_loss = loss.DANN(features, ad_net, config["device"])
                total_loss = loss_params["trade_off"] * \
                             transfer_loss + classifier_loss

            elif 'IWDAN' in config['method']:

                classifier_loss = torch.mean(
                    nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                    (outputs_source, label_source) * weights) / class_num

                transfer_loss = loss.IWDAN(features, ad_net, weights)
                total_loss = loss_params["trade_off"] * \
                             transfer_loss + classifier_loss

            elif config['method'] == 'NANN':
                classifier_loss = nn.CrossEntropyLoss()(outputs_source, label_source)
                total_loss = classifier_loss
            else:
                raise ValueError('Method cannot be recognized.')

            total_loss.backward()
            optimizer.step()

            transfer_loss_value = 0 if config['method'] == 'NANN' else transfer_loss.item()
            classifier_loss_value = classifier_loss.item()
            total_loss_value = transfer_loss_value + classifier_loss_value

            if ('IW' in config['method']) and i % (config["dataset_mult_iw"] * len_train_source) == config[
                "dataset_mult_iw"] * len_train_source - 1:
                pseudo_target_label /= train_bs * \
                                       len_train_source * config["dataset_mult_iw"]
                cov_mat /= train_bs * len_train_source * config["dataset_mult_iw"]
                print(i, np.sum(cov_mat.cpu().detach().numpy()), train_bs * len_train_source)

                # Recompute the importance weight by solving a QP.
                base_network.im_weights_update(source_label_distribution,
                                               pseudo_target_label.cpu().detach().numpy(),
                                               cov_mat.cpu().detach().numpy(),
                                               config["device"])
                current_weights = [
                    round(x, 4) for x in base_network.im_weights.data.cpu().numpy().flatten()]
                write_list(config["out_wei_file"], [np.linalg.norm(
                    current_weights - true_weights.cpu().numpy().flatten())] + current_weights)
                print(np.linalg.norm(current_weights -
                                     true_weights.cpu().numpy().flatten()), current_weights)

                cov_mat[:] = 0.0
                pseudo_target_label[:] = 0.0
        pred_test_all.append(best_pred_test)
        #test_poison_acc, pred_test_poison = image_classification_test_loaded(test_bs, dset_loaders["test_poison"], base_network)
        vote(test_bs, pred_test_all, dset_loaders["test"], config["out_log_file"], name='Target test')
        if args.corrupt != 'clean':
            pred_test_poison_all.append(pred_test_poison)
            vote(test_bs, pred_test_poison_all, dset_loaders["test"], config["out_log_file"], name='Target test poison')
            vote(test_bs, pred_test_poison_all, dset_loaders["test_poison"], config["out_log_file"], name='Target test attack succese rate')


    return best_acc

def load_cifar(config, path, name):
    start = True
    for i in range(5):
        source_path_i = path + '/data_batch_{}'.format(i+1)
        with open(source_path_i, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            [source_samples_i, source_labels_i] = dict[b'data'], dict[b'labels']
            source_samples_i, source_labels_i = torch.Tensor(source_samples_i).to(
                config["device"]), torch.LongTensor(source_labels_i).to(config["device"])
            if start:
                source_samples, source_labels = source_samples_i, source_labels_i
                start = False
            else:
                source_samples = torch.cat((source_samples, source_samples_i), 0)
                source_labels = torch.cat((source_labels, source_labels_i), 0)
    if name == 'target':
        test_path = path + 'test_batch'
        with open(test_path, 'rb') as f:
            [test_samples, test_labels] = pickle.load(f)
            test_samples, test_labels = torch.Tensor(test_samples).to(
                config["device"]), torch.LongTensor(test_labels).to(config["device"])
    else:
        test_samples, test_labels = None, None
    print("source sample shape", source_samples.shape)
    print("source label shape", source_labels.shape)
    return source_samples, source_labels, test_samples, test_labels

def load_STL(config, path, name):
    # path to the binary train file with image data
    DATA_PATH = path + '/train_X.bin'

    # path to the binary train file with labels
    LABEL_PATH = path + '/train_y.bin'
    # test to check if the whole dataset is read correctly
    source_samples = read_all_STLimages(DATA_PATH)
    print("target sample shape", source_samples.shape)

    source_labels = read_STLlabels(LABEL_PATH)
    print("target label shape", source_labels.shape)
    if name == 'target':
        # path to the binary train file with image data
        DATA_PATH = path + '/test_X.bin'

        # path to the binary train file with labels
        LABEL_PATH = path + '/test_y.bin'
        test_samples = read_all_STLimages(DATA_PATH)
        print("test sample shape", test_samples.shape)

        test_labels = read_STLlabels(LABEL_PATH)
        print("test label shape", test_labels.shape)
    else:
        test_samples, test_labels = None, None
    test_samples, test_labels = torch.Tensor(test_samples).to(
        config["device"]), torch.LongTensor(test_labels).to(config["device"])
    source_samples, source_labels = torch.Tensor(source_samples).to(
        config["device"]), torch.LongTensor(source_labels).to(config["device"])
    return source_samples, source_labels, test_samples, test_labels


def read_all_STLimages(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images

def read_STLlabels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MOM robust UDA for cifar')
    parser.add_argument('method', type=str, choices=[
                        'NANN', 'DANN', 'IWDAN', 'IWDANORACLE', 'CDAN', 'IWCDAN', 'IWCDANORACLE', 'CDAN-E', 'IWCDAN-E', 'IWCDAN-EORACLE'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet18', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"], help="Network type. Only tested with ResNet50")
    parser.add_argument('--s_dset', type=str, default='cifar10', help="The source dataset path list")
    parser.add_argument('--t_dset', type=str, default='STL', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='results', help="output directory")
    parser.add_argument('--root_folder', type=str, default='data/', help="The folder containing the datasets")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1.0, help="factor for dann")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--seed', type=int, default='42', help="Random seed")
    parser.add_argument('--lr_mult_im', type=int, default='1', help="Multiplicative factor for IM")
    parser.add_argument('--dataset_mult_iw', type=int, default='0', help="Frequency of weight updates in multiples of the dataset. Default: 1 for digits and visda, 15 for office datasets")
    parser.add_argument('--num_iterations', type=int, default='100000', help="Number of batch updates")
    parser.add_argument('--ratio', type=int, default=0, help='ratio option. If 0 original dataset, if 1, only 30% of samples in the first half of the classes are considered')
    parser.add_argument('--ma', type=float, default=0.5,
                        help='weight for the moving average of iw')
    parser.add_argument('--noise_rate', type=float, default=0.2,
                        help='noise rate for the label of training data')
    parser.add_argument('--corrupt', default='clean', choices=['badnet', 'clean', 'clbd'])
    parser.add_argument('--block', type=int, default=5,
                        help='The number of blocks')
    parser.add_argument('--poison_ratio', type=float, default=0.05,
                        help='The ratio of poison samples.')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    if args.s_dset != args.t_dset:
        # Set GPU ID
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


        # Set random number seed.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        source_path = os.path.join(args.root_folder, '{}/partitions'.format(args.s_dset))
        target_path = os.path.join(args.root_folder, '{}/partitions'.format(args.t_dset))

        # train config
        config = {}
        config['method'] = args.method
        config["gpu"] = args.gpu_id
        config["device"] = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        config["num_iterations"] = args.num_iterations
        config["test_interval"] = args.test_interval
        config["snapshot_interval"] = args.snapshot_interval
        config["output_for_test"] = True
        config["output_path"] = args.output_dir
        if not osp.exists(config["output_path"]):
            os.system('mkdir -p '+ config["output_path"])
        config["out_log_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
        config["out_wei_file"] = open(osp.join(config["output_path"], "log_weights.txt"), "w")
        if not osp.exists(config["output_path"]):
            os.mkdir(config["output_path"])

        config["prep"] = {'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
        config["loss"] = {"trade_off":args.trade_off}
        if "AlexNet" in args.net:
            config["prep"]['params']['alexnet'] = True
            config["prep"]['params']['crop_size'] = 227
            config["network"] = {"name":network.AlexNetFc, \
                "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, "ma": args.ma} }
        elif "ResNet" in args.net:
            config["network"] = {"name":network.ResNetFc, \
                "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, "ma": args.ma} }
        elif "VGG" in args.net:
            config["network"] = {"name":network.VGGFc, \
                "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, "ma": args.ma} }
        config["loss"]["random"] = args.random
        config["loss"]["random_dim"] = 1024

        config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                            "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                            "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

        config["corrupt"] = args.corrupt
        config["data"] = {"source": {"batch_size": 36}, \
                          "target": {"batch_size": 36}, \
                          "test": {"batch_size": 72},
                          "poison_test": {"batch_size": 72},

                          "root_folder": args.root_folder}
        config["network"]["params"]["class_num"] = 9



        config["lr_mult_im"] = args.lr_mult_im



        config["dataset_mult_iw"] = args.dataset_mult_iw
        config["out_log_file"].write(str(config) + "\n")
        config["out_log_file"].flush()


        print('Starting loading data')
        sys.stdout.flush()
        t_data = time.time()
        print('Found existing dataset for source')
        if args.s_dset == 'STL':
            d_source = data_loaders.load_stl(zero_centre=False)
        else:
            d_source = data_loaders.load_cifar10(range_01=False, val=False)
        source_samples, source_labels = torch.Tensor(d_source.train_X), torch.LongTensor(d_source.train_y)

        print('Found existing dataset for target and test')
        if args.t_dset == 'STL':
            d_target = data_loaders.load_stl(zero_centre=False)
        else:
            d_target = data_loaders.load_cifar10(range_01=False, val=False)
        target_samples, target_labels = torch.Tensor(d_target.train_X), torch.LongTensor(d_target.train_y)
        test_samples, test_labels = torch.Tensor(d_target.test_X), torch.LongTensor(d_target.test_y)
        #print("source sample shape", source_samples.shape)
        #print("source label shape", source_labels.shape)
        #print("target sample shape", target_samples.shape)
        #print("target label shape", target_labels.shape)
        #print("test sample shape", test_samples.shape)
        #print("test label shape", test_labels.shape)




        print("-" * 50, flush=True)
        print("\nRunning {} source {} and target {} and trade off {}\n".format(args.method,args.s_dset, args.t_dset, args.trade_off), flush=True )
        print("-" * 50, flush=True)
        train(config, source_samples, source_labels, target_samples, target_labels, test_samples, test_labels)