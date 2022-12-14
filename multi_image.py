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
from data.usps2mnist.noise_mnist import *
from tiny_imagenet.generate_poison_image import *
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
        for _, (data, target) in enumerate(dset):
            _, output = model(data)
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
    for i, (data, target) in enumerate(dset):
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


def shuffle_dataset(dset, shuffle=True):
    X_train = []
    y_train = []
    for i in range(len(dset)):
        img, label = dset[i]
        X_train.append(np.array(img))
        y_train.append(int(label))
    X_train = np.stack(X_train)
    y_train = np.array(y_train)
    if shuffle:
        # shuffle dataset
        shuffle_idx1 = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
        X_train = X_train[shuffle_idx1, :, :, :]
        y_train = y_train[shuffle_idx1]
    return X_train, y_train


def transform_image(data, transform):
    for i in range(data.shape[0]):
        data[i] = transform(data[i])
    return data


def train(config):

    ## Define start time
    start_time = time.time()

    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(norm=0)
    prep_dict["target"] = prep.image_train(norm=1)
    prep_dict["test"] = prep.image_test(norm=0)

    ## prepare data
    print("Preparing data", flush=True)
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    root_folder = data_config["root_folder"]
    dsets["source"] = ImageList(open(osp.join(root_folder, data_config["source"]["list_path"])).readlines(), \
                                transform=prep_dict["source"], root_folder=root_folder, ratios=config["ratios_source"])
    dsets["target"] = ImageList(open(osp.join(root_folder, data_config["target"]["list_path"])).readlines(), \
                                transform=prep_dict["target"], root_folder=root_folder, ratios=config["ratios_target"])
    dsets["test"] = ImageList(open(osp.join(root_folder, data_config["test"]["list_path"])).readlines(),
                              transform=prep_dict["test"], root_folder=root_folder, ratios=config["ratios_test"])
    # noisy label
    if args.noise_type != 'clean':
        source_labels = []
        target_labels = []
        for i, (img, l) in enumerate(dsets["source"].imgs):
            source_labels.append(l)
        for i, (img, l) in enumerate(dsets["target"].imgs):
            target_labels.append(l)
        #print("original source label", source_labels)
        #print("original target label", target_labels)
        #print("original source", dsets["source"].imgs)
        source_labels, source_actual_noise_rate, source_noise_or_not = noise_mnist(config["device"], args.noise_type,
                                                                                   torch.tensor(source_labels).to(config["device"]), args.noise_rate)
        target_labels, target_actual_noise_rate, target_noise_or_not = noise_mnist(config["device"], args.noise_type,
                                                                                   torch.tensor(target_labels).to(config["device"]), args.noise_rate)
        print("source_actual_noise_rate", source_actual_noise_rate, "source_noise_or_not", source_noise_or_not)
        print("target_actual_noise_rate", target_actual_noise_rate, "target_noise_or_not", target_noise_or_not)
        for i, (img, l) in enumerate(dsets["source"].imgs):
            dsets["source"].imgs[i] = (img, source_labels[i])
        for i, (img, l) in enumerate(dsets["target"].imgs):
            dsets["target"].imgs[i] = (img, target_labels[i])
    #shuffle source and target data
    source_samples, source_labels = shuffle_dataset(dsets["source"])
    target_samples, target_labels = shuffle_dataset(dsets["target"])


    test_samples, test_labels = shuffle_dataset(dsets["test"], False)
    print("finish shuffle dataset!")
    ##use backdoor attack to poison data
    # the position to add the trigger
    x, y = (200, 200)
    if args.poison != 0:
        source_samples, source_labels = generate_image(source_samples, source_labels, x, y, config["network"]["params"]["class_num"], args.poison_ratio)
    source_samples, source_labels = torch.Tensor(source_samples).to(
            config["device"]), torch.LongTensor(source_labels).to(config["device"])
    test_samples_poison, test_labels_poison = generate_image(test_samples, test_labels, x, y, config["network"]["params"]["class_num"], 1)
    test_samples_poison, test_labels_poison = torch.Tensor(test_samples_poison).to(
        config["device"]), torch.LongTensor(test_labels_poison).to(config["device"])
    target_samples, target_labels = torch.Tensor(target_samples).to(
            config["device"]), torch.LongTensor(target_labels).to(config["device"])
    test_samples, test_labels = torch.Tensor(test_samples).to(
        config["device"]), torch.LongTensor(test_labels).to(config["device"])
    print('finish poison! Poison data ratio is {}'.format(args.poison_ratio))

    #transform
    transform_method = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    source_samples = transform_image(source_samples, transform_method)
    test_samples = transform_image(test_samples, transform_method)
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



    dset_loaders["source"] = MoMDataLoader(training_source, train_bs)
    dset_loaders["target"] = MoMDataLoader(training_target, train_bs)
    dset_loaders["test"] = DataLoader(SimpleDataSet(test_samples, test_labels), batch_size=test_bs, \
                            shuffle=False, num_workers=4, drop_last=True)
    dset_loaders["test_poison"] = DataLoader(SimpleDataSet(test_samples_poison, test_labels_poison), batch_size=test_bs, \
                            shuffle=False, num_workers=4, drop_last=True)





    test_path = os.path.join(root_folder, data_config["test"]["dataset_path"])
    if os.path.exists(test_path):
        print('Found existing dataset for test', flush=True)
        with open(test_path, 'rb') as f:
            [test_samples, test_labels] = pickle.load(f)
            test_labels = torch.LongTensor(test_labels).to(config["device"])
    else:
        print('Missing test dataset', flush=True)
        print('Building dataset for test and writing to {}'.format(
            test_path), flush=True)
        dset_test = ImageList(open(osp.join(root_folder, data_config["test"]["list_path"])).readlines(),
                                transform=prep_dict["test"], root_folder=root_folder, ratios=config['ratios_test'])
        loaded_dset_test = LoadedImageList(dset_test)
        test_samples, test_labels = loaded_dset_test.samples.numpy(), loaded_dset_test.targets.numpy()
        with open(test_path, 'wb') as f:
            pickle.dump([test_samples, test_labels], f)

    class_num = config["network"]["params"]["class_num"]
    test_samples, test_labels = sample_ratios(
        test_samples, test_labels, config['ratios_test'])

    # compute labels distribution on the source and target domain
    source_label_distribution = np.zeros((class_num))
    for img in dsets["source"].imgs:
        source_label_distribution[img[1]] += 1
    print("Total source samples: {}".format(np.sum(source_label_distribution)), flush=True)
    print("Source samples per class: {}".format(source_label_distribution), flush=True)
    source_label_distribution /= np.sum(source_label_distribution)
    print("Source label distribution: {}".format(source_label_distribution), flush=True)
    target_label_distribution = np.zeros((class_num))
    for img in dsets["target"].imgs:
        target_label_distribution[img[1]] += 1
    print("Total target samples: {}".format(
        np.sum(target_label_distribution)), flush=True)
    print("Target samples per class: {}".format(target_label_distribution), flush=True)
    target_label_distribution /= np.sum(target_label_distribution)
    print("Target label distribution: {}".format(target_label_distribution), flush=True)
    mixture = (source_label_distribution + target_label_distribution) / 2
    jsd = (scipy.stats.entropy(source_label_distribution, qk=mixture) \
            + scipy.stats.entropy(target_label_distribution, qk=mixture)) / 2
    print("JSD : {}".format(jsd), flush=True)

    test_label_distribution = np.zeros((class_num))
    for img in test_labels:
        test_label_distribution[int(img.item())] += 1
    print("Test samples per class: {}".format(test_label_distribution), flush=True)
    test_label_distribution /= np.sum(test_label_distribution)
    print("Test label distribution: {}".format(test_label_distribution), flush=True)
    write_list(config["out_wei_file"], [round(x, 4) for x in test_label_distribution])
    write_list(config["out_wei_file"], [round(x, 4) for x in source_label_distribution])
    write_list(config["out_wei_file"], [round(x, 4) for x in target_label_distribution])
    true_weights = torch.tensor(
        target_label_distribution / source_label_distribution, dtype=torch.float, requires_grad=False)[:, None].to(config["device"])
    print("True weights : {}".format(true_weights[:, 0].cpu().numpy()))
    config["out_wei_file"].write(str(jsd) + "\n")

    ## train
    pred_test_all = []
    pred_test_poison_all = []
    pred_source_all = []
    len_test = len(dsets["test"])
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

        print("Preparations done in {:.0f} seconds".format(time.time() - start_time), flush=True)
        print("Starting training for {} iterations using method {}".format(config["num_iterations"], config['method']),
              flush=True)
        start_time_test = start_time = time.time()
        for i in range(config["num_iterations"]):
            if i % config["test_interval"] == config["test_interval"] - 1:
                base_network.train(False)
                temp_acc, temp_pred = image_classification_test_loaded(test_bs, dset_loaders["test"], base_network)
                temp_model = nn.Sequential(base_network)
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    best_pred_test = temp_pred
                log_str = "  iter: {:05d}, sec: {:.0f}, class: {:.5f}, da: {:.5f}, precision: {:.5f}".format(
                    i, time.time() - start_time_test, classifier_loss_value, transfer_loss_value, temp_acc)
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
            if args.corrupt != 'clean':
                for i in range(inputs_source.shape[0]):
                    inputs_source[i] = noisy(config["device"], args.corrupt, inputs_source[i])

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
        test_poison_acc, pred_test_poison = image_classification_test_loaded(test_bs, dset_loaders["test_poison"], base_network)
        pred_test_poison_all.append(pred_test_poison)
        vote(test_bs, pred_test_all, dset_loaders["test"], config["out_log_file"], name='Target test')
        vote(test_bs, pred_test_poison_all, dset_loaders["test"], config["out_log_file"], name='Target test poison')
        vote(test_bs, pred_test_poison_all, dset_loaders["test_poison"], config["out_log_file"], name='Target test attack succese rate')




    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, choices=[
                        'NANN', 'DANN', 'IWDAN', 'IWDANORACLE', 'CDAN', 'IWCDAN', 'IWCDANORACLE', 'CDAN-E', 'IWCDAN-E', 'IWCDAN-EORACLE'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"], help="Network type. Only tested with ResNet50")
    parser.add_argument('--dset', type=str, default='office-31', choices=['office-31', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_file', type=str, nargs='*', default='amazon_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_file', type=str, default='webcam_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=10000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='results', help="output directory")
    parser.add_argument('--root_folder', type=str, default=None, help="The folder containing the datasets")
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
    parser.add_argument('--noise_type', default='clean', choices=['clean', 'pairflip', 'symmetric'])
    parser.add_argument('--noise_rate', type=float, default=0.2,
                        help='noise rate for the label of training data')
    parser.add_argument('--corrupt', default='clean', choices=['gauss', 's&p', 'poisson', 'speckle'])
    parser.add_argument('--block', type=int, default=5,
                        help='The number of blocks')
    parser.add_argument('--poison', type=int, default=0,
                        help='poison or not')
    parser.add_argument('--poison_ratio', type=float, default=0.1,
                        help='The ratio of poison samples.')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    if args.root_folder is None:
        args.root_folder = 'data/{}/'.format(args.dset)

    if args.s_dset_file != args.t_dset_file:
        # Set GPU ID
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        # Set random number seed.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # train config
        config = {}
        config['method'] = args.method
        config["gpu"] = args.gpu_id
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        config["dataset"] = args.dset
        config["corrupt"] = args.corrupt
        config["data"] = {"source": {"list_path": args.s_dset_file,
                                     "dataset_path": "{}_source.pkl".format(args.s_dset_file),
                                     "batch_size": 36}, \
                              "target": {"list_path": args.t_dset_file,
                                         "dataset_path": "{}_target.pkl".format(args.t_dset_file),
                                         "batch_size": 36}, \
                              "test": {"list_path": args.t_dset_file,
                                       "dataset_path": "{}_test.pkl".format(args.t_dset_file), "batch_size": 72},
                          "poison_test": {"list_path": args.t_dset_file,
                                   "dataset_path": "{}_poison_test.pkl".format(args.t_dset_file), "batch_size": 72},

                              "root_folder": args.root_folder}


        config["lr_mult_im"] = args.lr_mult_im
        if config["dataset"] == "office-31":
            if ("amazon" in args.s_dset_file and "webcam" in args.t_dset_file) or \
            ("webcam" in args.s_dset_file and "dslr" in args.t_dset_file) or \
            ("webcam" in args.s_dset_file and "amazon" in args.t_dset_file) or \
            ("dslr" in args.s_dset_file and "amazon" in args.t_dset_file):
                config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
            elif ("amazon" in args.s_dset_file and "dslr" in args.t_dset_file) or \
                ("dslr" in args.s_dset_file and "webcam" in args.t_dset_file):
                config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
            config["network"]["params"]["class_num"] = 31
            config["ratios_source"] = [1] * 31
            if args.ratio == 1:
                config["ratios_source"] = [0.3] * 15 + [1] * 16
            config["ratios_target"] = [1] * 31
            if args.dataset_mult_iw == 0:
                args.dataset_mult_iw = 15
        elif config["dataset"] == "visda":
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
            config["network"]["params"]["class_num"] = 12
            config["ratios_source"] = [1] * 12
            if args.ratio == 1:
                config["ratios_source"] = [0.3] * 6 + [1] * 6
            config["ratios_target"] = [1] * 12
            if args.dataset_mult_iw == 0:
                args.dataset_mult_iw = 1
        elif config["dataset"] == "office-home":
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
            config["network"]["params"]["class_num"] = 65
            config["ratios_source"] = [1] * 65
            if args.ratio == 1:
                config["ratios_source"] = [0.3] * 32 + [1] * 33
            config["ratios_target"] = [1] * 65
            if args.dataset_mult_iw == 0:
                args.dataset_mult_iw = 15
        else:
            raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')



        config["dataset_mult_iw"] = args.dataset_mult_iw
        config["ratios_test"] = config["ratios_target"]
        config["out_log_file"].write(str(config) + "\n")
        config["out_log_file"].flush()


        print("-" * 50, flush=True)
        print("\nRunning {} on the {} dataset with source {} and target {} and trade off {}\n".format(args.method, args.dset,args.s_dset_file, args.t_dset_file, args.trade_off), flush=True )
        print("-" * 50, flush=True)
        train(config)