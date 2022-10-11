# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np
import os
import pickle
import scipy.stats
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import loss as loss_func
import network
from data_list import build_uspsmnist, sample_ratios, subsampling, build_digits
from data.usps2mnist.noise_mnist import noisy, noise_mnist
from imgaug import augmenters as iaa
from PIL import Image
import os.path as osp
from scipy import stats
from tiny_imagenet.generate_poison import *
from poison_crafting.craft_poisons_clbd import clbd_attack, clbd_attack_digits

def pil_loader(args, path):
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



def write_list(f, l):
    f.write(",".join(map(str, l)) + "\n")
    f.flush()
    sys.stdout.flush()


def train(args, model, ad_net,
          source_samples, source_labels, target_samples, target_labels, optimizer, optimizer_ad,
          epoch, start_epoch, method,
          source_label_distribution, out_wei_file,
          cov_mat, pseudo_target_label, class_weights, true_weights):
    model.train()

    cov_mat[:] = 0.0
    pseudo_target_label[:] = 0.0

    len_source = source_labels.shape[0]
    len_target = target_labels.shape[0]


    size = max(len_source, len_target)
    num_iter = int(size / args.batch_size)

    for batch_idx in range(num_iter):
        t = time.time()
        source_idx = np.random.choice(len_source, args.batch_size)
        target_idx = np.random.choice(len_target, args.batch_size)
        data_source, label_source = source_samples[source_idx], source_labels[source_idx]
        data_target, _ = target_samples[target_idx], target_labels[target_idx]


        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        feature, output = model(torch.cat((data_source, data_target), 0))

        if 'IW' in method:
            ys_onehot = torch.zeros(args.batch_size, 10).to(args.device)
            ys_onehot.scatter_(1, label_source.view(-1, 1), 1)
            # Compute weights on source data.
            if 'ORACLE' in method:
                weights = torch.mm(ys_onehot, true_weights)
            else:
                weights = torch.mm(ys_onehot, model.im_weights)

            source_preds, target_preds = output[:
                                                args.batch_size], output[args.batch_size:]
            # Compute the aggregated distribution of pseudo-label on the target domain.
            pseudo_target_label += torch.sum(
                F.softmax(target_preds, dim=1), dim=0).view(-1, 1).detach()
            # Update the covariance matrix on the source domain as well.
            cov_mat += torch.mm(F.softmax(source_preds,
                                            dim=1).transpose(1, 0), ys_onehot).detach()

            loss = torch.mean(
                nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                (output.narrow(0, 0, data_source.size(0)), label_source) * weights) / 10.0
        else:
            loss = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)

        if epoch > start_epoch:
            if method == 'CDAN-E':
                softmax_output = nn.Softmax(dim=1)(output)
                entropy = loss_func.Entropy(softmax_output)
                loss += loss_func.CDAN([feature, softmax_output], ad_net, entropy, network.calc_coeff(
                    num_iter*(epoch-start_epoch)+batch_idx), None, device=args.device)

            elif 'IWCDAN-E' in method:
                    softmax_output = nn.Softmax(dim=1)(output)
                    entropy = loss_func.Entropy(softmax_output)
                    loss += loss_func.CDAN([feature, softmax_output], ad_net, entropy, network.calc_coeff(
                        num_iter*(epoch-start_epoch)+batch_idx), None, weights=weights, device=args.device)

            elif method == 'CDAN':
                    softmax_output = nn.Softmax(dim=1)(output)
                    loss += loss_func.CDAN([feature, softmax_output],
                                        ad_net, None, None, None, device=args.device)

            elif 'IWCDAN' in method:
                    softmax_output = nn.Softmax(dim=1)(output)
                    loss += loss_func.CDAN([feature, softmax_output],
                                        ad_net, None, None, None, weights=weights, device=args.device)

            elif method == 'DANN':
                loss += loss_func.DANN(feature, ad_net, args.device)

            elif 'IWDAN' in method:
                    dloss = loss_func.IWDAN(feature, ad_net, weights)
                    loss += args.mu * dloss

            elif method == 'NANN':
                pass

            else:
                raise ValueError('Method cannot be recognized.')

        loss.backward()
        optimizer.step()

        if epoch > start_epoch and method != 'NANN':
            optimizer_ad.step()

    if 'IW' in method  and epoch > start_epoch:
        pseudo_target_label /= args.batch_size * num_iter
        cov_mat /= args.batch_size * num_iter
        # Recompute the importance weight by solving a QP.
        model.im_weights_update(source_label_distribution,
                                pseudo_target_label.cpu().detach().numpy(),
                                cov_mat.cpu().detach().numpy(),
                                args.device)
        current_weights = [round(x, 4) for x in model.im_weights.data.cpu().numpy().flatten()]
        write_list(out_wei_file, [np.linalg.norm(
            current_weights - true_weights.cpu().numpy().flatten())] + current_weights)
        print(np.linalg.norm(current_weights - true_weights.cpu().numpy().flatten()), current_weights)


def test(args, epoch, model, test_samples, test_labels, start_time_test, out_log_file, name=''):
    model.eval()
    test_loss = 0
    correct = 0
    len_test = test_labels.shape[0]
    pred_result = []
    for i in range(len_test):
        sample, target = test_samples[i].unsqueeze(0), test_labels[i].unsqueeze(0)
        _, output = model(sample)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.data.cpu().max(1, keepdim=True)[1]
        correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()
        pred_result.append(pred)

    test_loss /= len_test
    temp_acc = 100. * correct / len_test
    log_str = "  {}, iter: {:05d}, sec: {:.0f}, loss: {:.5f}, accuracy: {}/{}, precision: {:.5f}".format(name, epoch, time.time() - start_time_test, test_loss, correct, len_test, temp_acc)
    print(log_str)
    sys.stdout.flush()
    out_log_file.write(log_str+"\n")
    out_log_file.flush()
    return pred_result

def vote(pred, test_labels, out_log_file, name):
    num = len(pred)
    len_data = test_labels.shape[0]
    correct = 0
    for i in range(len_data):
        pred_i = []
        for j in range(num):
            pred_i.append(pred[j][i])
        y = torch.tensor(stats.mode(pred_i)[0][0])
        target = test_labels[i].unsqueeze(0)
        correct += y.eq(target.data.cpu().view_as(y)).sum().item()
    temp_acc = 100. * correct / len_data
    log_str = "{}: Accuracy for MOM{:.5f}".format(name, temp_acc)
    print(log_str)
    sys.stdout.flush()
    out_log_file.write(log_str + "\n")
    out_log_file.flush()

def normalize_image(samples):
    Norm_method = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    for i in range(samples.shape[0]):
        samples[i] = Norm_method(samples[i])
    return samples



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CDAN USPS MNIST')
    parser.add_argument('method', type=str, default='CDAN-E',
                        choices=['CDAN', 'CDAN-E', 'DANN', 'IWDAN', 'NANN', 'IWDANORACLE', 'IWCDAN', 'IWCDANORACLE', 'IWCDAN-E', 'IWCDAN-EORACLE'])
    parser.add_argument('--s_dset', type=str, default='MNIST', help="The source dataset path list")
    parser.add_argument('--t_dset', type=str, default='MNIST_M', help="The target dataset path list")
    parser.add_argument('--task', default='mnist2usps', help='task to perform', choices=['usps2mnist', 'mnist2usps'])
    parser.add_argument('--noise_type', default='clean', choices=['clean', 'pairflip', 'symmetric'])
    parser.add_argument('--noise_rate', type=float, default=0.2,
                        help='noise rate for the label of training data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=70, metavar='N',
                        help='number of epochs to train (default: 70)')
    parser.add_argument('--lr', type=float, default=0.0, metavar='LR',
                        help='learning rate (default: 0.02)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--root_folder', type=str, default='data/', help="The folder containing the datasets and the lists")
    parser.add_argument('--output_dir', type=str, default='results', help="output directory")
    parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient of the domain adversarial loss", type=float, default=1.0)
    parser.add_argument('--ratio', type =float, default=0, help='ratio option')
    parser.add_argument('--ma', type=float, default=0.5, help='weight for the moving average of iw')
    parser.add_argument('--corrupt', default='clean', choices=['badnet', 'clean', 'clbd'])
    parser.add_argument('--block', type=int, default=5,
                        help='The number of blocks')
    parser.add_argument('--poison_ratio', type=float, default=0.05,
                        help='The ratio of poison samples.')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Running the JSD experiment on fewer epochs for efficiency
    if args.ratio >= 100:
        args.epochs = 25

    print('Running {} on {} for {} epochs on task {}'.format(
        args.method, args.device, args.epochs, args.task))

    # Set random number seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # CDAN parameters
    decay_epoch = 6
    decay_frac = 0.5
    lr = 0.02
    start_epoch = 1
    # model = network.LeNet(args.ma)
    build_dataset = build_uspsmnist
    transform_map = {}
    transform_map['MNIST'] = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    transform_map['MNIST_M'] = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_map['SVHN'] = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
    ])
    transform_map['SynthDigits'] = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
    ])
    transform_map['USPS'] = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])


    source_path = os.path.join(args.root_folder, '{}/partitions'.format(args.s_dset))
    target_path = os.path.join(args.root_folder, '{}/partitions'.format(args.t_dset))
    test_path = os.path.join(args.root_folder, '{}/test.pkl'.format(args.t_dset))



    out_log_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    out_log_file_train = open(os.path.join(
        args.output_dir, "log_train.txt"), "w")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #model = model.to(args.device)
    class_num = 10

    if args.lr > 0:
      lr = args.lr

    print('Starting loading data')
    sys.stdout.flush()
    t_data = time.time()
    if os.path.exists(source_path):
        print('Found existing dataset for source')
        start = True
        for i in range(10):
            source_path_i = source_path + '/train_part{}.pkl'.format(i)
            with open(source_path_i, 'rb') as f:
                [source_samples_i, source_labels_i] = pickle.load(f)
                source_samples_i, source_labels_i = torch.Tensor(source_samples_i).to(
                    args.device), torch.LongTensor(source_labels_i).to(args.device)
                if start:
                    source_samples, source_labels = source_samples_i, source_labels_i
                    start = False
                else:
                    source_samples = torch.cat((source_samples, source_samples_i),0)
                    source_labels = torch.cat((source_labels, source_labels_i),0)

    if os.path.exists(target_path):
        print('Found existing dataset for target')
        for i in range(10):
            target_path_i = target_path + '/train_part{}.pkl'.format(i)
            start = True
            with open(target_path_i, 'rb') as f:
                [target_samples_i, target_labels_i] = pickle.load(f)
                target_samples_i, target_labels_i = torch.Tensor(
                    target_samples_i).to(args.device), torch.LongTensor(target_labels_i).to(args.device)
                if start:
                    target_samples, target_labels = target_samples_i, target_labels_i
                    start = False
                else:
                    target_samples = torch.cat((target_samples, target_samples_i),0)
                    target_labels = torch.cat((target_labels, target_labels_i),0)


    if os.path.exists(test_path):
        print('Found existing dataset for test')
        with open(test_path, 'rb') as f:
            [test_samples, test_labels] = pickle.load(f)
            test_samples, test_labels = torch.Tensor(
                test_samples).to(args.device), torch.LongTensor(test_labels).to(args.device)

    # transform 1 channel to 3 channels
    source_samples = build_digits(args, source_samples, transform_map[args.s_dset])
    target_samples = build_digits(args, target_samples, transform_map[args.t_dset])
    test_samples = build_digits(args, test_samples, transform_map[args.t_dset])

    if args.ratio != 0:
        # RATIO OPTION 1
        # 30% of the samples from the first 5 classes
        print('Subsample ratio', args.ratio)
        ratios_source = [args.ratio] * 10
        ratios_target = [1] * 10
        ratios_test = ratios_target
        # Subsample dataset if need be
        source_samples, source_labels = sample_ratios(
            source_samples, source_labels, ratios_source)
        target_samples, target_labels = sample_ratios(
            target_samples, target_labels, ratios_target)
        test_samples, test_labels = sample_ratios(
            test_samples, test_labels, ratios_test)



   ##use backdoor attack to poison data
    # the position to add the trigger
    x, y = np.random.choice([2, 25]), np.random.choice([2, 25])
    if args.corrupt == 'badnet':
        if args.poison_ratio > 0:
            source_samples, source_labels = generate_image2(source_samples, source_labels, x, y, 10, args.poison_ratio)
        test_samples_poison, test_labels_poison = generate_image2(test_samples,test_labels, x, y, 10, 1)
        test_samples_poison, test_labels_poison = torch.Tensor(test_samples_poison).to(
            args.device), torch.LongTensor(test_labels_poison).to(args.device)
    elif args.corrupt == 'clbd':
        source_samples, source_labels, test_samples, test_labels, test_samples_poison, test_labels_poison = clbd_attack_digits(source_samples,source_labels, test_samples, test_labels, args.poison_ratio, args.ma)
    print('finish poison! Poison data number is {}'.format(args.poison_ratio))
    source_samples, source_labels = torch.Tensor(source_samples).to(
        args.device), torch.LongTensor(source_labels).to(args.device)
    #source_samples = source_samples.permute(0, 2, 3, 1)
    #target_samples = target_samples.permute(0, 2, 3, 1)
    #test_samples = test_samples.permute(0, 2, 3, 1)

    source_samples = normalize_image(source_samples)
    target_samples = normalize_image(target_samples)
    test_samples = normalize_image(test_samples)
    if args.corrupt != 'clean':
        test_samples_poison = normalize_image(test_samples_poison)
    print("finish normalize!")


    print('Data loaded in {}'.format(time.time() - t_data))



    # noisy label
    if args.noise_type != 'clean':
        source_labels, source_actual_noise_rate, source_noise_or_not = noise_mnist(args.device, args.noise_type,
                                                                                   source_labels, args.noise_rate)
        target_labels, target_actual_noise_rate, target_noise_or_not = noise_mnist(args.device, args.noise_type,
                                                                                   target_labels, args.noise_rate)
        print("source_actual_noise_rate", source_actual_noise_rate, "source_noise_or_not", source_noise_or_not)
        print("target_actual_noise_rate", target_actual_noise_rate, "target_noise_or_not", target_noise_or_not)


    # compute labels distribution on the source and target domain
    source_label_distribution = np.zeros((class_num))
    for img in source_labels:
        source_label_distribution[int(img.item())] += 1
    print("Total source samples: {}".format(
        np.sum(source_label_distribution)), flush=True)
    print("Source samples per class: {}".format(source_label_distribution))
    source_label_distribution /= np.sum(source_label_distribution)
    write_list(out_log_file, source_label_distribution)
    print("Source label distribution: {}".format(source_label_distribution))
    target_label_distribution = np.zeros((class_num))
    for img in target_labels:
        target_label_distribution[int(img.item())] += 1
    print("Total target samples: {}".format(
        np.sum(target_label_distribution)), flush=True)
    print("Target samples per class: {}".format(target_label_distribution))
    target_label_distribution /= np.sum(target_label_distribution)
    write_list(out_log_file, target_label_distribution)
    print("Target label distribution: {}".format(target_label_distribution))
    test_label_distribution = np.zeros((class_num))
    for img in test_labels:
        test_label_distribution[int(img.item())] += 1
    print("Test samples per class: {}".format(test_label_distribution))
    test_label_distribution /= np.sum(test_label_distribution)
    write_list(out_log_file, test_label_distribution)
    print("Test label distribution: {}".format(test_label_distribution))
    mixture = (source_label_distribution + target_label_distribution) / 2
    jsd = (scipy.stats.entropy(source_label_distribution, qk=mixture)
           + scipy.stats.entropy(target_label_distribution, qk=mixture)) / 2
    print("JSD source to target : {}".format(jsd))
    log_str = "JSD source to target : {}".format(jsd)
    sys.stdout.flush()
    out_log_file.write(log_str + "\n")
    out_log_file.flush()
    mixture_2 = (test_label_distribution + target_label_distribution) / 2
    jsd_2 = (scipy.stats.entropy(test_label_distribution, qk=mixture_2)
           + scipy.stats.entropy(target_label_distribution, qk=mixture_2)) / 2
    print("JSD test to target : {}".format(jsd_2))
    log_str = "JSD test to target : {}".format(jsd_2)
    sys.stdout.flush()
    out_log_file.write(log_str + "\n")
    out_log_file.flush()
    out_wei_file = open(os.path.join(args.output_dir, "log_weights_{}.txt".format(jsd)), "w")
    write_list(out_wei_file, [round(x, 4) for x in source_label_distribution])
    write_list(out_wei_file, [round(x, 4) for x in target_label_distribution])
    out_wei_file.write(str(jsd) + "\n")
    true_weights = torch.tensor(
        target_label_distribution / source_label_distribution, dtype=torch.float, requires_grad=False)[:, None].to(args.device)
    print("True weights : {}".format(true_weights[:, 0].cpu().numpy()))




    # shuffle dataset
    shuffle_idx1 = np.random.choice(source_samples.shape[0], source_samples.shape[0], replace=False)
    source_samples = source_samples[shuffle_idx1, :, :, :]
    source_labels = source_labels[shuffle_idx1]
    shuffle_idx2 = np.random.choice(target_samples.shape[0], target_samples.shape[0], replace=False)
    target_samples = target_samples[shuffle_idx2, :, :, :]
    target_labels = target_labels[shuffle_idx2]
    # divide blocks
    n_size_s = int(source_samples.shape[0] / args.block)
    n_size_t = int(target_samples.shape[0] / args.block)
    training_data = []
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
        training_data.append((source_samples_i, source_labels_i, target_samples_i, target_labels_i))
    print('finish devide blocks!')

    pred_test_all = []
    pred_test_poison_all = []
    pred_source_all = []

    for model_idx in range(args.block):
        print("model", model_idx)
        model = network.Model_digit().to(args.device)
        source_samples, source_labels, target_samples, target_labels = training_data[model_idx]
        if 'CDAN' in args.method:
            ad_net = network.AdversarialNetwork(
                model.output_num() * class_num, 500, sigmoid='WDANN' not in args.method)
        else:
            ad_net = network.AdversarialNetwork(
                model.output_num(), 500, sigmoid='WDANN' not in args.method)

        ad_net = ad_net.to(args.device)

        optimizer = optim.SGD(model.parameters(), lr=lr,
                              weight_decay=0.0005, momentum=0.9)
        optimizer_ad = optim.SGD(
            ad_net.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)

        # Maintain two quantities for the QP.
        cov_mat = torch.tensor(np.zeros((class_num, class_num), dtype=np.float32),
                               requires_grad=False).to(args.device)
        pseudo_target_label = torch.tensor(np.zeros((class_num, 1), dtype=np.float32),
                                           requires_grad=False).to(args.device)
        # Maintain one weight vector for BER.
        class_weights = torch.tensor(
            1.0 / source_label_distribution, dtype=torch.float, requires_grad=False).to(args.device)
        for epoch in range(1, args.epochs + 1):
            start_time_test = time.time()
            if epoch % decay_epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * decay_frac
            test(args, epoch, model, test_samples, test_labels, start_time_test, out_log_file, name='Target test')
            if args.corrupt != 'clean':
                test(args, epoch + 1, model, test_samples_poison, test_labels_poison,
                     0, out_log_file,
                     name='Target test attack succese rate')
            train(args, model, ad_net, source_samples,
                  source_labels, target_samples, target_labels,
                  optimizer, optimizer_ad, epoch, start_epoch, args.method, source_label_distribution, out_wei_file,
                  cov_mat, pseudo_target_label, class_weights, true_weights)
        pred_test = test(args, epoch + 1, model, test_samples, test_labels, start_time_test, out_log_file, name='Target test')
        pred_test_all.append(pred_test)
        pred_source = test(args, epoch + 1, model, source_samples, source_labels,
             start_time_test, out_log_file_train, name='Source train')
        pred_source_all.append(pred_source)
        if args.corrupt != 'clean':
            pred_test_poison = test(args, epoch + 1, model, test_samples_poison, test_labels, start_time_test, out_log_file,
                                name='Target test poison')
            pred_test_poison_all.append(pred_test_poison)

    vote(pred_test_all, test_labels, out_log_file, name='Target test')
    if args.corrupt != 'clean':
        vote(pred_test_poison_all, test_labels, out_log_file, name='Target test poison')
        vote(pred_test_poison_all, test_labels_poison, out_log_file, name='Target test attack succese rate')




if __name__ == '__main__':
    main()