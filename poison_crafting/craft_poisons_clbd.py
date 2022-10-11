############################################################
#
# craft_poisons_clbd.py
# Clean-label Backdoor Attack
# June 2020
#
# Reference: A. Turner, D. Tsipras, and A. Madry. Clean-label
#     backdoor attacks. 2018.
############################################################
import argparse
import os
import pickle
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F
from network import LeNet, Model_digit

sys.path.append(os.path.realpath("."))
from .learning_module import (
    TINYIMAGENET_ROOT,
    load_model_from_checkpoint,
    now,
    get_transform,
    NormalizeByChannelMeanStd,
    data_mean_std_dict,
    PoisonedDataset,
    compute_perturbation_norms,
get_model,
get_Poison_dataset

)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from tinyimagenet_module import TinyImageNet
class SimpleDataSet(Dataset):
    """ load synthetic time series data"""
    def __init__(self, x, y):
        self.x = x
        self.targets = y

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
            self.targets[idx],
        )

class AttackPGD(nn.Module):
    """Class for the PGD adversarial attack"""

    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = config["step_size"]
        self.epsilon = config["epsilon"]
        self.num_steps = config["num_steps"]
        self.network_name = config['model']

    def forward(self, inputs, targets):
        """Forward function for the nn class
        inputs:
            inputs:     The input to the network
            targets:    True labels
        reutrn:
            adversarially perturbed inputs
        """
        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                if self.network_name == 'LeNet' or self.network_name == 'model_digit':
                    loss = nn.functional.cross_entropy(
                        self.basic_net(x)[1], targets, reduction="sum"
                    )
                else:
                    loss = nn.functional.cross_entropy(
                        self.basic_net(x), targets, reduction="sum"
                    )
            grad = torch.autograd.grad(loss, [x], retain_graph=False, create_graph=False)[0]
            x = x.detach() + self.step_size * grad.sign()
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return x


def clbd(config, source_samples,source_labels, test_samples, test_labels):
    """Main function to generate the CLBD poisons
    inputs:
        args:           Argparse object
    reutrn:
        void
    """

    print(now(), "craft_poisons_clbd.py main() running...")
    mean, std = data_mean_std_dict['cifar10']
    mean = list(mean)
    std = list(std)
    normalize_net = NormalizeByChannelMeanStd(mean, std)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model_from_checkpoint(
        config['model'], config['model_path'], config['pretrain_dataset']
    )
    model.eval()
    if config['normalize']:
        model = nn.Sequential(normalize_net, model)
    model = model.to(device)



    with open(config['poison_setups'], "rb") as handle:
        setup_dicts = pickle.load(handle)
    setup = setup_dicts[config['setup_idx']]

    base_indices = (
        setup["base indices"] if config["base_indices"] is None else config["base_indices"]
    )

    # get single target
    target_img = torch.stack([test_samples[i] for i in config["target_img_idx"]],0).to(device)
    target_label_poison = torch.LongTensor(config["poison_label"]*np.ones((len(config["target_img_idx"])))).to(device)
    target_label = torch.LongTensor([test_labels[i] for i in config["target_img_idx"]]).to(device)


    # get multiple bases
    base_imgs = torch.stack([source_samples[i] for i in base_indices], 0).to(device)
    base_labels = torch.LongTensor([source_labels[i].item() for i in base_indices]).to(device)

    # get attacker

    attacker = AttackPGD(model, config)

    # get patch

    if config['model'] == "LeNet":
        trans_trigger = transforms.Compose(
            [transforms.Resize((config['patch_size'], config['patch_size'])), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        trigger = Image.open("./poison_crafting/triggers/clbd.png").convert("L")
        trigger = trans_trigger(trigger).to(device)
    else:
        trans_trigger = transforms.Compose(
            [transforms.Resize((config['patch_size'], config['patch_size'])), transforms.ToTensor()]
        )
        trigger = Image.open("./poison_crafting/triggers/clbd.png").convert("RGB")
        trigger = trans_trigger(trigger).unsqueeze(0).to(device)
    # Starting coordinates of the patch
    start_x = config['image_size'] - config['patch_size']
    start_y = config['image_size'] - config['patch_size']

    #base_imgs[
    #:,
    #:,
    #start_y: start_y + config['patch_size'],
    #start_x: start_x + config['patch_size'],
    #] = trigger

    # craft poisons
    num_batches = int(np.ceil(base_imgs.shape[0] / 1000))
    batches = [
        (base_imgs[1000 * i : 1000 * (i + 1)], base_labels[1000 * i : 1000 * (i + 1)])
        for i in range(num_batches)
    ]
    # attack all the bases
    adv_batches = []
    for batch_img, batch_labels in batches:
        adv_batches.append(attacker(batch_img, batch_labels))
    adv_bases = torch.cat(adv_batches)



    # Mask
    mask = torch.ones_like(adv_bases)

    # uncomment for patching all corners
    mask[
        :, start_y : start_y + config['patch_size'], start_x : start_x + config['patch_size']
    ] = 0
    # mask[:, 0 : args.patch_size, start_x : start_x + args.patch_size] = 0
    # mask[:, start_y : start_y + args.patch_size, 0 : args.patch_size] = 0
    # mask[:, 0 : args.patch_size, 0 : args.patch_size] = 0

    pert = (adv_bases - base_imgs) * mask
    adv_bases_masked = base_imgs + pert
    # Attching patch to the masks
    for i in range(len(base_imgs)):
        # uncomment for patching all corners
        adv_bases_masked[
            i,
            :,
            start_y : start_y + config['patch_size'],
            start_x : start_x + config['patch_size'],
        ] = trigger
        # adv_bases_masked[
        #     i, :, 0 : args.patch_size, start_x : start_x + args.patch_size
        # ] = trigger
        # adv_bases_masked[
        #     i, :, start_y : start_y + args.patch_size, 0 : args.patch_size
        # ] = torch.flip(trigger, (-1,))
        # adv_bases_masked[i, :, 0 : args.patch_size, 0 : args.patch_size] = torch.flip(
        #     trigger, (-1,)
        # )

    final_pert = torch.clamp(adv_bases_masked - base_imgs, -config['epsilon'], config['epsilon'])
    base_imgs[
    :,
    :,
    start_y: start_y + config['patch_size'],
    start_x: start_x + config['patch_size'],
    ]= trigger
    poisons = base_imgs + final_pert


    poisons = poisons.clamp(0, 1).cpu()
    poisoned_tuples = [
        ((poisons[i]), base_labels[i].item())
        for i in range(poisons.shape[0])
    ]

    target_tuple = (
        target_img, target_label_poison,
        trigger.squeeze(0).cpu(),
        [start_x, start_y],
    )

    ####################################################
    #        Save Poisons
    print(now(), "Saving poisons...")
    if not os.path.isdir(config['poisons_path']):
        os.makedirs(config['poisons_path'])
    with open(os.path.join(config['poisons_path'], "poisons.pickle"), "wb") as handle:
        pickle.dump(poisoned_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(config['poisons_path'], "target.pickle"), "wb") as handle:
        pickle.dump(
            target_tuple,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with open(os.path.join(config['poisons_path'], "base_indices.pickle"), "wb") as handle:
        pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ####################################################

    print(now(), "craft_poisons_clbd.py done.")
    return target_img, target_label

def transform_image(data, transform):
    for i in range(data.shape[0]):
        data[i] = transform(data[i])
    return data

def show_image(img):
	o_img = img.permute(1, 2, 0)*255
	o_img = Image.fromarray(np.array(o_img).astype('uint8').squeeze())
	o_img.show()
	return o_img

def clbd_attack(source_samples,source_labels, test_samples, test_labels, poison_ratio):
    config = {}
    config['epsilon'] = 16 / 255
    config['model'] = "resnet18"
    config['model_path'] = "poison_crafting/pretrained_models/ResNet18_CIFAR10.pth"
    config['num_steps'] = 40
    config['step_size'] = 2 / 255
    config['poison_setups'] = "poison_setups/cifar10_transfer_learning.pickle"
    config['poisons_path'] = "poison_examples/clbd_poisons_cifar"
    config['pretrain_dataset'] = 'CIFAR10'
    config['normalize'] = True
    config['patch_size'] = 5
    config['image_size'] = 32
    config['setup_idx'] = 0
    config['iteration'] = 10000
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config["target_class"] = 4
    config["poison_label"] = 2
    config["test_poison_label"] = [i for i in range(9) if i != config["poison_label"]]
    config["test_poison_ratio"] = [1, 1, 1, 1, 1, 1, 1, 1, 1]


    source_labels_copy = source_labels.cpu().numpy().copy()
    test_labels_copy = test_labels.cpu().numpy().copy()
    ind = np.argwhere(source_labels_copy == config["poison_label"])
    poison_num = int(len(list(ind.squeeze())) * poison_ratio)
    print("poison num", poison_num)
    config["base_indices"] = random.sample(list(ind.squeeze()), poison_num)
    poison_num_test = 0
    config["target_img_idx"] = []
    for i in config["test_poison_label"]:
        ind = np.argwhere(test_labels_copy == i)
        poison_num = int(len(list(ind.squeeze())) * config["test_poison_ratio"][i])
        poison_num_test += poison_num
        target_ind = random.sample(list(ind.squeeze()), poison_num)
        config["target_img_idx"].extend(target_ind)
    print("poison num test", poison_num_test)

    #print("base_indices", config["base_indices"])
    #print("target_img_idx", config["target_img_idx"])

    if os.path.exists(config['model_path']):
        print("The pretrain model already exsit!")
    else:
        print("get the pretrain model!")
        net = get_model(config['model'], "CIFAR10").to(config["device"])
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        # transform
        transform_method = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        source_samples_t = transform_image(source_samples, transform_method)
        train_loader = DataLoader(SimpleDataSet(source_samples_t,source_labels), batch_size=32, shuffle=True,)
        len_train = len(train_loader)
        net.train()
        for i in range(config['iteration']):
            if i % len_train == 0:
                iter_source = iter(train_loader)
            inputs_source, label_source = iter_source.next()

            data, target = inputs_source.to(config["device"]), label_source.to(config["device"])

            # forward
            y = net(data)
            # backward

            optimizer.zero_grad()
            loss = F.cross_entropy(y, target)

            loss.backward()

            optimizer.step()
            if i%100 == 0:
                print("iteration {}: loss {}".format(i,loss))
        torch.save(net.state_dict(), config['model_path'])
        print("saving pretrain model!")



    trainset = SimpleDataSet(source_samples, source_labels)

    test_samples, test_labels = clbd(config, source_samples,source_labels, test_samples, test_labels)
    # load the poisons and their indices within the training set from pickled files
    with open(os.path.join(config['poisons_path'], "poisons.pickle"), "rb") as handle:
        poison_tuples = pickle.load(handle)
        print(len(poison_tuples), " poisons in this trial.")
        poisoned_label = poison_tuples[0][1]
    with open(os.path.join(config['poisons_path'], "base_indices.pickle"), "rb") as handle:
        poison_indices = pickle.load(handle)
    # get the dataset
    trainset_poison = PoisonedDataset(
        trainset, poison_tuples, None, None, poison_indices
    )
    #show poison train sample
    #count = 0
    #for j in range(len(trainset_poison)):
    #    if trainset_poison[j][2] == 1:
    #        print("show poison image", show_image(trainset_poison[j][0]))
    #        print("show poison label", trainset_poison[j][1], trainset_poison[j][2])
    #        count += 1
    #        if count == 2:
    #            break



    # get the target image from pickled file
    with open(os.path.join(config['poisons_path'], "target.pickle"), "rb") as handle:
        target_img_tuple = pickle.load(handle)
        target_class = target_img_tuple[1]
        if len(target_img_tuple) == 4:
            patch = target_img_tuple[2] if torch.is_tensor(target_img_tuple[2]) else \
                torch.tensor(target_img_tuple[2])
            if patch.shape[0] != 3 or patch.shape[1] != config['patch_size'] or \
                    patch.shape[2] != config['patch_size']:
                print(
                    f"Expected shape of the patch is [3, {config['patch_size']}, {config['patch_size']}] "
                    f"but is {patch.shape}. Exiting from poison_test.py."
                )
                sys.exit()

            startx, starty = target_img_tuple[3]
            target_img_pil = target_img_tuple[0]
            h, w = target_img_pil.shape[2], target_img_pil.shape[3]

            if starty + config['patch_size'] > h or startx + config['patch_size'] > w:
                print(
                    "Invalid startx or starty point for the patch. Exiting from poison_test.py."
                )
                sys.exit()

            target_img_tensor = target_img_pil
            target_img_tensor[:, :, starty: starty + config['patch_size'],
            startx: startx + config['patch_size']] = patch

    #get source samples and source labels
    X_poisoned = np.stack([trainset_poison[i][0] for i in range(len(trainset_poison))], 0)
    Y_poisoned = np.stack([trainset_poison[i][1] for i in range(len(trainset_poison))], 0)

        #target_img = transform_test(target_img_pil)

    poison_perturbation_norms = compute_perturbation_norms(
        poison_tuples, trainset, poison_indices
    )


    #show poison test sample
    #count = 0
    #for j in range(target_img_tensor.shape[0]):
    #    print("show poison image", show_image(target_img_tensor[j]))
    #    print("show poison label", target_class[j])
    #    count += 1
    #    if count == 2:
    #       break

    # the limit is '8/255' but we assert that it is smaller than 9/255 to account for PIL
    # truncation.
    #assert max(poison_perturbation_norms) - config['epsilon'] < 1e-5, "Attack not clean label!"
    return X_poisoned, Y_poisoned, test_samples.cpu().numpy(), test_labels.cpu().numpy(), target_img_tensor, target_class


def clbd_attack_digits(source_samples,source_labels, test_samples, test_labels, poison_ratio, ma):
    config = {}
    config['epsilon'] = 16 / 255
    config['model'] = "model_digit"
    config['model_path'] = "poison_crafting/pretrained_models/model_digit_MNIST20000.pth"
    config['num_steps'] = 40
    config['step_size'] = 2 / 255
    config['poison_setups'] = "poison_setups/cifar10_transfer_learning.pickle"
    config['poisons_path'] = "poison_examples/clbd_poisons_digits"
    config['pretrain_dataset'] = "digits"
    config['normalize'] = False
    config['patch_size'] = 5
    config['image_size'] = 28
    config['setup_idx'] = 0
    config['iteration'] = 20000
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config["target_class"] = 4
    config["poison_label"] = 5
    config["test_poison_label"] = [i for i in range(10) if i != config["poison_label"]]
    config["test_poison_ratio"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    config['ma'] = ma

    source_labels_copy = source_labels.cpu().numpy().copy()
    test_labels_copy = test_labels.cpu().numpy().copy()
    ind = np.argwhere(source_labels_copy == config["poison_label"])
    poison_num = int(len(list(ind.squeeze())) * poison_ratio)
    print("poison num", poison_num)
    config["base_indices"] = random.sample(list(ind.squeeze()), poison_num)
    poison_num_test = 0
    config["target_img_idx"] = []
    for i in config["test_poison_label"]:
        ind = np.argwhere(test_labels_copy == i)
        poison_num = int(len(list(ind.squeeze())) * config["test_poison_ratio"][i])
        poison_num_test += poison_num
        target_ind = random.sample(list(ind.squeeze()), poison_num)
        config["target_img_idx"].extend(target_ind)
    print("poison num test", poison_num_test)


    #print("base_indices", config["base_indices"])
    #print("target_img_idx", config["target_img_idx"])

    if os.path.exists(config['model_path']):
        print("The pretrain model already exsit!")
    else:
        print("get the pretrain model!")
        net = Model_digit().to(config['device'])
        optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
        train_loader = DataLoader(SimpleDataSet(source_samples,source_labels), batch_size=32, shuffle=True)
        len_train = len(train_loader)
        net.train()
        for i in range(config['iteration']):
            if i % len_train == 0:
                iter_source = iter(train_loader)
            inputs_source, label_source = iter_source.next()

            data, target = inputs_source.to(config["device"]), label_source.to(config["device"])

            # forward
            _, y = net(data)
            # backward

            optimizer.zero_grad()
            loss = F.cross_entropy(y, target)

            loss.backward()

            optimizer.step()
            if i%100 == 0:
                print("iteration {}: loss {}".format(i,loss))
        torch.save(net.state_dict(), config['model_path'])
        print("saving pretrain model!")

    source_samples = source_samples.cpu()
    source_labels = source_labels.cpu()
    trainset = SimpleDataSet(source_samples, source_labels)

    test_samples, test_labels = clbd(config, source_samples,source_labels, test_samples, test_labels)
    # load the poisons and their indices within the training set from pickled files
    with open(os.path.join(config['poisons_path'], "poisons.pickle"), "rb") as handle:
        poison_tuples = pickle.load(handle)
        print(len(poison_tuples), " poisons in this trial.")
        poisoned_label = poison_tuples[0][1]
    with open(os.path.join(config['poisons_path'], "base_indices.pickle"), "rb") as handle:
        poison_indices = pickle.load(handle)
    # get the dataset
    trainset_poison = PoisonedDataset(
        trainset, poison_tuples, None, None, poison_indices
    )
    #show poison train sample
    #count = 0
    #for j in range(len(trainset_poison)):
    #    if trainset_poison[j][2] == 1:
    #        print("show poison image", show_image(trainset_poison[j][0]))
    #        print("show poison label", trainset_poison[j][1], trainset_poison[j][2])
    #        count += 1
    #        if count == 3:
    #            break



    # get the target image from pickled file
    with open(os.path.join(config['poisons_path'], "target.pickle"), "rb") as handle:
        target_img_tuple = pickle.load(handle)
        target_class = target_img_tuple[1]
        if len(target_img_tuple) == 4:
            patch = target_img_tuple[2] if torch.is_tensor(target_img_tuple[2]) else \
                torch.tensor(target_img_tuple[2])


            startx, starty = target_img_tuple[3]
            target_img_pil = target_img_tuple[0]
            h, w = target_img_pil.shape[2], target_img_pil.shape[3]

            if starty + config['patch_size'] > h or startx + config['patch_size'] > w:
                print(
                    "Invalid startx or starty point for the patch. Exiting from poison_test.py."
                )
                sys.exit()

            target_img_tensor = target_img_pil
            target_img_tensor[:, :, starty: starty + config['patch_size'],
            startx: startx + config['patch_size']] = patch

    #get source samples and source labels
    X_poisoned = np.stack([trainset_poison[i][0] for i in range(len(trainset_poison))], 0)
    Y_poisoned = np.stack([trainset_poison[i][1] for i in range(len(trainset_poison))], 0)

        #target_img = transform_test(target_img_pil)

    poison_perturbation_norms = compute_perturbation_norms(
        poison_tuples, trainset, poison_indices
    )


    #show poison test sample
    #count = 0
    #for j in range(target_img_tensor.shape[0]):
    #    print("show poison image", show_image(target_img_tensor[j]))
    #    print("show poison label", target_class[j])
    #    count += 1
    #    if count == 3:
    #       break

    # the limit is '8/255' but we assert that it is smaller than 9/255 to account for PIL
    # truncation.
    #assert max(poison_perturbation_norms) - config['epsilon'] < 1e-5, "Attack not clean label!"
    return X_poisoned, Y_poisoned, test_samples, test_labels, target_img_tensor, target_class






